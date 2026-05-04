import math
import os
from collections import OrderedDict
from collections.abc import Mapping
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ('1', 'true', 'yes', 'y', 'on')


def _to_2tuple(value):
    if isinstance(value, tuple):
        return value
    return value, value


def _trunc_normal_(tensor, std=0.02):
    if hasattr(nn.init, 'trunc_normal_'):
        return nn.init.trunc_normal_(tensor, std=std)
    return nn.init.normal_(tensor, std=std)


def _drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return _drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            _trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x, height, width):
        x = self.fc1(x)
        x = self.dwconv(x, height, width)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim {} should be divided by num_heads {}.'.format(dim, num_heads)

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            _trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x, height, width):
        batch, num_tokens, channels = x.shape
        q = self.q(x).reshape(batch, num_tokens, self.num_heads, channels // self.num_heads)
        q = q.permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(batch, channels, height, width)
            x_ = self.sr(x_).reshape(batch, channels, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(batch, -1, 2, self.num_heads, channels // self.num_heads)
        else:
            kv = self.kv(x).reshape(batch, -1, 2, self.num_heads, channels // self.num_heads)

        kv = kv.permute(2, 0, 3, 1, 4)
        key, value = kv[0], kv[1]

        attn = (q @ key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ value).transpose(1, 2).reshape(batch, num_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            _trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x, height, width):
        x = x + self.drop_path(self.attn(self.norm1(x), height, width))
        x = x + self.drop_path(self.mlp(self.norm2(x), height, width))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = _to_2tuple(img_size)
        patch_size = _to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H = img_size[0] // patch_size[0]
        self.W = img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            _trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, height, width


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, height, width):
        batch, _, channels = x.shape
        x = x.transpose(1, 2).view(batch, channels, height, width)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MixVisionTransformer(nn.Module):
    def __init__(
        self,
        output_type='label',
        pretrain_path='',
        image_size=256,
        patch_size=4,
        in_chans=3,
        embed_dims=None,
        num_heads=None,
        mlp_ratios=None,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=None,
        sr_ratios=None,
    ):
        super().__init__()
        embed_dims = embed_dims or [64, 128, 320, 512]
        num_heads = num_heads or [1, 2, 5, 8]
        mlp_ratios = mlp_ratios or [4, 4, 4, 4]
        depths = depths or [3, 4, 18, 3]
        sr_ratios = sr_ratios or [8, 4, 2, 1]

        self.depths = depths
        self.output_type = output_type
        self.image_size = image_size

        self.patch_embed1 = OverlapPatchEmbed(
            img_size=image_size,
            patch_size=7,
            stride=4,
            in_chans=in_chans,
            embed_dim=embed_dims[0],
        )
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=image_size // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1],
        )
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=image_size // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2],
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=image_size // 16,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3],
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([
            Block(
                dim=embed_dims[0],
                num_heads=num_heads[0],
                mlp_ratio=mlp_ratios[0],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[0],
            )
            for i in range(depths[0])
        ])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([
            Block(
                dim=embed_dims[1],
                num_heads=num_heads[1],
                mlp_ratio=mlp_ratios[1],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[1],
            )
            for i in range(depths[1])
        ])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([
            Block(
                dim=embed_dims[2],
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[2],
            )
            for i in range(depths[2])
        ])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([
            Block(
                dim=embed_dims[3],
                num_heads=num_heads[3],
                mlp_ratio=mlp_ratios[3],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[3],
            )
            for i in range(depths[3])
        ])
        self.norm4 = norm_layer(embed_dims[3])

        if output_type == 'label':
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(embed_dims[-1], 1),
            )
        elif output_type == 'mask':
            out_channels = embed_dims[-1]
            self.head = nn.Sequential(
                nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 2, out_channels // 4, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 4, 1, kernel_size=1),
                nn.Upsample(size=(image_size, image_size), mode='bilinear', align_corners=False),
            )
        else:
            raise ValueError('Unsupported output_type: {}'.format(output_type))

        self.head.apply(self._init_weights)
        if output_type == 'mask':
            nn.init.normal_(self.head[4].weight, std=1e-3)
            if self.head[4].bias is not None:
                nn.init.constant_(self.head[4].bias, 0)
        self._load_pretrain(pretrain_path)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            _trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def _load_pretrain(self, pretrain_path):
        if not pretrain_path:
            print('ForensicHub Segformer-b3 pretrain: not set, training from random initialization.')
            return
        if not os.path.isfile(pretrain_path):
            print('ForensicHub Segformer-b3 pretrain not found: {}'.format(pretrain_path))
            print('ForensicHub Segformer-b3 will train from random initialization.')
            return

        print('Loading ForensicHub Segformer-b3 pretrain from {}'.format(pretrain_path))
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        state_dict = self._normalize_checkpoint_state(checkpoint)
        incompatible = self.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            print('ForensicHub Segformer-b3 missing pretrain keys: {}'.format(len(incompatible.missing_keys)))
        if incompatible.unexpected_keys:
            print('ForensicHub Segformer-b3 unexpected pretrain keys: {}'.format(len(incompatible.unexpected_keys)))

    def _normalize_checkpoint_state(self, checkpoint):
        if isinstance(checkpoint, Mapping):
            for key in ('state_dict', 'model_state_dict', 'model'):
                value = checkpoint.get(key)
                if isinstance(value, Mapping):
                    checkpoint = value
                    break

        normalized = OrderedDict()
        for key, value in checkpoint.items():
            new_key = key
            for prefix in ('module.', 'backbone.', 'encoder.'):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            if new_key.startswith('head.'):
                continue
            normalized[new_key] = value
        return normalized

    def forward_features(self, x):
        batch = x.shape[0]

        x, height, width = self.patch_embed1(x)
        for block in self.block1:
            x = block(x, height, width)
        x = self.norm1(x)
        x = x.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()

        x, height, width = self.patch_embed2(x)
        for block in self.block2:
            x = block(x, height, width)
        x = self.norm2(x)
        x = x.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()

        x, height, width = self.patch_embed3(x)
        for block in self.block3:
            x = block(x, height, width)
        x = self.norm3(x)
        x = x.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()

        x, height, width = self.patch_embed4(x)
        for block in self.block4:
            x = block(x, height, width)
        x = self.norm4(x)
        x = x.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()

        return x

    def forward_binary_logits(self, x):
        x = self.forward_features(x)
        out = self.head(x)
        if out.dim() == 1:
            out = out.unsqueeze(1)
        return out

    def forward(self, image, *args, **kwargs):
        return self.forward_binary_logits(image)


class Segformerb3(MixVisionTransformer):
    def __init__(self, output_type='label', pretrain_path='', image_size=256):
        super().__init__(
            output_type=output_type,
            pretrain_path=pretrain_path,
            image_size=image_size,
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
        )


class ForensicHubSegformerB3(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.input_rgb_range = float(getattr(args, 'rgb_range', 1))
        self.image_size = int(
            getattr(args, 'forensichub_image_size', None)
            or os.environ.get('FORENSICHUB_IMAGE_SIZE')
            or 256
        )
        self.resize_input = _env_flag('FORENSICHUB_RESIZE_INPUT', False)
        self.normalize_input = _env_flag('FORENSICHUB_NORMALIZE_INPUT', True)

        pretrain_path = (
            getattr(args, 'forensichub_segformer_pretrain', None)
            or os.environ.get('FORENSICHUB_SEGFORMER_B3_PRETRAIN')
            or self._default_pretrain_path()
        )

        print('==> Building ForensicHub Segformer-b3 model...')
        print('ForensicHub input normalization: {}'.format('on' if self.normalize_input else 'off'))
        self.model = Segformerb3(
            output_type='mask',
            pretrain_path=pretrain_path,
            image_size=self.image_size,
        )

        self.register_buffer(
            'image_mean',
            torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            'image_std',
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )

    def _default_pretrain_path(self):
        candidate = os.path.join(os.path.dirname(__file__), 'pretrain', 'mit_b3.pth')
        if os.path.isfile(candidate):
            return candidate
        return ''

    def _prepare_inputs(self, x):
        if x.dim() != 4:
            raise ValueError('ForensicHub Segformer-b3 expects BCHW input, got {}'.format(tuple(x.shape)))

        x = x.float()
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.size(1) > 3:
            x = x[:, :3, :, :]

        if self.input_rgb_range != 1:
            x = x / self.input_rgb_range
        x = x.clamp(0, 1)

        if self.resize_input and x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)

        if self.normalize_input:
            x = (x - self.image_mean) / self.image_std
        return x

    def forward(self, x, y=None):
        output_size = x.shape[-2:]
        mask_logits = self.model(self._prepare_inputs(x))
        if mask_logits.shape[-2:] != output_size:
            mask_logits = F.interpolate(mask_logits, size=output_size, mode='bilinear', align_corners=False)
        return torch.sigmoid(mask_logits)
