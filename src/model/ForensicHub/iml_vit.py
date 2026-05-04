import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .iml_vit_model import IML_ViT


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "y", "on")


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


class ForensicHubIMLViT(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.input_rgb_range = float(getattr(args, "rgb_range", 1))
        self.input_size = int(
            _first_not_none(
                getattr(args, "forensichub_iml_vit_input_size", None),
                os.environ.get("FORENSICHUB_IML_VIT_INPUT_SIZE"),
                256,
            )
        )
        self.resize_input = _env_flag("FORENSICHUB_IML_VIT_RESIZE_INPUT", True)
        self.normalize_input = _env_flag("FORENSICHUB_IML_VIT_NORMALIZE_INPUT", True)
        self.edge_mask_width = int(
            _first_not_none(
                getattr(args, "forensichub_iml_vit_edge_mask_width", None),
                os.environ.get("FORENSICHUB_IML_VIT_EDGE_MASK_WIDTH"),
                7,
            )
        )
        if self.edge_mask_width % 2 == 0:
            self.edge_mask_width += 1

        vit_pretrain_path = (
            _first_not_none(
                getattr(args, "forensichub_iml_vit_pretrain", None),
                os.environ.get("FORENSICHUB_IML_VIT_PRETRAIN"),
                self._default_pretrain_path(),
            )
            or None
        )
        if vit_pretrain_path and not os.path.isfile(vit_pretrain_path):
            print("ForensicHub IML_ViT MAE pretrain not found: {}".format(vit_pretrain_path))
            print("ForensicHub IML_ViT will train from random initialization.")
            vit_pretrain_path = None

        edge_lambda = int(
            _first_not_none(
                getattr(args, "forensichub_iml_vit_edge_lambda", None),
                os.environ.get("FORENSICHUB_IML_VIT_EDGE_LAMBDA"),
                20,
            )
        )
        predict_head_norm = _first_not_none(
            getattr(args, "forensichub_iml_vit_predict_head_norm", None),
            os.environ.get("FORENSICHUB_IML_VIT_PREDICT_HEAD_NORM"),
            "BN",
        )

        print("==> Building ForensicHub IML_ViT model...")
        print("ForensicHub IML_ViT input size: {}".format(self.input_size))
        print("ForensicHub IML_ViT input normalization: {}".format("on" if self.normalize_input else "off"))
        print("ForensicHub IML_ViT edge mask width: {}".format(self.edge_mask_width))
        self.model = IML_ViT(
            input_size=self.input_size,
            vit_pretrain_path=vit_pretrain_path,
            predict_head_norm=predict_head_norm,
            edge_lambda=edge_lambda,
        )

        self.register_buffer(
            "image_mean",
            torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )

    def _default_pretrain_path(self):
        candidate = os.path.join(os.path.dirname(__file__), "pretrain", "mae_pretrain_vit_base.pth")
        if os.path.isfile(candidate):
            return candidate
        return ""

    def _prepare_image(self, x):
        if x.dim() != 4:
            raise ValueError("ForensicHub IML_ViT expects BCHW input, got {}".format(tuple(x.shape)))

        x = x.float()
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.size(1) > 3:
            x = x[:, :3, :, :]

        if self.input_rgb_range != 1:
            x = x / self.input_rgb_range
        x = x.clamp(0, 1)

        if self.resize_input and x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)

        if self.normalize_input:
            x = (x - self.image_mean) / self.image_std
        return x

    def _prepare_forgery_mask(self, label, device, dtype):
        if label.dim() == 3:
            label = label.unsqueeze(1)
        elif label.dim() == 4 and label.size(1) != 1:
            label = label[:, :1, :, :]
        elif label.dim() != 4:
            raise ValueError("ForensicHub IML_ViT mask must be BCHW or BHW, got {}".format(tuple(label.shape)))

        label = label.to(device=device, dtype=dtype)
        if label.max().detach() > 1:
            label = label / 255.0
        label = label.clamp(0, 1)

        # MFLDCF masks are 1 for pristine/background and 0 for tampered regions.
        mask = 1 - label
        if mask.shape[-2:] != (self.input_size, self.input_size):
            mask = F.interpolate(mask, size=(self.input_size, self.input_size), mode="nearest")
        return mask

    def _edge_mask(self, mask):
        kernel_size = max(int(self.edge_mask_width), 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = torch.zeros((1, 1, kernel_size, kernel_size), device=mask.device, dtype=mask.dtype)
        center = kernel_size // 2
        kernel[0, 0, center:center + 1, :] = 1
        kernel[0, 0, :, center:center + 1] = 1

        binary = (mask > 0.5).to(mask.dtype)
        dilated_fake = (F.conv2d(binary, kernel, stride=1, padding=center) > 0).to(mask.dtype)
        dilated_real = (F.conv2d(1 - binary, kernel, stride=1, padding=center) > 0).to(mask.dtype)
        return (-torch.abs(dilated_real - dilated_fake) + 1 > 0).to(mask.dtype)

    def _predict_only(self, image):
        mask_logits = self.model.forward_features(image)
        mask_logits = F.interpolate(
            mask_logits,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        return {
            "backward_loss": None,
            "pred_mask_logits": mask_logits,
            "pred_mask": torch.sigmoid(mask_logits),
            "pred_label": None,
        }

    def forward(self, x, y=None):
        output_size = x.shape[-2:]
        image = self._prepare_image(x)

        if y is None:
            output_dict = self._predict_only(image)
        else:
            mask = self._prepare_forgery_mask(y, device=image.device, dtype=image.dtype)
            edge_mask = self._edge_mask(mask)
            output_dict = self.model(image=image, mask=mask, edge_mask=edge_mask)
            output_dict["target_mask"] = mask
            output_dict["edge_mask"] = edge_mask

        pred_mask = output_dict.get("pred_mask")
        if torch.is_tensor(pred_mask) and pred_mask.shape[-2:] != output_size:
            output_dict["pred_mask"] = F.interpolate(pred_mask, size=output_size, mode="bilinear", align_corners=False)

        pred_mask_logits = output_dict.get("pred_mask_logits")
        if torch.is_tensor(pred_mask_logits) and pred_mask_logits.shape[-2:] != output_size:
            output_dict["pred_mask_logits"] = F.interpolate(
                pred_mask_logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

        return output_dict
