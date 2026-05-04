import os

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ForensicHubMesorch(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.input_rgb_range = float(getattr(args, "rgb_range", 1))
        self.input_size = int(
            _first_not_none(
                getattr(args, "forensichub_mesorch_input_size", None),
                os.environ.get("FORENSICHUB_MESORCH_INPUT_SIZE"),
                256,
            )
        )
        self.resize_input = _env_flag("FORENSICHUB_MESORCH_RESIZE_INPUT", True)
        self.normalize_input = _env_flag("FORENSICHUB_MESORCH_NORMALIZE_INPUT", True)
        conv_pretrain = _env_flag("FORENSICHUB_MESORCH_CONV_PRETRAIN", False)

        seg_pretrain_path = (
            _first_not_none(
                getattr(args, "forensichub_mesorch_seg_pretrain", None),
                os.environ.get("FORENSICHUB_MESORCH_SEG_PRETRAIN"),
                self._default_seg_pretrain_path(),
            )
            or None
        )
        if seg_pretrain_path and not os.path.isfile(seg_pretrain_path):
            print("ForensicHub Mesorch SegFormer pretrain not found: {}".format(seg_pretrain_path))
            print("ForensicHub Mesorch will train SegFormer branch from random initialization.")
            seg_pretrain_path = None

        try:
            from .mesorch_model import Mesorch
        except ModuleNotFoundError as exc:
            if exc.name == "timm":
                raise ModuleNotFoundError(
                    "ForensicHub Mesorch requires timm, matching the official ForensicHub/IMDLBenCo code. "
                    "Install it with: pip install timm"
                ) from exc
            raise

        print("==> Building ForensicHub Mesorch model...")
        print("ForensicHub Mesorch input size: {}".format(self.input_size))
        print("ForensicHub Mesorch input normalization: {}".format("on" if self.normalize_input else "off"))
        print("ForensicHub Mesorch ConvNeXt pretrain: {}".format("on" if conv_pretrain else "off"))
        self.model = Mesorch(seg_pretrain_path=seg_pretrain_path, conv_pretrain=conv_pretrain)

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

    def _default_seg_pretrain_path(self):
        candidate = os.path.join(os.path.dirname(__file__), "pretrain", "mit_b3.pth")
        if os.path.isfile(candidate):
            return candidate
        return ""

    def _prepare_image(self, x):
        if x.dim() != 4:
            raise ValueError("ForensicHub Mesorch expects BCHW input, got {}".format(tuple(x.shape)))

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
            raise ValueError("ForensicHub Mesorch mask must be BCHW or BHW, got {}".format(tuple(label.shape)))

        label = label.to(device=device, dtype=dtype)
        if label.max().detach() > 1:
            label = label / 255.0
        label = label.clamp(0, 1)

        mask = 1 - label
        if mask.shape[-2:] != (512, 512):
            mask = F.interpolate(mask, size=(512, 512), mode="nearest")
        return mask

    def _predict_only(self, image):
        mask_logits = self.model.forward_features(image)
        mask_logits = self.model.resize(mask_logits)
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
            label = (mask.flatten(1).max(dim=1).values > 0.5).to(image.dtype)
            output_dict = self.model(image=image, mask=mask, label=label)
            output_dict["target_mask"] = mask

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
