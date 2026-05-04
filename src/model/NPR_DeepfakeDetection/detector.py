import os
from collections import OrderedDict
from collections.abc import Mapping

import torch

from .resnet import Bottleneck, ResNet


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ('1', 'true', 'yes', 'y', 'on')


class NPRDeepfakeDetector(ResNet):
    """
    NPR detector adapted for this repository's classification loss.

    The upstream CVPR 2024 implementation trains a one-logit ResNet50 with
    BCEWithLogitsLoss. This subclass keeps that one-logit parameter layout so
    upstream checkpoints load cleanly, then exposes two-class logits in forward
    for this project's CrossEntropyLoss-based training and testing paths.
    """

    def __init__(self, args=None, normalize_input=None):
        super(NPRDeepfakeDetector, self).__init__(
            Bottleneck,
            [3, 4, 6, 3],
            num_classes=1,
        )
        self.args = args
        self.input_rgb_range = float(getattr(args, 'rgb_range', 1))

        if normalize_input is None:
            normalize_input = getattr(args, 'npr_normalize_input', None)
        if normalize_input is None:
            normalize_input = _env_flag('NPR_NORMALIZE_INPUT', True)
        self.normalize_input = bool(normalize_input)

        self.logit_mode = (
            getattr(args, 'npr_logit_mode', None)
            or os.environ.get('NPR_LOGIT_MODE')
            or 'zero'
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

        print('==> Building NPR-DeepfakeDetection model...')
        print('NPR input normalization: {}'.format('on' if self.normalize_input else 'off'))

    def _prepare_inputs(self, x):
        if x.dim() != 4:
            raise ValueError('NPR expects BCHW input, got shape {}'.format(tuple(x.shape)))

        x = x.float()
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.size(1) > 3:
            x = x[:, :3, :, :]

        if self.input_rgb_range != 1:
            x = x / self.input_rgb_range

        if self.normalize_input:
            x = (x - self.image_mean) / self.image_std

        return x

    def _binary_to_two_class_logits(self, binary_logits):
        if binary_logits.dim() == 1:
            binary_logits = binary_logits.unsqueeze(1)

        if binary_logits.size(1) != 1:
            return binary_logits

        if self.logit_mode == 'symmetric':
            return torch.cat((-binary_logits, binary_logits), dim=1)

        return torch.cat((torch.zeros_like(binary_logits), binary_logits), dim=1)

    def forward(self, x, y=None):
        binary_logits = super(NPRDeepfakeDetector, self).forward(self._prepare_inputs(x))
        return self._binary_to_two_class_logits(binary_logits)

    def load_state_dict(self, state_dict, strict=True):
        state_dict = self._normalize_checkpoint_state(state_dict)
        return super(NPRDeepfakeDetector, self).load_state_dict(state_dict, strict=strict)

    def _normalize_checkpoint_state(self, state_dict):
        if isinstance(state_dict, Mapping):
            for key in ('state_dict', 'model_state_dict', 'model'):
                value = state_dict.get(key)
                if isinstance(value, Mapping):
                    state_dict = value
                    break

        normalized = OrderedDict()
        for key, value in state_dict.items():
            new_key = key
            for prefix in ('module.', 'model.', 'backbone.'):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            normalized[new_key] = value

        weight = normalized.get('fc1.weight')
        if torch.is_tensor(weight) and weight.dim() == 2 and weight.size(0) == 2:
            normalized['fc1.weight'] = weight[1:2] - weight[0:1]

        bias = normalized.get('fc1.bias')
        if torch.is_tensor(bias) and bias.dim() == 1 and bias.size(0) == 2:
            normalized['fc1.bias'] = bias[1:2] - bias[0:1]

        return normalized
