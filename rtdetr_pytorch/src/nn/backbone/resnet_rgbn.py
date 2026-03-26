"""
PResNetRGBN — 4-channel (RGB + NIR) variant of PResNet for RT-DETR.

Identical to PResNet in all respects except that the very first conv
in the stem is replaced with one that accepts 4 input channels.

When `pretrained=True`:
 - RGB weights are loaded from the standard PResNet pretrained checkpoint.
 - The NIR channel weights are initialised as the mean of the 3 RGB
   channel weights (a commonly used strategy for spectral transfer).
"""

import torch
import torch.nn as nn
from collections import OrderedDict

from .presnet import PResNet, donwload_url
from .common import ConvNormLayer
from src.core import register

__all__ = ['PResNetRGBN']


@register
class PResNetRGBN(PResNet):
    """
    4-channel (RGBN) backbone built on top of PResNet.

    All constructor arguments are identical to PResNet.
    The `pretrained` flag loads the standard RGB PResNet weights and
    then initialises the extra NIR channel from the mean RGB weights.
    """

    def __init__(
        self,
        depth,
        variant='d',
        num_stages=4,
        return_idx=[0, 1, 2, 3],
        act='relu',
        freeze_at=-1,
        freeze_norm=True,
        pretrained=False,
    ):
        # Build the standard 3-channel backbone first (pretrained=False so we
        # can manipulate weights ourselves before potentially loading them).
        super().__init__(
            depth=depth,
            variant=variant,
            num_stages=num_stages,
            return_idx=return_idx,
            act=act,
            freeze_at=-1,        # freeze after weight surgery
            freeze_norm=False,   # same
            pretrained=False,
        )

        # ── Replace the first conv layer: 3 → 4 input channels ──────────
        # PResNet variant 'd' has a 3-conv stem; the very first conv is
        # self.conv1.conv1_1.  For variant 'b'/'a' it is self.conv1.conv1_1
        # as well (single-conv stem stored under the same key).
        first_conv: ConvNormLayer = self.conv1.conv1_1

        # Build a new ConvNormLayer with in_channels=4; everything else same.
        new_conv = ConvNormLayer(
            ch_in=4,
            ch_out=first_conv.conv.out_channels,
            k=first_conv.conv.kernel_size[0],
            s=first_conv.conv.stride[0],
            act=act,
        )

        if pretrained:
            # Load the pretrained RGB checkpoint
            state = torch.hub.load_state_dict_from_url(donwload_url[depth])
            self.load_state_dict(state)
            print(f'[PResNetRGBN] Loaded PResNet{depth} pretrained weights')

            # Now re-initialise the first conv for 4 channels
            with torch.no_grad():
                rgb_w = first_conv.conv.weight.clone()   # (out, 3, k, k)
                nir_w = rgb_w.mean(dim=1, keepdim=True)  # (out, 1, k, k)
                new_conv.conv.weight = nn.Parameter(
                    torch.cat([rgb_w, nir_w], dim=1)     # (out, 4, k, k)
                )
                # BN parameters copied from the existing first conv's BN
                new_conv.norm.weight = first_conv.norm.weight
                new_conv.norm.bias   = first_conv.norm.bias
                new_conv.norm.running_mean = first_conv.norm.running_mean
                new_conv.norm.running_var  = first_conv.norm.running_var
        else:
            nn.init.kaiming_normal_(
                new_conv.conv.weight, mode='fan_out', nonlinearity='relu'
            )

        # Swap in the new conv
        self.conv1.conv1_1 = new_conv

        # ── Apply requested freezing/norm-freezing ───────────────────────
        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[i])

        if freeze_norm:
            self._freeze_norm(self)

        print(f'[PResNetRGBN] Built 4-channel PResNet{depth}-{variant.upper()} '
              f'(pretrained={pretrained})')