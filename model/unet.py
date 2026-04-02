"""
Stub model for Sprint 0. Replace in Sprint 3 with the funkelab U-Net:
  https://github.com/funkelab/funlib.learn.torch

Sprint 3 target usage:
    from funlib.learn.torch.models import UNet

    def get_model():
        return UNet(
            in_channels=1,
            num_fmaps=12,
            fmap_inc_factor=5,
            downsample_factors=[(2,2,2), (2,2,2), (2,2,2)],
            kernel_size_down=[[[3,3,3],[3,3,3]]] * 4,
            kernel_size_up=[[[3,3,3],[3,3,3]]] * 3,
            num_fmaps_out=1,
            constant_upsample=True,
        )

NOTE: funlib UNet uses valid convolutions — output is spatially smaller than
input. A 132^3 input patch is a reasonable safe starting point; verify output
shape before committing to a patch size in Sprint 2/3.
"""

import torch.nn as nn


class IdentityUNet(nn.Module):
    """Identity placeholder — passes input through unchanged."""

    def forward(self, x):
        return x


def get_model():
    """
    Stub: returns a minimal nn.Module with a forward() that passes input
    through unchanged. Correct input/output shapes for binary segmentation.
    Replace with funkelab UNet in Sprint 3.
    """
    return IdentityUNet()
