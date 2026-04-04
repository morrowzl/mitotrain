from funlib.learn.torch.models import UNet


def get_model():
    return UNet(
        in_channels=1,
        num_fmaps=12,
        fmap_inc_factor=5,
        downsample_factors=[(2, 2, 2), (2, 2, 2), (2, 2, 2)],
        kernel_size_down=[[[3, 3, 3], [3, 3, 3]]] * 4,
        kernel_size_up=[[[3, 3, 3], [3, 3, 3]]] * 3,
        num_fmaps_out=1,
        constant_upsample=True,
    )
