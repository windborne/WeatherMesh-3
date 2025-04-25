import torch.nn.functional as F
from model_latlon.primatives2d import POLE_PAD_MODE

def earth_pad3d(x, pad):
    """
    :param x: (N, C, D, H, W) input tensor
    :param pad: (D_pad, H_pad, W_pad) tuple of padding
    :return: padded tensor
    """
    D_pad, H_pad, W_pad = pad
    assert D_pad == 0, "We should not be performing any depth pad"
    
    # Pad depth dimension with reflection padding
    x = F.pad(x, (0, 0, 0, 0, D_pad, D_pad), mode='reflect')  # Pad D
    # Pad latitude (H) dimension with constant padding
    x = F.pad(x, (0, 0, H_pad, H_pad, 0, 0), mode=POLE_PAD_MODE)  # Pad H
    # Pad longitude (W) dimension with circular padding
    x = F.pad(x, (W_pad, W_pad, 0, 0, 0, 0), mode='circular')  # Pad W
    
    return x