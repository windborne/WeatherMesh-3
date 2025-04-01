import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from model_latlon.primatives2d import *

def earth_pad3d(x, pad):
    """
    :param x: (N, C, D, H, W) input tensor
    :param pad: (D_pad, H_pad, W_pad) tuple of padding
    :return: padded tensor
    """
    D_pad, H_pad, W_pad = pad
    assert D_pad == 0, "I'm sus that we ever want to depth pad tbh"
    # Pad depth dimension with reflection padding
    x = F.pad(x, (0, 0, 0, 0, D_pad, D_pad), mode='reflect')  # Pad D
    # Pad latitude (H) dimension with constant padding
    x = F.pad(x, (0, 0, H_pad, H_pad, 0, 0), mode=POLE_PAD_MODE)  # Pad H
    # Pad longitude (W) dimension with circular padding
    x = F.pad(x, (W_pad, W_pad, 0, 0, 0, 0), mode='circular')  # Pad W
    return x

def earth_wrap3d(x, target_W, Wpad):
    """
    Adjusts the horizontal wrapping for a tensor after it has gone through a conv transpose to go up.
    """
    B, C, D, H, W = x.shape
    assert W > target_W, "Input width must be larger than target width."
    RWpad = W - target_W - Wpad
    if Wpad == 1: assert RWpad == 0
    elif Wpad > 1: assert RWpad > 0, "Invalid padding sizes."

    xout = x[:, :, :, :, Wpad:target_W+Wpad]  # Select required width
    xout[:, :, :, :, -Wpad:] += x[:, :, :, :, :Wpad]  # Wrap left to right
    if RWpad > 0:
        xout[:, :, :, :, :RWpad] += x[:, :, :, :, -RWpad:]  # Wrap right to left
    return xout

def southpole_pad3d(x):
    D, H, W = x.shape[2], x.shape[3], x.shape[4]
    assert H % 2 == 0
    x = F.pad(x, (0, 0, 0, 1, 0, 0), mode=POLE_PAD_MODE)  # Pad H dimension
    return x

def southpole_unpad3d(x):
    D, H, W = x.shape[2], x.shape[3], x.shape[4]
    assert H % 2 == 1
    x = x[:, :, :, :-1, :]
    return x

class EarthResBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, intermediate_channels=None, time_emb_dim=None, dropout=0.0, kernel_size=(1,3,3),affine=True, use_pole_convs=False):
        super(EarthResBlock3d, self).__init__()
        assert kernel_size[0] == 1, "Depth kernel size must be 1"
        self.kernel_size = kernel_size
        self.use_pole_convs = use_pole_convs
        
        if intermediate_channels is None:
            intermediate_channels = out_channels
        self.norm1 = nn.InstanceNorm3d(in_channels,affine=affine)
        self.conv1 = nn.Conv3d(in_channels, intermediate_channels, kernel_size=kernel_size, padding=0)
        if use_pole_convs: self.pole_conv1 = nn.Conv3d(in_channels, intermediate_channels, kernel_size=1, padding=0)
        self.norm2 = nn.InstanceNorm3d(intermediate_channels,affine=affine)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(intermediate_channels, out_channels, kernel_size=kernel_size, padding=0)
        if use_pole_convs: self.pole_conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=1, padding=0)

        self.nin_shortcut = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        if time_emb_dim is not None:
            self.time_mlp = nn.Linear(time_emb_dim, out_channels)

    def forward(self, x, t=None):
        dpad = 0  # For depth, we are kernel size = 1, so no padding
        hpad, wpad = [k//2 for k in self.kernel_size[1:]] 
        h = self.norm1(x)
        h = F.silu(h)
        h = earth_pad3d(h, (dpad, hpad, wpad))
        h = self.conv1(h)
        if self.use_pole_convs:
            h[:,:,:,0:1,:] = self.pole_conv1(h[:,:,:,0:1,:])
            h[:,:,:,-1:,:] = self.pole_conv1(h[:,:,:,-1:,:])

        if t is not None:
            time_emb = self.time_mlp(t)
            time_emb = time_emb[:, :, None, None, None]  # (N, C, 1, 1, 1)
            h = h + time_emb

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = earth_pad3d(h, (dpad, hpad, wpad))
        h = self.conv2(h)
        if self.use_pole_convs:
            h[:,:,:,0:1,:] = self.pole_conv2(h[:,:,:,0:1,:])
            h[:,:,:,-1:,:] = self.pole_conv2(h[:,:,:,-1:,:])

        return h + self.nin_shortcut(x)

class EarthConvDown3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2,3,3), stride=(2,2,2)):
        super(EarthConvDown3d, self).__init__()
        assert all(k % 2 == 1 for k in kernel_size[1:]), "Kernel size for H and W must be odd"
        assert stride[1:] == (2,2), "Stride must be (X,2,2)"
        assert stride[0] == 2 or stride[0] == 1, "Stride must be 2 or 1 for depth"
        assert kernel_size[0] == stride[0], "Depth stride must be the same as the kernel size"
        self.kernel_size = kernel_size
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        _, _, D, H, W = x.shape
        dpad = 0  # For depth, we are doing stride = kernel size, so no padding
        hpad, wpad = [k//2 for k in self.kernel_size[1:]]
        h = earth_pad3d(x, (dpad, hpad, wpad))
        h = self.conv(h)
        return h

class EarthConvUp3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2,3,3), stride=(2,2,2)):
        super(EarthConvUp3d, self).__init__()
        assert all(k % 2 == 1 for k in kernel_size[1:]), "Kernel size for H and W must be odd"
        assert stride[1:] == (2,2), "Stride must be (X,2,2)"
        assert stride[0] == 1 or stride[0] == 2, "Stride must be 1 or 2"
        assert kernel_size[0] == stride[0], "Depth stride must be the same as the kernel size"
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0, bias=False)

    def forward(self, x):
        B, C, Ds, Hs, Ws = x.shape
        D = Ds * self.stride[0] 
        H = Hs * self.stride[1] - (Hs % 2) 
        W = Ws * self.stride[2]
        dpad = 0  # For depth, we are doing stride = kernel size, so no padding
        hpad, wpad = [k//2 for k in self.kernel_size[1:]]
        h = self.conv(x)
        h = earth_wrap3d(h, W, wpad)
        h_unpad = h[:, :, dpad:D+dpad, hpad:H+hpad, :]
        assert h_unpad.shape[2:] == (D, H, W), h_unpad.shape
        return h_unpad

def weights1(module):
    for p in module.parameters():
        nn.init.constant_(p, 1)
    return module

def imsave3(path,t,i=0):
    print("-----")
    print(f"{path}: {[x for x in t.shape]}")
    print(f"{path}: {[int(x) for x in torch.unique(t).tolist()]}")
    plt.imsave(path,t[0,0,i].detach().numpy())
    print("----")

def find_periodicity_3d(tensor):
    if tensor.dim() != 5:
        raise ValueError("Input tensor must be 5-dimensional (B,C,D,H,W)")
        
    B, C, D, H, W = tensor.shape

    def find_period(vec):
        for period in range(1, W // 2 + 1):
            if W % period == 0:
                if torch.all(torch.eq(vec.view(-1, period), vec[:period])):
                    return period
        return W  # If no periodicity found, return the full length

    periodicities = torch.zeros((B, C, D, H), dtype=torch.int64, device=tensor.device)
    for b in range(B):
        for c in range(C):
            for d in range(D):
                for h in range(H):
                    periodicities[b, c, d, h] = find_period(tensor[b, c, d, h])
    return periodicities[0, 0].cpu().numpy()

