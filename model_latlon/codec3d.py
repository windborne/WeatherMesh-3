import torch
import torch.nn as nn
from model_latlon.primatives3d import EarthResBlock3d, EarthConvDown3d, EarthConvUp3d

class EarthConvEncoder3d(nn.Module):
    def __init__(self, in_channels, conv_dims=[32,64,128,256],depth_strides=[2,1,2], affine=True):
        super(EarthConvEncoder3d, self).__init__()
        assert len(depth_strides) == len(conv_dims) - 1
        self.depth_strides = depth_strides

        # Encoder
        self.down_layers = nn.ModuleList()

        for i in range(len(conv_dims) - 1):
            in_dim = conv_dims[i] if i != 0 else in_channels
            mid_dim = conv_dims[i]
            next_dim = conv_dims[i+1]
            ksres = (1,3,3)
            ksup = (depth_strides[i],3,3) if i == 0 else (depth_strides[i],3,3)
            layer = nn.Sequential(
                EarthResBlock3d(in_dim, mid_dim, kernel_size=ksres, affine=affine),
                #EarthResBlock3d(mid_dim, mid_dim, kernel_size=ks),
                EarthConvDown3d(mid_dim, next_dim, stride=(depth_strides[i], 2, 2),kernel_size=ksup),
            )
            self.down_layers.append(layer)

    def forward(self, x):
        # Encoder forward pass
        for down_layer in self.down_layers:
            x = down_layer(x)
        
        return x

class EarthConvDecoder3d(nn.Module):
    def __init__(self, out_channels, conv_dims=[256,128,64,32], skip_dims=[0,0,0,0], depth_strides=[2,1,2], affine=True):
        super(EarthConvDecoder3d, self).__init__()
        
        # Decoder
        self.up_layers = nn.ModuleList()
        
        for i in range(len(conv_dims) - 1):
            in_dim = conv_dims[i]
            out_dim = conv_dims[i+1]
            ksres = (1,3,3) 
            ksup = (depth_strides[i],3,3) 
            layer = nn.Sequential(
                EarthResBlock3d(in_dim+skip_dims[i], in_dim, kernel_size=ksres, affine=affine),
                EarthConvUp3d(in_dim, out_dim, stride=(depth_strides[i], 2, 2), kernel_size=ksup),
            )
            self.up_layers.append(layer)
        
        self.final = nn.Sequential(
            #EarthResBlock3d(conv_dims[-1] + skip_dims[-1], conv_dims[-1],kernel_size=(1,3,3)),
            EarthResBlock3d(conv_dims[-1], out_channels,kernel_size=(1,3,3), affine=affine)
        )

    def forward(self, x, skips=None):
        if skips is None:
            skips = [None]*len(self.up_layers)

        # Decoder forward pass
        for up_layer, skip in zip(self.up_layers, skips):
            x = up_layer(torch.cat([x, skip], dim=1) if skip is not None else x)
        
        # Final output
        x = self.final(torch.cat([x, skips[-1]], dim=1) if skips[-1] is not None else x)
        
        return x