import torch
import torch.nn as nn
from model_latlon.primatives2d import EarthResBlock2d, EarthConvDown2d, EarthConvUp2d

class EarthConvEncoder2d(nn.Module):
    def __init__(self, in_channels, conv_dims=[64,192,576,1024], affine=True, use_pole_convs=True):
        super(EarthConvEncoder2d, self).__init__()
        
        # Encoder
        self.down_layers = nn.ModuleList()
        
        for i in range(len(conv_dims) - 1):
            in_dim = conv_dims[i] if i != 0 else in_channels
            mid_dim = conv_dims[i]
            next_dim = conv_dims[i+1]
            
            layer = nn.Sequential(
                EarthResBlock2d(in_dim, mid_dim, affine=affine, use_pole_convs=use_pole_convs),
                EarthResBlock2d(mid_dim, mid_dim, affine=affine, use_pole_convs=use_pole_convs),
                EarthConvDown2d(mid_dim, next_dim)
            )
            self.down_layers.append(layer)
        
        self.latent_dim = conv_dims[-1]

    def forward(self, x):
        # Encoder forward pass
        for down_layer in self.down_layers:
            x = down_layer(x)
        
        return x
    
class EarthConvDecoder2d(nn.Module):
    def __init__(self, out_channels, conv_dims=[1024,576,192,64], skip_dims=[0,0,0,0], affine=True, use_pole_convs=True):
        # output_skip_dim is used for adding the constant data back in at the end
        super(EarthConvDecoder2d, self).__init__()
        
        # Decoder
        self.up_layers = nn.ModuleList()
        
        for i in range(len(conv_dims) - 1):
            in_dim = conv_dims[i]
            out_dim = conv_dims[i+1]
            
            layer = nn.Sequential(
                EarthResBlock2d(in_dim+skip_dims[i], in_dim, affine=affine, use_pole_convs=use_pole_convs),
                EarthResBlock2d(in_dim, in_dim, affine=affine, use_pole_convs=use_pole_convs),
                EarthConvUp2d(in_dim, out_dim),
            )
            self.up_layers.append(layer)
        
        self.final = nn.Sequential(
            EarthResBlock2d(conv_dims[-1] + skip_dims[-1], conv_dims[-1], affine=affine, use_pole_convs=use_pole_convs),
            EarthResBlock2d(conv_dims[-1], out_channels, affine=affine, use_pole_convs=use_pole_convs)
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