import torch
import torch.nn as nn
from model_latlon.vars import get_constant_vars, get_additional_vars, N_ADDL_VARS
from model_latlon.codec2d import EarthConvEncoder2d
from model_latlon.transformer3d import SlideLayers3D, posemb_sincos_3d
from model_latlon.primatives2d import southpole_pad2d, call_checkpointed
from model_latlon.primatives3d import earth_pad3d

class ResConvEncoder(nn.Module):
    def __init__(self, mesh, config):
        super(ResConvEncoder, self).__init__()
        self.mesh = mesh
        self.config = config
        assert self.mesh.n_levels % 4 == 0, f"n_levels must be divisible by 4 but got {self.mesh.n_levels}"
        
        data,cvars = get_constant_vars(mesh)
        data = southpole_pad2d(data.unsqueeze(0).permute(0,3,1,2))
        self.register_buffer('const_data', data)

        self.total_sfc = self.mesh.n_sfc_vars + N_ADDL_VARS + len(cvars)
        
        self.pr_encoder = nn.Conv3d(in_channels=self.mesh.n_pr_vars, out_channels=self.config.latent_size, kernel_size=(4,8,8), stride=(4,8,8))
        self.sfc_encoder = EarthConvEncoder2d(self.total_sfc, conv_dims=[96,192,512,self.config.latent_size], affine=self.config.affine, use_pole_convs=config.use_pole_convs)
        self.tr = SlideLayers3D(dim=self.config.latent_size, depth=self.config.encdec_tr_depth,num_heads=self.config.num_heads,window_size=self.config.window_size, checkpoint_type=self.config.checkpoint_type)

    def surface_encoding(self, x, t0):
        x_sfc = southpole_pad2d(x[:,:,:,self.mesh.n_pr:].permute(0,3,1,2)) # the permute is to put channels in the 2nd dim, needed for convs
        addl = southpole_pad2d(get_additional_vars(t0).permute(0,3,1,2)) # addl: additional vars which change with time that we can compute purely from time, things like solar radiation

        x_sfc = torch.concat((x_sfc,addl,self.const_data),dim=1) # const data is stuff like elevation, soil types
        x_sfc = self.sfc_encoder(x_sfc)
        return x_sfc

    def pressure_encoding(self, x):
        B,H,W,C = x.shape
        D = self.mesh.n_levels
        Cpr = self.mesh.n_pr_vars
        
        # Reshape to gather only pressure variables
        x_pr = x[:,:,:,:D*Cpr]
        # [B,H,W,D*Cpr] -> [B,H,W,Cpr,D] -> [B,Cpr,D,H,W] for convs
        x_pr = x_pr.reshape(B,H,W,Cpr,D).permute(0,3,4,1,2) 
        
        x_pr = self.pr_encoder(x_pr)
        
        return x_pr
    
    def embed(self, x_pr, x_sfc):
        x_pr = x_pr[:,:,:, :90, :]
        x_sfc = x_sfc[:,:,:90]
        x = torch.concat((x_pr.permute(0,2,3,4,1), x_sfc.permute(0,2,3,1)[:,None]),dim=1) # x_pr: B,C,D,H,W -> B,D,H,W,C
                                                                                          # x_sfc: B,C,H,W -> B,H,W,C -> B,1,H,W,C
                                                                                          # then we stack them on dim 1 (depth), with sfc at the end

        B, Dl, Hl, Wl, C = x.shape
        if self.config.tr_embedding == 'sincos':
            pe = posemb_sincos_3d(((0, Dl, Hl, Wl, C), x.device, x.dtype)).unsqueeze(0).reshape(B,Dl,Hl,Wl,C)
            x = x + pe
        
        _, _, _, W, _ = x.shape
        wpad = self.config.window_size[2]//2
        Wp = W + wpad*2
        x = x.permute(0,4,1,2,3) # To conv-style
        x = earth_pad3d(x,(0,0,wpad)) # we just need to pad the longitude dim for wrapping around earth
        x = x.permute(0,2,3,4,1)
        return x

    def forward(self, x, t0):
        """
        Forward pass for the ResConvEncoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, H, W, C) where B is the batch size, H is the height, W is the width, and C is the number of channels.
            t0 (int): timestamp for the input data

        Returns:
            x (torch.Tensor): Output latent space tensor 
        """
        x_pr = call_checkpointed(self.pressure_encoding, x)
        x_sfc = call_checkpointed(self.surface_encoding, x, t0)
        x = call_checkpointed(self.embed, x_pr, x_sfc)
        x = self.tr(x)
        return x