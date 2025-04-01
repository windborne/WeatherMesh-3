import torch
import torch.nn as nn
from model_latlon.data import get_constant_vars, get_additional_vars, N_ADDL_VARS
from model_latlon.codec2d import EarthConvEncoder2d
from model_latlon.codec3d import EarthConvEncoder3d
from model_latlon.transformer3d import SlideLayers3D, posemb_sincos_3d, add_posemb, tr_pad
from model_latlon.primatives2d import southpole_pad2d, call_checkpointed
from model_latlon.primatives3d import southpole_pad3d, earth_pad3d

class ResConvEncoder(nn.Module):
    def __init__(self, mesh, config):
        super(ResConvEncoder, self).__init__()
        self.mesh = mesh
        assert self.mesh.n_levels % 4 == 0, "n_levels must be divisible by 4"
        self.config = config
        data,cvars = get_constant_vars(mesh)
        data = southpole_pad2d(data.unsqueeze(0).permute(0,3,1,2))
        self.register_buffer('const_data', data)

        self.total_sfc = self.mesh.n_sfc_vars + N_ADDL_VARS + len(cvars)

        self.sfc_encoder = EarthConvEncoder2d(self.total_sfc, conv_dims=[96,192,512,self.config.latent_size], affine=self.config.affine, use_pole_convs=config.use_pole_convs)

        if self.config.oldpr:
            self.pr_encoder = nn.Conv3d(in_channels=self.mesh.n_pr_vars, out_channels=self.config.latent_size, kernel_size=(4,8,8), stride=(4,8,8))
        else:
            self.pr_encoder = EarthConvEncoder3d(self.mesh.n_pr_vars, conv_dims=self.config.pr_dims + [self.config.latent_size], affine=self.config.affine)

        self.tr = SlideLayers3D(dim=self.config.latent_size, depth=self.config.encdec_tr_depth,num_heads=self.config.num_heads,window_size=self.config.window_size, embedding_module=self.config.embedding_module, checkpoint_type=self.config.checkpoint_type)


    def surface_crap(self, x, t0, x_pr):
        #print("calling surface_crap", x.shape, t0.shape)
        x_sfc = southpole_pad2d(x[:,:,:,self.mesh.n_pr:].permute(0,3,1,2)) # the permute is to put channels in the 2nd dim, needed for convs
        addl = southpole_pad2d(get_additional_vars(t0).permute(0,3,1,2)) # addl: additional vars which change with time that we can compute purely from time, things like solar radiation

        #x_sfc = torch.concat((x_sfc,addl,self.const_data[:,:,:720]),dim=1) # const data is stuff like elevation, soil types
        x_sfc = torch.concat((x_sfc,addl,self.const_data),dim=1) # const data is stuff like elevation, soil types
        #x_sfc = call_checkpointed(self.sfc_encoder,x_sfc)
        x_sfc = self.sfc_encoder(x_sfc)
        #return x_sfc
        return self.embedding_crap(x_pr, x_sfc)

    def pressure_crap(self, x):
        #print("calling pressure_crap", x.shape)
        B,H,W,OHP = x.shape
        D = self.mesh.n_levels
        Cpr = self.mesh.n_pr_vars
        assert OHP % Cpr == 0
        D = OHP // Cpr
        #x_pr = x[:,:,:,:self.mesh.n_pr]
        if not self.config.oldpr:
            x_pr = southpole_pad3d(x.view(B,H,W,Cpr,D).permute(0,3,4,1,2)) # x_pr is shape B,C,D,H,W for convs
        else:
            x_pr = (x.view(B,H,W,Cpr,D).permute(0,3,4,1,2)) # x_pr is shape B,C,D,H,W for convs
        x_pr = self.pr_encoder(x_pr)
        #print("made it", x_pr.shape)
        #import pdb
        #pdb.set_trace()
        return x_pr
    
    def embedding_crap(self, x_pr, x_sfc):
        #print("calling embedding crap", x_pr.shape, x_sfc.shape)
        if self.config.oldpr:
            #print("bef", x_pr.shape, x_sfc.shape)
            #print("aaa", x_pr.shape)
            x_pr = x_pr[:,:,:, :90, :]
            x_sfc = x_sfc[:,:,:90]
            #print("ughhh", x_pr.shape, x_sfc.shape)
        x = torch.concat((x_pr.permute(0,2,3,4,1), x_sfc.permute(0,2,3,1)[:,None]),dim=1) # x_pr: B,C,D,H,W -> B,D,H,W,C
                                                                                          # x_sfc: B,C,H,W -> B,H,W,C -> B,1,H,W,C
                                                                                          # then we stack them on dim 1 (depth), with sfc at the end
        #print("postemb", x.shape)

        B, Dl, Hl, Wl, C = x.shape
        if self.config.tr_embedding == 'sincos':
            pe = posemb_sincos_3d(((0, Dl, Hl, Wl, C), x.device, x.dtype)).unsqueeze(0).reshape(B,Dl,Hl,Wl,C)
            #print("x",x.shape, "pe", pe.shape)
            x = x + pe
        _, _, _, W, _ = x.shape
        wpad = self.config.window_size[2]//2
        Wp = W + wpad*2
        #x0 = x.clone()
        #print("beforepadding", x.shape)
        x = x.permute(0,4,1,2,3) # To conv-style
        x = earth_pad3d(x,(0,0,wpad)) # we just need to pad the longitude dim for wrapping around earth
        x = x.permute(0,2,3,4,1)
        #print("afterpadding", x.shape, (x[:,:,:,:wpad]-x0[:,:,:,-wpad:]).sum(), wpad)
        return x


    def forward(self, x, t0):
        B,H,W,_ = x.shape
        D = self.mesh.n_levels
        Cpr = self.mesh.n_pr_vars

        if self.config.oldpr:
            x_pr = self.pressure_crap(x[..., :D*Cpr]) # TODO: maybe call_checkpointed this actually
        else:
            xv = x[..., :D*Cpr].view(B, H, W, Cpr, D)
            xs = []
            for i in range(D // 4):
                #print("inp", x.shape, x[:,i:i+1].shape)
                st = 4*self.mesh.n_pr_vars*i
                en = 4*self.mesh.n_pr_vars*(i+1)
                #xs.append(call_checkpointed(self.pressure_crap, x[:,:,:,st:en])) # <----- bad
                xs.append(call_checkpointed(self.pressure_crap, xv[:,:,:,:,4*i:4*(i+1)].reshape(B, H, W, Cpr*4), checkpoint_type=self.config.checkpoint_type)) # TODO: optimize once it works
            x_pr = torch.cat(xs, dim=2)

        x = call_checkpointed(self.surface_crap, x, t0, x_pr, checkpoint_type=self.config.checkpoint_type)

        # Q: why is this commented out? why did you merge sfc and embedding?
        # A: matepoint is annoying and the order in which pytorch decides to do the backwards pass is at least somewhat random. this works(TM) otherwise you get checksum asserts
        #x = call_checkpointed(self.embedding_crap, x_pr, x_sfc)
        
        x = self.tr(x)
        return x