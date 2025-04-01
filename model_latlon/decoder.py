import torch
import torch.nn as nn
import pickle
import numpy as np
import os
import json

from eval import eval_rms_train
from model_latlon.data import get_constant_vars, N_ADDL_VARS
from model_latlon.transformer3d import SlideLayers3D
from model_latlon.primatives2d import southpole_pad2d, southpole_unpad2d
from utils import CONSTS_PATH, ORANGE, load_state_norm
from model_latlon.codec2d import EarthConvDecoder2d
from model_latlon.codec3d import EarthConvDecoder3d
from model_latlon.primatives2d import southpole_pad2d, southpole_unpad2d, call_checkpointed
from model_latlon.primatives3d import southpole_pad3d, southpole_unpad3d

# General decoder type (on the edge of being boilerplate, but definitely necessary as it prevents functions from not being defined)
class Decoder(nn.Module):
    def __init__(self, decoder_name):
        super(Decoder, self).__init__()
        self.decoder_name = decoder_name
        
        self.default_decoder_loss_weight = 1
        self._check_log_information()
    
    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError(f"You are required to implement a compute_loss function for your decoder {self.decoder_name}")
    
    def compute_errors(self, *args, **kwargs):
        raise NotImplementedError(f"You are required to implement a compute_errors function for your decoder {self.decoder_name}")
    
    def log_information(self, *args, **kwargs):
        return 
    
    def _check_log_information(self):
        subclass_method = getattr(self.__class__, 'log_information', None)
        parent_method = getattr(Decoder, 'log_information', None)
        if subclass_method is parent_method: print(ORANGE(f"WARNING: log_information function is not defined for {self.decoder_name}"))

def load_matrices(mesh):
    state_norm, state_norm_matrix = load_state_norm(mesh.wh_lev, mesh)
    
    # nan_mask is only False for nan values eg. all land for sstk (like the last few rows)
    nan_mask_dict = pickle.load(open(f"{CONSTS_PATH}/nan_mask.pkl", 'rb'))
    nan_mask = np.ones((len(mesh.lats), len(mesh.lons), mesh.n_vars), dtype=bool)
    # Apply the appropriate mask for each variable
    for i, var_name in enumerate(mesh.sfc_vars):
        if var_name in nan_mask_dict:
            nan_mask[:,:,mesh.n_pr + i] = nan_mask_dict[var_name]
    
    return state_norm, torch.from_numpy(state_norm_matrix), nan_mask

def gather_variable_weights(mesh):
    default_variable_weight = 2.5
    
    variable_weights = json.load(open(f"norm/variable_weights.json", 'r'))
    
    # Output should be a tensor with shape (n_pr_vars * n_levels + n_sfc_vars)
    output_weights = []
    
    for var_name in mesh.pressure_vars:
        if var_name not in variable_weights: raise Exception(f"Pressure variable {var_name} not found in variable_weights.json. You are required to place something there.")
        assert isinstance(variable_weights[var_name], list), f"Pressure variable {var_name}'s variable weights must be a list in variable_weights.json"
        for level in mesh.levels:
            variable_level_index = variable_weights['levels'].index(level)
            variable_weight = variable_weights[var_name][variable_level_index]
            output_weights.append(variable_weight)
    
    for var_name in mesh.sfc_vars:
        if var_name not in variable_weights: 
            variable_weight = default_variable_weight # Default value
            assert var_name != 'zeropad', "im trying to load a weight for zeropad, this is sus and it feels like ur using the wrong mesh. https://chat.windbornesystems.com/#narrow/stream/201-tech-dlnwp/topic/loss.20weights/near/6939952"
            print(ORANGE(f"😵‍💫😵‍💫😵‍💫 ACHTUNG!!! No variable weight found for {var_name}. Defaulting to {default_variable_weight}. Make sure you want to do this batman..."))
            #raise Exception(f"Surface variable {var_name} not found in variable_weights.json. You are required to place something there.")
        else:
            variable_weight = variable_weights[var_name]
        assert not isinstance(variable_weight, list), f"Surface variable {var_name} must be a scalar value in variable_weights.json"
        output_weights.append(variable_weight)
        
    # Put it in a (B, N1, N2, D) shape to make broadcasting more obvious
    output_weights = torch.tensor(output_weights)
    return output_weights[np.newaxis, np.newaxis, np.newaxis, :]

def gather_geospatial_weights(mesh):
    _, Lats = np.meshgrid(mesh.lons, mesh.lats)
     
    def calc_geospatial_weight(lats):
        F = torch.FloatTensor
        weights = np.cos(lats * np.pi/180)

        # Where we want the parabola to start for the pole weights
        boundary = 50 
        # Weight at poles
        top_of_parabola = 0.3

        progress = np.arange(boundary) / (boundary - 1) # Weights for top and bottom pixels
        parabola = top_of_parabola * (1 - progress) ** 2
        weights[:boundary] += parabola[:, np.newaxis]
        weights[-boundary:] += parabola[::-1, np.newaxis]

        return F(weights)

    output_weights = calc_geospatial_weight(Lats)
    
    # Put it in a (B, N1, N2, D) shape to make broadcasting more obvious
    return output_weights[np.newaxis, :, :, np.newaxis]

# Loss function typically used for decoders operating on LatLonGrids
# Written here to reduce code duplication 
def default_compute_loss(self, y_gpu, yt_gpu):
    # Actual loss
    loss = torch.abs(y_gpu - yt_gpu)
    
    # Weights for loss
    combined_weight = self.geospatial_loss_weight * self.variable_loss_weight 
    weights = self.decoder_loss_weight * combined_weight / torch.sum(combined_weight)
    
    return torch.sum(loss * weights)

# This is also on life support
# Can further simplify this by removing DDP and random_time_subset
# Check out https://github.com/windborne/deep/pull/28 for more details
# Compute errors typically used for decoders operating on LatLonGrids
# Written here to reduce code duplication
def default_compute_errors(self, y_gpu, yt_gpu, trainer=None):
    B, N1, N2, D = y_gpu.shape
    nL = self.mesh.n_levels
    nP = self.mesh.n_pr_vars
    nPL = self.mesh.n_pr; assert nPL == nPL
    nS = self.mesh.n_sfc_vars
    eval_shape = (B,N1,N2,nP,nL)
    to_eval = lambda z : z[...,:nPL].view(eval_shape)
    pred = to_eval(y_gpu)
    actual = to_eval(yt_gpu)
    weight = self.geospatial_loss_weight.squeeze()
    
    pnorm = self.state_norm_matrix[:nPL].view(nP, nL)
    ddp_reduce = trainer.DDP and not trainer.data.config.random_timestep_subset
    #print("pred actual", pred.device, actual.device)
    rms = eval_rms_train(pred, actual, pnorm, weight, keys=self.mesh.pressure_vars, by_level=True, stdout=False, ddp_reduce=ddp_reduce, mesh=self.mesh)

    eval_shape = (B,N1,N2,nS)
    to_eval = lambda z : z[...,nPL:].view(eval_shape)
    pred = to_eval(y_gpu)
    actual = to_eval(yt_gpu)
    pred = pred.to(torch.float32)
    actual = actual.to(torch.float32)
    pnorm = self.state_norm_matrix[nPL:]
    rms.update(eval_rms_train(pred, actual, pnorm, weight, keys=self.mesh.sfc_vars, stdout=False, ddp_reduce=ddp_reduce, mesh=self.mesh))
 
    return rms

# Log information typically used for decoders operating on LatLonGrids
# Written here to reduce code duplication
def default_log_information(self, y_gpu, yt_gpu, rms_dict, writer, dt, n_step, prefix, img_dir=None):
    # Log rms 
    for var_name in self.mesh.pressure_vars + self.mesh.sfc_vars:
        writer.add_scalar(prefix + f"_{dt}/" + var_name, rms_dict[var_name], n_step)
    for var_name in self.mesh.pressure_vars:
        name500 = var_name + "_500"
        writer.add_scalar(prefix + f"_{dt}/" + name500, rms_dict[name500], n_step)


class ResConvDecoder(Decoder):
    def __init__(self, mesh, config, fuck_the_poles=False, decoder_loss_weight=None):
        super(ResConvDecoder, self).__init__(self.__class__.__name__)
        self.mesh = mesh
        self.config = config
        
        # Load norm matrices 
        _, state_norm_matrix, _ = load_matrices(mesh)
        self.register_buffer(f'state_norm_matrix', state_norm_matrix)
       
        # Gather loss weights
        geospatial_loss_weight = gather_geospatial_weights(self.mesh)
        variable_loss_weight = gather_variable_weights(self.mesh)
        self.decoder_loss_weight = decoder_loss_weight if decoder_loss_weight is not None else self.default_decoder_loss_weight
        self.register_buffer(f'geospatial_loss_weight', geospatial_loss_weight)
        self.register_buffer(f'variable_loss_weight', variable_loss_weight)
        
        self.n_convs = 3
        self.fuck_the_poles = fuck_the_poles
        data,cvars = get_constant_vars(mesh)
        nc = len(cvars)
        data = southpole_pad2d(data.unsqueeze(0).permute(0,3,1,2))
        self.register_buffer('const_data_0', data)
        """
        for i in range(3):
            data = southpole_pad2d(F.avg_pool2d(data, kernel_size=2, stride=2))
            self.register_buffer(f'const_data_{i+1}', data)
        """

        self.total_sfc = self.mesh.n_sfc_vars + N_ADDL_VARS + len(cvars)

        self.tr = SlideLayers3D(
            dim=self.config.latent_size, 
            depth=self.config.encdec_tr_depth, 
            num_heads=self.config.num_heads, 
            window_size=self.config.window_size,
            checkpoint_type=self.config.checkpoint_type
        )
        self.sfc_decoder = EarthConvDecoder2d(self.mesh.n_sfc_vars, conv_dims=[self.config.latent_size,512,192,96],skip_dims=[0,0,0,nc], affine=self.config.affine, use_pole_convs=config.use_pole_convs)

        if self.config.oldpr:
            self.pr_decoder = nn.ConvTranspose3d(out_channels=self.mesh.n_pr_vars, in_channels=self.config.latent_size, kernel_size=(4,8,8), stride=(4,8,8))
        else:
            self.pr_decoder = EarthConvDecoder3d(self.mesh.n_pr_vars, conv_dims=[self.config.latent_size]+self.config.pr_dims[::-1], affine=self.config.affine)
        assert len(self.sfc_decoder.up_layers) == self.n_convs
        #self.pr_decoder = nn.ConvTranspose3d(out_channels=self.mesh.n_pr_vars, in_channels=self.config.latent_size, kernel_size=(4,8,8), stride=(4,8,8))
        #self.sfc_decoder = nn.ConvTranspose2d(out_channels=self.mesh.n_sfc_vars, in_channels=self.config.latent_size, kernel_size=(8,8), stride=(8,8))


    def decoder_inner(self, x):
        pass

    def decoder_sfc(self, x):

        x_sfc = x.permute(0,3,1,2)
        if self.config.oldpr:
            x_sfc = (self.sfc_decoder(x_sfc,skips=[None,None,None,self.const_data_0[:,:,:720]]))
        else:
            x_sfc = southpole_unpad2d(self.sfc_decoder(x_sfc,skips=[None,None,None,self.const_data_0]))

        x_sfc = x_sfc.permute(0,2,3,1)
        return x_sfc

    def decoder_pr(self, x):
        x_pr = x.permute(0,4,1,2,3)

        if self.config.oldpr:
            x_pr = (self.pr_decoder(x_pr))
        else:
            x_pr = southpole_unpad3d(self.pr_decoder(x_pr))

        x_pr = x_pr.permute(0,3,4,1,2)
        #x_pr = torch.flatten(x_pr, start_dim=-2) # no!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        return x_pr

    def forward(self, x):
        x = self.tr(x)
        _, _, _, W, _ = x.shape
        wpad = self.config.window_size[2]//2
        x = x[:,:,:,wpad:-wpad,:]

        if self.config.oldpr:
            pr = self.decoder_pr(x[:,:-1]) # TODO: maybe checkpoint this actually
        else:
            xs = []
            for i in range(self.mesh.n_levels // 4):
                #print("inp", x.shape, x[:,i:i+1].shape)
                xs.append(call_checkpointed(self.decoder_pr, x[:,i:i+1], checkpoint_type=self.config.checkpoint_type))
                #print("er", xs[-1].shape)
            pr = torch.cat(xs, axis=-1)
        pr = torch.flatten(pr, start_dim=-2)
        #xs.append(call_checkpointed(self.decoder_sfc, x[:,-1]))
        sfc = call_checkpointed(self.decoder_sfc, x[:,-1], checkpoint_type=self.config.checkpoint_type)
        #print("ersfc", xs[-1].shape)
        y = torch.cat([pr, sfc], axis=-1)
        if self.fuck_the_poles:
            y[:,0,:,:] = y[:,3,:,:]
            y[:,-1,:,:] = y[:,-4,:,:]
        #y = call_checkpointed(self.decoder_inner, x)

        return y

    def compute_loss(self, y_gpu, yt_gpu):
        return default_compute_loss(self, y_gpu, yt_gpu)
    
    def compute_errors(self, y_gpu, yt_gpu, *args, **kwargs):
        return default_compute_errors(self, y_gpu, yt_gpu, *args, **kwargs)
        
    def log_information(self, y_gpu, yt_gpu, rms_dict, writer, dt, n_step, prefix, img_dir=None):
        default_log_information(self, y_gpu, yt_gpu, rms_dict, writer, dt, n_step, prefix, img_dir)
