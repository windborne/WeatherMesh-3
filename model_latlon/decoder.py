import json
import torch
import pickle
import numpy as np
import torch.nn as nn

from model_latlon.codec2d import EarthConvDecoder2d
from model_latlon.transformer3d import SlideLayers3D
from model_latlon.vars import get_constant_vars, N_ADDL_VARS
from model_latlon.primatives2d import southpole_pad2d, call_checkpointed
from utils import ORANGE, load_normalization

class Decoder(nn.Module):
    """
    Base Decoder class for all decoders. 
    
    This abstract class guarantees all WeatherMesh decoders have compute_loss() and compute_errors() functions. 
    The log_information() function is optional as it's only used during training, though it is encouraged. 
    This class also declares helpful defaults (such as the default_decoder_loss_weight).
    When building a new decoder, you should inherit from this class and implement the compute_loss (not-optional), compute_errors (not-optional) and log_information (optional) functions.

    Args:
        decoder_name (str): Name identifier of the decoder. This is used for logging and debugging purposes.
    """
    
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
        """
        Checks if the log_information function is defined for the decoder. If not, it raises a warning.
        """
        subclass_method = getattr(self.__class__, 'log_information', None)
        parent_method = getattr(Decoder, 'log_information', None)
        if subclass_method is parent_method: print(ORANGE(f"WARNING: log_information function is not defined for {self.decoder_name}"))

def load_matrices(mesh):
    """
    Loads the state norm and state norm matrix for the given mesh. Useful for normalizing variables upon output

    Args:
        mesh (Mesh): Mesh object from meshes.py

    Returns:
        norm (dict): Dictionary containing all the normalization parameters (mean and std) with keys for each weather variable
        normalization_matrix_std (torch.Tensor): Tensor containing the standard deviation for each variable
        nan_mask (np.array): Mask for nan values in the data. This is used to ignore nan values in the loss function
    """
    norm, normalization_matrix_std = load_normalization(mesh, with_means=False)
    
    # nan_mask is False for nan values eg. all land for sstk 
    with open("constants/nan_mask.pickle", 'rb') as f:
        nan_mask_dict = pickle.load(f)
        
    nan_mask = np.ones((len(mesh.lats), len(mesh.lons), mesh.n_vars), dtype=bool)
    for i, var_name in enumerate(mesh.sfc_vars):
        if var_name in nan_mask_dict:
            nan_mask[:,:,mesh.n_pr + i] = nan_mask_dict[var_name]
    
    return norm, torch.from_numpy(normalization_matrix_std), nan_mask

def gather_variable_weights(mesh):
    """
    Gathers the variable weights for the given mesh and its variables.
    
    This is used to weight the loss function appropriately for each variable.

    Args:
        mesh (Mesh): Mesh object from meshes.py which contains the variables for which we want weights for

    Returns:
        torch.Tensor: Tensor containing the variable weights for each variable in the mesh. The shape of the tensor is (1, 1, 1, D) where D is the number of variables in the mesh. 
    """
    default_variable_weight = 2.5
    
    with open("constants/variable_weights.json", 'r') as f:
        variable_weights = json.load(f)

    output_weights = [] # Output should be a tensor with shape (n_pr_vars * n_levels + n_sfc_vars) == D
    
    # Gather pressure variable weights
    for pressure_var in mesh.pressure_vars:
        if pressure_var not in variable_weights: raise Exception(f"Pressure variable {pressure_var} not found in constants/variable_weights.json. Pressure variables must have a variable weight.")
        assert isinstance(variable_weights[pressure_var], list), f"Pressure variable {pressure_var}'s variable weights must be a list in constants/variable_weights.json"
        for level in mesh.levels:
            level_index = variable_weights['levels'].index(level)
            variable_weight = variable_weights[pressure_var][level_index]
            output_weights.append(variable_weight)
    
    # Gather surface variable weights
    for surface_var in mesh.sfc_vars:
        if surface_var not in variable_weights: 
            variable_weight = default_variable_weight
            assert surface_var != 'zeropad', "Loading a weight for zeropad which isn't expected. Perhaps you're using the wrong mesh?"
            print(ORANGE(f"WARNING: No variable weight found for {surface_var}. Defaulting to {default_variable_weight}."))
        else:
            variable_weight = variable_weights[surface_var]
        assert not isinstance(variable_weight, list), f"Surface variable {surface_var} must be a scalar value in variable_weights.json"
        output_weights.append(variable_weight)
        
    # Put it in a (1, 1, 1, D) shape to make broadcasting to (B, H, W, D) more obvious
    output_weights = torch.tensor(output_weights)
    return output_weights[None, None, None, :]

def gather_geospatial_weights(mesh):
    """
    Gathers the geospatial weights for the given mesh and its variables.
    
    Geospatial weights are used to weigh the loss function appropriately for each variable based on its geospatial location.

    Args:
        mesh (Mesh): Mesh object from meshes.py which contains the latitudes and longitudes for which we want weights for

    Returns:
        torch.Tensor: Tensor containing the geospatial weights for each variable in the mesh. The shape of the tensor is (1, W, H, 1) where W is the width (longitude) and H is the height (latitude).
    """
    longitudes, latitudes = np.meshgrid(mesh.lons, mesh.lats)
     
    def calculate_geospatial_weight(lats):
        F = torch.FloatTensor
        weights = np.cos(lats * np.pi/180)

        boundary = 50 # Where we want the parabola to start for the pole weights
        top_of_parabola = 0.1 # Weight at poles

        progress = np.arange(boundary) / (boundary - 1) # Weights for top and bottom pixels
        parabola = top_of_parabola * (1 - progress) ** 2
        weights[:boundary] += parabola[:, np.newaxis]
        weights[-boundary:] += parabola[::-1, np.newaxis]

        return F(weights)

    # We only actually scale weights by latitude since longitude should remain uniform across the globe
    output_weights = calculate_geospatial_weight(latitudes)
    
    # Put it in (1, H, W, 1) shape to make broadcasting to (B, H, W, D) more obvious
    return output_weights[np.newaxis, :, :, np.newaxis]

def default_compute_loss(self, y_gpu, yt_gpu):
    """
    Default loss function for decoders operating on LatLonGrids.
    
    This function computes the l1 loss 

    Args:
        y_gpu (torch.Tensor): Predicted y values from the decoder on the GPU. Shape is (B, H, W, D)
        yt_gpu (_type_): Actual y values from data on the GPU. Shape is (B, H, W, D)

    Returns:
        torch.Tensor: A scalar tensor containing the loss value. 
    """
    
    loss = torch.abs(y_gpu - yt_gpu)
    
    combined_weight = self.geospatial_loss_weight * self.variable_loss_weight 
    scaled_weights = self.decoder_loss_weight * combined_weight / torch.sum(combined_weight)
    
    return torch.sum(loss * scaled_weights)

# Compute errors typically used for decoders operating on LatLonGrids
def default_compute_errors(self, *args, **kwargs):
    assert False, "This is not yet open source"

# Log information typically used for decoders operating on LatLonGrids
def default_log_information(self, *args, **kwargs):
    assert False, "This is not yet open source"

class ResConvDecoder(Decoder):
    def __init__(self, mesh, config, decoder_loss_weight=None):
        super(ResConvDecoder, self).__init__(self.__class__.__name__)
        self.mesh = mesh
        self.config = config
        
        # Load norm matrices 
        _, normalization_matrix_std, _ = load_matrices(mesh)
        self.register_buffer(f'normalization_matrix_std', normalization_matrix_std)
       
        # Gather loss weights
        geospatial_loss_weight = gather_geospatial_weights(self.mesh)
        variable_loss_weight = gather_variable_weights(self.mesh)
        self.decoder_loss_weight = decoder_loss_weight if decoder_loss_weight is not None else self.default_decoder_loss_weight
        self.register_buffer(f'geospatial_loss_weight', geospatial_loss_weight)
        self.register_buffer(f'variable_loss_weight', variable_loss_weight)
        
        self.n_convs = 3
        data,cvars = get_constant_vars(mesh)
        nc = len(cvars)
        data = southpole_pad2d(data.unsqueeze(0).permute(0,3,1,2))
        self.register_buffer('const_data_0', data)

        self.total_sfc = self.mesh.n_sfc_vars + N_ADDL_VARS + len(cvars)

        self.tr = SlideLayers3D(
            dim=self.config.latent_size, 
            depth=self.config.encdec_tr_depth, 
            num_heads=self.config.num_heads, 
            window_size=self.config.window_size,
            checkpoint_type=self.config.checkpoint_type
        )
        self.sfc_decoder = EarthConvDecoder2d(self.mesh.n_sfc_vars, conv_dims=[self.config.latent_size,512,192,96],skip_dims=[0,0,0,nc], affine=self.config.affine, use_pole_convs=config.use_pole_convs)

        self.pr_decoder = nn.ConvTranspose3d(out_channels=self.mesh.n_pr_vars, in_channels=self.config.latent_size, kernel_size=(4,8,8), stride=(4,8,8))
        assert len(self.sfc_decoder.up_layers) == self.n_convs

    def decoder_sfc(self, x):

        x_sfc = x.permute(0,3,1,2)
        x_sfc = (self.sfc_decoder(x_sfc,skips=[None,None,None,self.const_data_0[:,:,:720]]))
        
        x_sfc = x_sfc.permute(0,2,3,1)
        return x_sfc

    def decoder_pr(self, x):
        x_pr = x.permute(0,4,1,2,3)

        x_pr = (self.pr_decoder(x_pr))

        x_pr = x_pr.permute(0,3,4,1,2)
        return x_pr

    def forward(self, x):
        x = self.tr(x)
        wpad = self.config.window_size[2]//2
        x = x[:,:,:,wpad:-wpad,:]

        pr = self.decoder_pr(x[:,:-1]) 
        pr = torch.flatten(pr, start_dim=-2)
        sfc = call_checkpointed(self.decoder_sfc, x[:,-1], checkpoint_type=self.config.checkpoint_type)
        y = torch.cat([pr, sfc], axis=-1)
        
        return y

    def compute_loss(self, y_gpu, yt_gpu):
        return default_compute_loss(self, y_gpu, yt_gpu)
    
    def compute_errors(self, *args, **kwargs):
        return default_compute_errors(self, *args, **kwargs)
        
    def log_information(self, *args, **kwargs):
        default_log_information(self, *args, **kwargs)
