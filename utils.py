import json
import torch
import numpy as np
from datetime import datetime, timedelta, timezone

# Common pressure levels used for WeatherMesh training 
# len(levels_tiny) = 13
levels_tiny = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000] 
# len(levels_hres) = 20
levels_hres = [10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 850, 900, 925, 950, 1000]
# len(levels_gfs) = 25
levels_gfs = [10, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 925, 950, 975, 1000]
# len(levels_medium) = 28
levels_medium = [10, 30, 50, 70, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 875, 900, 925, 950, 975, 1000]
# len(levels_full) = 37 
levels_full = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]

# Core variables predicted by WeatherMesh (see https://codes.ecmwf.int/grib/param-db/ for more detail)
core_pressure_vars = ["129_z", "130_t", "131_u", "132_v", "133_q"]
core_sfc_vars = ["165_10u", "166_10v", "167_2t", "151_msl"]
vars_with_nans = ["034_sstk", "tc-maxws", "tc-minp"] # Variables that may have NaNs in their data

# Stores the mapping from the number of levels to the actual levels
num2levels = {}
for levels in [levels_tiny, levels_hres, levels_gfs, levels_medium, levels_full]:
    if len(levels) in num2levels: continue
    num2levels[len(levels)] = levels

# Useful defaults to use for colorful printing
def WHITE(text): return f"\033[97m{text}\033[0m"
def RED(text): return f"\033[91m{text}\033[0m"
def ORANGE(text): return f"\033[38;5;214m{text}\033[0m"
def YELLOW(text): return f"\033[93m{text}\033[0m"
def GREEN(text): return f"\033[92m{text}\033[0m"
def BLUE(text): return f"\033[94m{text}\033[0m"
def CYAN(text): return f"\033[96m{text}\033[0m"
def MAGENTA(text): return f"\033[95m{text}\033[0m"

def get_date(date):
    """
    Returns the date in UTC timezone.
    """
    if type(date) == datetime:
        assert date.tzinfo is None or date.tzinfo == timezone.utc, f"get_date() was not designed to be used with non-UTC dates. Got date: {date} with {date.tzinfo}"
        return date.replace(tzinfo=timezone.utc) 
    elif np.issubdtype(type(date), np.number):
        nix = int(date)
        return datetime(1970,1,1,tzinfo=timezone.utc)+timedelta(seconds=nix)
    assert False, f"Unable to parse date: {date} of type: {type(date)}"

def to_unix(date):
    """
    Returns the date in UTC timezone and converts it to a unix timestamp.
    """
    if date.tzinfo is None:
        date = date.replace(tzinfo=timezone.utc)
    else: 
        assert date.tzinfo == timezone.utc, "to_unix was never intended to be used with non-utc dates"
    return int((date - datetime(1970,1,1,tzinfo=timezone.utc)).total_seconds())
        
def interp_levels(x, mesh, levels_in, levels_out):
    """
    Interpolates the input tensor x from levels_in to levels_out.
    
    Args:
        x (torch.Tensor): Input tensor of shape (B, Nlat, Nlon, D) or (Nlat, Nlon, D) where B is the batch size and always 1 and D represents levels
        mesh (LatLonGrid): Mesh object containing the levels information
        levels_in (list): List of input pressure levels
        levels_out (list): List of output pressure levels
        
    Returns:
        torch.Tensor: Output tensor of shape (B, Nlat, Nlon, D) or (Nlat, Nlon, D) with the levels interpolated (dimension D)
    """
    
    # Prepare the input tensor x
    n_dim_x = len(x.shape)
    if n_dim_x == 3: x = x.unsqueeze(0)
    B, Nlat, Nlon, D = x.shape
    
    # Gathers just the pressure variables from the input tensor
    # [B, Nlat, Nlon, D] -> [B, Nlat, Nlon, :n_pr_in] (where n_pr_in <= D) -> [B, Nlat, Nlon, n_pr_vars, len(levels_in)] since (n_pr_in = n_pr_vars * len(levels_in)
    n_pr_in = mesh.n_pr_vars * len(levels_in)
    xlevels = x[:,:,:,:n_pr_in]
    xlevels = xlevels.view(B, Nlat, Nlon, mesh.n_pr_vars, len(levels_in))
    outlevels = torch.zeros(B, Nlat, Nlon, mesh.n_pr_vars, len(levels_out), dtype=x.dtype, device=x.device)
    
    # Does the interpolating
    for index_levels_out, level in enumerate(levels_out):
        index_levels_in = np.searchsorted(levels_in, level)
        previous_level_index = max(index_levels_in - 1, 0)
        
        previous_level = levels_in[previous_level_index]
        next_level = levels_in[index_levels_in]
        interpolation_factor = 0 if index_levels_in == 0 else (level - previous_level) / (next_level - previous_level)
        
        if len(levels_in) > len(levels_out):
            assert interpolation_factor == 1.0 or (interpolation_factor == 0.0 and previous_level == next_level), f"When going to a fewer number of levels, you shouldn't actually be interpolating anything (len(levels_in) = {len(levels_in)} and len(levels_out) = {len(levels_out)}). Interpolation factor: {interpolation_factor}"
            outlevels[:,:,:,:,index_levels_out] = xlevels[:,:,:,:,index_levels_in]
        else:
            outlevels[:,:,:,:,index_levels_out] = xlevels[:,:,:,:,previous_level_index] * (1 - interpolation_factor) + xlevels[:,:,:,:,index_levels_in] * interpolation_factor
            
    # Reshapes the output tensor to match the original input tensor shape
    out = torch.cat((outlevels.flatten(start_dim=3), x[:,:,:,n_pr_in:]), dim=-1)
    if n_dim_x == 3: out = out.squeeze(0)
    return out

def load_normalization(mesh, with_means=False):
    """
    Loads normalization parameters from the mesh (at the levels specified by the mesh)
    
    Args:
        mesh (LatLonGrid): Mesh object containing the levels information
        with_means (bool): Whether to return the means along with the standard deviations
        
    Returns:
        tuple: A tuple containing the normalization parameters (norms, normalization_matrix_std, normalization_matrix_mean (if with_means is True))
    """
    with open('constants/normalization.json', 'r') as f:
        norms = json.load(f)
        
    normalization_matrix_mean = []
    normalization_matrix_std = []
    
    # Load pressure variable norms (only for the specified level)
    for pressure_var in mesh.pressure_vars:
        normalization_matrix_mean.append(np.array(norms[pressure_var]['mean'])[mesh.which_level])
        normalization_matrix_std.append(np.array(norms[pressure_var]['std'])[mesh.which_level])
        
    # Load surface variable norms 
    for surface_var in mesh.sfc_vars:
        if surface_var == 'zeropad':
            normalization_matrix_mean.append(np.array([0.]))
            normalization_matrix_std.append(np.array([1.]))
            continue
        normalization_matrix_mean.append(np.array([norms[surface_var]['mean']]))
        normalization_matrix_std.append(np.array([norms[surface_var]['std']]))

    normalization_matrix_mean = np.concatenate(normalization_matrix_mean).astype(np.float32)
    normalization_matrix_std = np.concatenate(normalization_matrix_std).astype(np.float32)
    
    if with_means:
        return norms, normalization_matrix_std, normalization_matrix_mean
    return norms, normalization_matrix_std
