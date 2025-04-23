import os
import torch
import pickle
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

# Converts a test date into a folder directory
def get_date_dir(date):
    return date.split('')[0].replace('-','')

# Returns the date in UTC timezone
def get_date(date):
    if type(date) == datetime:
        assert date.tzinfo is None or date.tzinfo == timezone.utc, f"get_date() was not designed to be used with non-UTC dates. Got date: {date} with {date.tzinfo}"
        return date.replace(tzinfo=timezone.utc) 
    elif np.issubdtype(type(date), np.number):
        nix = int(date)
        return datetime(1970,1,1,tzinfo=timezone.utc)+timedelta(seconds=nix)
    assert False, f"Unable to parse date: {date} of type: {type(date)}"

# Returns the date in UTC timezone and converts it to a unix timestamp
def to_unix(date):
    if date.tzinfo is None:
        date = date.replace(tzinfo=timezone.utc)
    else: 
        assert date.tzinfo == timezone.utc, "to_unix was never intended to be used with non-utc dates"
    return int((date - datetime(1970,1,1,tzinfo=timezone.utc)).total_seconds())

# JACK_CHANGE_LATER
def interp_levels(x,mesh,levels_in,levels_out):
    #assert len(levels_in) < len(levels_out)
    xdim = len(x.shape)
    if xdim == 3: x = x.unsqueeze(0)
    B,Nlat,Nlon,D = x.shape
    Nlev = len(levels_out)
    n_pr_in = mesh.n_pr_vars*len(levels_in)
    xlevels = x[:,:,:,:n_pr_in].view(B,Nlat,Nlon,mesh.n_pr_vars,len(levels_in))
    outlevels = torch.zeros(B,Nlat,Nlon,mesh.n_pr_vars,len(levels_out),dtype=x.dtype,device=x.device)
    for i,l in enumerate(levels_out):
        i2 = np.searchsorted(levels_in,l); i1 = max(i2-1,0)
        #if i == 10: i2 += 1 
        l1 = levels_in[i1]; l2 = levels_in[i2]
        th = 0 if i2 == 0 else (l-l1)/(l2-l1)
        if len(levels_in) > len(levels_out):
            #print("hey uh", i, l, l1, l2)
            assert th==1.0 or (th==0.0 and l1==l2), f'th: {th}. When going to a fewer number of levels, you shouldnt actually be interpolating stuff'
            outlevels[:,:,:,:,i] = xlevels[:,:,:,:,i2]
        else:
            outlevels[:,:,:,:,i] = xlevels[:,:,:,:,i1] * (1-th) + xlevels[:,:,:,:,i2] * th
        #print(i1,i2,l,l1,l2,th)
    #print(outlevels.shape)
    out = torch.cat((outlevels.flatten(start_dim=3),x[:,:,:,n_pr_in:]),dim=-1)
    if xdim == 3: out = out.squeeze(0)
    return out

# Loads normalization parameters from the mesh (at the levels specified by the mesh)
def load_normalization(mesh, with_means=False):
    with open('constants/normalization.pickle', 'rb') as f:
        norms = pickle.load(f)
        
    for key, value in norms.items():
        # We store mean and variance in the pickle file for all variables
        mean, variance = value 
        std = np.sqrt(variance)
        norms[key] = (mean, std)
    
    normalization_matrix_mean = []
    normalization_matrix_std = []
    
    # Load pressure variable norms (only for the specified level)
    for pressure_var in mesh.pressure_vars:
        normalization_matrix_mean.append(norms[pressure_var][0][mesh.which_level])
        normalization_matrix_std.append(norms[pressure_var][1][mesh.which_level])
        
    # Load surface variable norms 
    for surface_var in mesh.sfc_vars:
        if surface_var == 'zeropad':
            normalization_matrix_mean.append(np.array([0.]))
            normalization_matrix_std.append(np.array([1.]))
            continue
        normalization_matrix_mean.append(norms[surface_var][0])
        normalization_matrix_std.append(norms[surface_var][1])

    normalization_matrix_mean = np.concatenate(normalization_matrix_mean).astype(np.float32)
    normalization_matrix_std = np.array(normalization_matrix_std).astype(np.float32)
    
    if with_means:
        return norms, normalization_matrix_std, normalization_matrix_mean
    return norms, normalization_matrix_std

# Default collate function for the dataloader
def collate_fn(batch):
    assert len(batch) == 1, f"We assume batch size is 1, current batch size is: {len(batch)}"
    batch = batch[0]
    
    # Gather inputs
    for input_index, input_data in enumerate(batch.inputs):
        mesh_ids, tensors = input_data
        collated_tensors = [tensor.unsqueeze(0) for tensor in tensors]
        batch.inputs[input_index] = [mesh_ids, collated_tensors]
        
    # Gather outputs
    for output_index, output_data in enumerate(batch.outputs):
        mesh_ids, tensors = output_data
        collated_tensors = []
        for tensor in tensors:
            if type(tensor) == torch.Tensor:
                tensor = tensor.unsqueeze(0)
            collated_tensors.append(tensor)
        batch.outputs[output_index] = [mesh_ids, collated_tensors]
        
    return batch

# Unnormalizes the output tensor using the normalization parameters from the mesh
def unnorm(x, mesh):
    assert 'zeropad' not in mesh.full_varlist, "There shouldn't be zeropads in the output mesh."
    assert x.shape[-1] == mesh.normalization_matrix_std.shape[-1], f"x.shape[-1] {x.shape[-1]} does not match mesh.normalization_matrix_std.shape[-1] {mesh.normalization_matrix_std.shape[-1]}"
    
    means = torch.Tensor(mesh.normalization_matrix_mean).to(x.device)
    stds = torch.Tensor(mesh.normalization_matrix_std).to(x.device)
    x = (x[..., :len(mesh.full_varlist)] * stds) + means
    
    assert x.dtype == torch.float32, "Make sure output is in float32"
    return x

# Saves an instance of WeatherMesh output along with its metadata
def save_instance(x, save_dir, hour, mesh):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
        
    json, hash = mesh.to_json()
    os.makedirs(save_dir, exist_ok=True)
    save_metadata(json, hash, save_dir)
    
    assert x.shape[0] == 1, f"Batch size should be 1, got {x.shape[0]}"
    x = x[0]
    filepath = save_dir + f"WeatherMesh_at_{hour:02}z.{hash}.npy"
    np.save(filepath, x)

# Saves the metadata for the WeatherMesh output
def save_metadata(json, hash, save_dir):
    meta_path = os.path.join(save_dir, f'meta.{hash}.json')
    
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            json2 = f.read()
        assert json == json2, "metadata mismatch"
    
    with open(meta_path, 'w') as f:
        f.write(json)