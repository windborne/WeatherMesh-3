import torch
import numpy as np
from utils import get_date

N_ADDL_VARS = 4
G_RAD_CACHE = {}

def interpget(source, time_of_year, hr, a=300, b=400):
    global G_RAD_CACHE
    
    if source not in G_RAD_CACHE: 
        G_RAD_CACHE[source] = {}
        
    def load(x, hr):
        if (x, hr) in G_RAD_CACHE[source]:
            return G_RAD_CACHE[source][(x, hr)]
        
        F = torch.HalfTensor
        data = F(((np.load('constants/additional_vars/%s/%d_%d.npy' % (source, x, hr)) - a) / b))
        G_RAD_CACHE[source][(x, hr)] = data
        return data
    
    if time_of_year % 2 == 1:
        avg = load((time_of_year-1)%366, hr) * 0.5 + load((time_of_year+1)%366, hr) * 0.5
    else:
        avg = load(time_of_year, hr)
    
    return avg

def get_additional_vars(t0s):
    """
    Adds additional variables to the input tensor which are not dependent on the input data but do change vs time.
    
    Args:
        t0s (torch.Tensor): A tensor of shape (B,) containing the time of the input data
    """
    device = t0s.device
    dates = [get_date(t0.item()) for t0 in t0s]
    start_of_years = [date.replace(month=1, day=1) for date in dates]
    time_of_years = [int((date - soy).total_seconds()/86400) for date, soy in zip(dates, start_of_years)]
    
    radiations = [interpget("neoradiation_1", time_of_year, date.hour)[:720,:,np.newaxis].unsqueeze(0).to(device) for time_of_year, date in zip(time_of_years, dates)]
    radiations = torch.cat(radiations, dim=0)
    hours = torch.tensor([date.hour/24 for date in dates]).to(device)
    time_of_day = (torch.zeros_like(radiations, device=hours.device) + hours[:,None, None, None])
    
    solar_angles = [interpget("solarangle_1", time_of_year, date.hour, a=0, b=180/np.pi) for time_of_year, date in zip(time_of_years, dates)]
    solar_angles = [angle[:720, :, np.newaxis].unsqueeze(0).to(device) for angle in solar_angles]
    solar_angles = torch.cat(solar_angles, dim=0)
    
    sin_angles = torch.sin(solar_angles)
    cos_angles = torch.cos(solar_angles)
    
    out = torch.cat((radiations, time_of_day, sin_angles, cos_angles), axis=-1)
    assert out.shape[3] == N_ADDL_VARS, f"Expected {N_ADDL_VARS} additional variables, but got {out.shape[3]}"
    return out

def get_constant_vars(mesh):
    const_vars = []
    to_cat = []
        
    latlon = torch.FloatTensor(mesh.xpos)
    slatlon = torch.sin((latlon*torch.Tensor([np.pi/2,np.pi])))
    clatlon = torch.cos((latlon*torch.Tensor([np.pi/2,np.pi])))
    const_vars += ['sinlat','sinlon','coslat','coslon'] 
    to_cat += [slatlon,clatlon]

    land_mask_np = np.load('constants/additional_variables/land_mask.npy')
    land_mask = torch.BoolTensor(np.round(downsample(land_mask_np, mesh.xpos.shape)))
    const_vars += ['land_mask']
    to_cat += [land_mask.unsqueeze(-1)]

    soil_type_np = np.load('constants/additional_variables/soil_type.npy')
    soil_type_np = downsample(soil_type_np, mesh.xpos.shape, reduce=np.min)
    soil_type = torch.BoolTensor(to_onehot(soil_type_np))
    const_vars += [f'soil_type{i}' for i in range(soil_type.shape[-1])]
    to_cat += [soil_type]

    elevation_np = np.load('constants/additional_variables/topography.npy')
    elevation_np = downsample(elevation_np, mesh.xpos.shape, reduce=np.mean)
    elevation_np = elevation_np / np.max(elevation_np)
    elevation = torch.FloatTensor(elevation_np)
    const_vars += ['elevation']
    to_cat += [elevation.unsqueeze(-1)]

    const_data = torch.cat(to_cat, axis=-1)
    
    assert const_data.shape[-1] == len(const_vars), f"{const_data.shape[-1]} vs {len(const_vars)}"
    return const_data, const_vars

def downsample(mask,shape,reduce=np.mean):
    dlat = (mask.shape[0]-1) // shape[0]
    dlon = mask.shape[1] // shape[1]
    assert dlon == dlat
    d = dlat
    toshape = (shape[0], d, shape[1], d)
    ret = reduce(mask[:-1,:].reshape(toshape),axis=(1,3)) # Remove the south pole
    assert ret.shape == shape[:2], (ret.shape, shape[:2])
    return ret

def to_onehot(x):
    x = x.astype(int)
    D = np.max(x)+1
    return np.eye(D)[x]