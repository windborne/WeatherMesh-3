import torch
import numpy as np
from utils import get_date

# JACK_CHANGE_LATER
G_rad_cache = {}
def interpget(src, toy, hr, a=300, b=400):
    global G_rad_cache
    if src not in G_rad_cache:
        G_rad_cache[src] = {}

    def load(xx, hr):
        if (xx,hr) in G_rad_cache[src]:
            return G_rad_cache[src][(xx, hr)]

        F = torch.HalfTensor  
        ohp = F(((np.load(CONSTS_PATH+'/%s/%d_%d.npy' % (src, xx, hr)) - a) / b))
        G_rad_cache[src][(xx,hr)] = ohp
        return ohp
    
    if toy % 2 == 1:
        avg = load((toy-1)%366, hr) * 0.5 + load((toy+1)%366, hr) * 0.5
    else:
        avg = load(toy, hr)
    return avg

def get_radiation(self, toy, hr=0):
    return interpget("neoradiation_1", toy, hr)
# JACK_CHANGE_LATER

N_ADDL_VARS = 4
def get_additional_vars(t0s):
    # This function adds additional variables to the input tensor which are not dependent on the input data,
    # but do change vs time. This is different than the additional variables which are constant vs time, those are 
    # done in the EarchSpecificModel class.

    device = t0s.device
    dates = [get_date(t0.item()) for t0 in t0s]
    soys = [date.replace(month=1, day=1) for date in dates]
    toys = [int((date - soy).total_seconds()/86400) for date, soy in zip(dates, soys)]

    radiations = [get_radiation(toy, date.hour)[:720,:,np.newaxis].unsqueeze(0).to(device) for toy, date in zip(toys, dates)]
    radiations = torch.cat(radiations, dim=0)
    hours = torch.tensor([date.hour/24 for date in dates]).to(device)   
    timeofday = (torch.zeros_like(radiations, device=hours.device) + hours[:,None, None, None])

    angs = [interpget("solarangle_1", toy, date.hour, a=0, b=180/np.pi) for toy, date in zip(toys, dates)]
    angs = [ang[:720, :, np.newaxis].unsqueeze(0).to(device) for ang in angs]
    angs = torch.cat(angs, dim=0)

    sa = torch.sin(angs)
    ca = torch.cos(angs)

    out = torch.cat((radiations, timeofday, sa, ca), axis=-1)
    del angs
    del sa
    del ca
    del radiations
    del timeofday

    assert out.shape[3] == N_ADDL_VARS
    return out


def get_constant_vars(mesh,sincos_latlon=True):
    const_vars = []
    to_cat = []
    if sincos_latlon == True:
        latlon = torch.FloatTensor(mesh.xpos)
        slatlon = torch.sin((latlon*torch.Tensor([np.pi/2,np.pi])))
        clatlon = torch.cos((latlon*torch.Tensor([np.pi/2,np.pi])))
        const_vars += ['sinlat','sinlon','coslat','coslon']; to_cat += [slatlon,clatlon]
    else:
        latlon = torch.FloatTensor(mesh.xpos)
        const_vars += ['lat','lon']; to_cat += [latlon]

    land_mask_np = np.load('constants/additional_variables/land_mask.npy')
    land_mask = torch.BoolTensor(np.round(downsample(land_mask_np, mesh.xpos.shape)))
    const_vars += ['land_mask']; to_cat += [land_mask.unsqueeze(-1)]

    soil_type_np = np.load('constants/additional_variables/soil_type.npy')
    soil_type_np = downsample(soil_type_np, mesh.xpos.shape,reduce=np.min)
    soil_type = torch.BoolTensor(to_onehot(soil_type_np))
    const_vars += [f'soil_type{i}' for i in range(soil_type.shape[-1])]; to_cat += [soil_type]

    elevation_np = np.load('constants/additional_variables/topography.npy')
    elevation_np = downsample(elevation_np, mesh.xpos.shape,reduce=np.mean)
    elevation_np = elevation_np / np.max(elevation_np)
    elevation = torch.FloatTensor(elevation_np)
    const_vars += ['elevation']; to_cat += [elevation.unsqueeze(-1)]

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