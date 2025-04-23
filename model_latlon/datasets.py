import os
import numpy as np
from utils import to_unix, get_date, RED, interp_levels, core_sfc_vars
import torch
from itertools import product
from meshes import LatLonGrid
from datetime import timedelta
from functools import reduce

def get_proc_path_base(mesh, load_location=None, extra='base'):
    mesh_source_path = mesh.source.split("-")[0]
    loadloc = mesh.load_locations[0] if load_location is None else load_location
    base = os.path.join(loadloc, mesh_source_path)
    
    if extra != 'base':
        base = os.path.join(base,f'extra/{extra}')
        
    if not os.path.exists(base):
        raise FileNotFoundError(f"Path {base} does not exist. Please check the load_location or extra parameter.")
    
    if 'f000' in os.listdir(base): 
        base = os.path.join(base,'f000')
        
    return base

# Finds the path for the data type at a given date
def get_proc_path_time(date, mesh, load_location=None, extra='base'):
    base = get_proc_path_base(mesh, load_location=load_location, extra=extra)

    paths = [f'{base}/',f'{base}/{date.strftime("%Y%m")}/', f'{base}/{date.strftime("%Y")}/']
    
    fnames = [f'{to_unix(date)}.npz',f'{date.strftime("%Y%m%d%H")}.npz', f'{date.strftime("%Y%m%d")}.npz']
    
    for path in [s+e for s,e in product(paths,fnames)]:
        if os.path.exists(path):
            return path
    assert False, f'Could not find {date} with base {base} and extra {extra} at load location {load_location}'

class AnalysisDataset():
    def __init__(self, mesh, is_required=True):
        self.mesh = mesh
        self.is_required = is_required
        
        # From mesh
        self.source = mesh.source
        self.load_locations = mesh.load_locations
        
        # JACK_CHANGE_LATER
        self.all_var_files = mesh.all_var_files
        self.clamp_input = 13 
        self.clamp_output = 13
        
    def summary_str(self):
        return self.mesh.summary_str()

    def get_file_times(base, datemin, datemax):
        if 'f000' in base:
            ext = '.npz'
            filenames = np.array([f.name[:-len(ext)] for f in os.scandir(base) if f.name[:-len(ext)].isdigit() and f.name.endswith(ext)], dtype=np.int64)
            tots = np.sort(filenames)
            tots = tots[np.where(np.logical_and(to_unix(datemin) <= tots, tots <= to_unix(datemax)))]
        else:
            assert False, "This is not yet open source"
        assert len(tots) > 0, f"No dates found for {base} in {datemin} to {datemax}"
        return tots
        
    def get_loadable_times(self, datemin, datemax):
        alldates = []
        for var in self.all_var_files: 
            varx = var if var != 'zeropad' else '45_tcc'
            base = get_proc_path_base(self.mesh, load_location=self.load_locations[0], extra=varx)
            assert base is not None, f"No base found for {varx} in {datemin} to {datemax}"
            dates = self.get_file_times(base, datemin - timedelta(hours=24), datemax + timedelta(hours=24))
            assert len(dates) > 0, f"No dates found for {base} in {datemin} to {datemax}"
            alldates.append(dates)
            
        dates = reduce(np.intersect1d,alldates)
        dates = dates[np.where(np.logical_and(to_unix(datemin) <= dates, dates <= to_unix(datemax)))] 
        assert len(dates) > 0, f"No dates found for {self.mesh.string_id} in {datemin} to {datemax}"
        return dates

    def neo_get_latlon_input(self, date, is_output):
        mesh = self.mesh
        load_loc = self.load_locations[0]
        assert isinstance(mesh, LatLonGrid)
        
        path = get_proc_path_time(date, mesh, load_location=load_loc)
        
        if not os.path.exists(path):
            print(RED(f"Didn't find data for path: {path}"))
            return False, False
        
        data = np.load(path)
        
        if mesh.levels == []:
            pr, sfc = None, data["sfc"]
        elif data["pr"].shape[0] == 721:
            pr, sfc = data["pr"][:720], data["sfc"][:720]
        elif data["pr"].shape[0] == 1801:
            pr, sfc = data["pr"][:1800], data["sfc"][:1800]
        else:
            pr, sfc = data["pr"], data["sfc"]
            
        assert not np.isnan(pr).any(), "NaNs in pressure for " + str(date.strftime("%Y-%m-%d"))
        assert mesh.extra_pressure_vars == [], "This is not yet supported"
        
        if len(mesh.extra_sfc_vars) > 0 or mesh.extra_sfc_pad > 0:
            sfc_total = np.zeros((sfc.shape[0],sfc.shape[1],len(mesh.sfc_vars)), dtype=np.float16)
            sfc_total[:,:,:len(core_sfc_vars)] = sfc
            
            for i,extra in enumerate(mesh.extra_sfc_vars):
                if extra == "zeropad":
                    sfc_total[:,:,len(core_sfc_vars)+i] = 0
                    continue
                
                extra_path = get_proc_path_time(date, mesh, load_location=load_loc, extra=extra)
                assert os.path.exists(extra_path), f"extra_path {extra_path} does not exist"
                extra_data = np.load(extra_path)['x']
                if extra_data.shape == (721,1440): 
                    extra_data = extra_data[:720,:]
                if extra_data.shape[0] == 721:
                    extra_data = extra_data[:720]
                assert (extra_data.shape == sfc.shape[:-1]) or (extra_data.shape[0] == 1 and extra_data.shape[1:] == sfc.shape[:-1]), f"{extra_path} is shape {extra_data.shape}, expected {sfc.shape[:-1]}"
                
                # Convert nans to 0
                if not is_output: extra_data = np.nan_to_num(extra_data, nan=0.0)
                sfc_total[:,:,len(core_sfc_vars)+i] = extra_data 
            
            sfc = sfc_total
        assert sfc.shape[-1] == len(mesh.sfc_vars), f"sfc shape {sfc.shape} doesn't match sfc_vars {mesh.sfc_vars}"
        return pr,sfc

    def load_data(self, nix, *args, **kwargs):
        mesh = self.mesh
        F = torch.HalfTensor
        is_output = kwargs['is_output']
        
        pr, sfc = self.neo_get_latlon_input(get_date(nix), is_output)
        xsfc = F(sfc)
        assert pr is not None or mesh.levels == [], f"Missing pr data for {mesh.source} at {nix}"
        
        if pr is not None:
            xpr = F(pr)
            x_pr_flat = torch.flatten(xpr, start_dim=-2)
            x = torch.cat((x_pr_flat, xsfc), -1)
        else:
            x = xsfc
            
        if len(mesh.input_levels) != len(mesh.levels) and x.shape[-1] != mesh.n_vars:
            last_lev = mesh.input_levels
            for inter_lev in mesh.intermediate_levels + [mesh.levels]:
                x = interp_levels(x,mesh,last_lev,inter_lev)
                last_lev = inter_lev
        
        clamp = self.clamp_output if is_output else self.clamp_input
        x = torch.clamp(x, -clamp, clamp)
        assert x.shape[-1] == mesh.n_vars, f"{x.shape} vs {mesh.n_vars}, {mesh.source}"
        
        return x 
