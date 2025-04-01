import torch
import numpy as np
import json
import hashlib
import base64

from utils import SourceCodeLogger, levels_full, levels_tiny, core_pressure_vars, core_sfc_vars, num2levels, ORANGE, CONSTS_PATH, PROC_PATH, RUNS_PATH

class Mesh():
    def __init__(self):
        global CONSTS_PATH,PROC_PATH,RUNS_PATH
        self.PROC_PATH = PROC_PATH ; self.CONSTS_PATH = CONSTS_PATH ; self.RUNS_PATH = RUNS_PATH
        
        self.is_required = True
        self.all_var_files = ['base']
        self.ens_num = None
        self.subsamp = 1
        self.resolution = 0.25
        self.hourly = False
        
    def __post__init__(self):
        # Set the string id for the mesh
        # Calls the child's override first unless it is not specified
        self.set_string_id() 

    # The shape of the expected tensor in torch.Size([...]) format
    def shape(self):
        return -1 # Case where there is no shape 
    
    # Defines a unique string id to identify the mesh in the dataloader
    def set_string_id(self):
        raise NotImplementedError("set_string_id not implemented for this Mesh. You are required to define a unique string id for your mesh")

def get_grid_latlons(res):
    lats = np.arange(90, -90.01, -res)[:-1]
    
    lons = np.arange(0, 359.99, res)
    lons[lons >= 180] -= 360

    return lats, lons

class LatLonGrid(Mesh, SourceCodeLogger):
    def __init__(self, **kwargs):
        super().__init__()
        self.source = 'era5-28'
        self.load_locations = ['/fast/proc/']
        self.hour_offset = 0
        
        self.input_levels = None
        self.levels = None
        self.extra_sfc_vars = []
        self.extra_sfc_pad = 0
        self.extra_pressure_vars = []
        self.intermediate_levels = []
        
        lats = np.arange(90, -90.01, -self.resolution)[:-1]
        lats.shape = (lats.shape[0]//self.subsamp, self.subsamp)
        self.lats = np.mean(lats, axis=1)

        lons = np.arange(0, 359.99, self.resolution)
        lons.shape = (lons.shape[0]//self.subsamp, self.subsamp)
        lons[lons >= 180] -= 360
        self.lons = np.mean(lons, axis=1)
        self.parent = None
        self.bbox = None
        
        for k,v in kwargs.items():
            assert hasattr(self,k), f"{k} is not a NeoDatasetConfig attribute"
            setattr(self,k,v)
        
        self.update()
        self.update_mesh()
        if not hasattr(self, 'string_id'): super().__post__init__()

    def shape(self):
        return torch.Size([len(self.lats), len(self.lons), self.n_vars])

    def set_string_id(self):
        j, hash = self.to_json()
        dtstr = '' if self.hour_offset == 0 else f"(-{self.hour_offset}h)"
        s = f"{dtstr}{self.source}>{self.n_levels}p{self.n_pr_vars}s{self.n_sfc_vars-self.extra_sfc_pad}z{self.extra_sfc_pad}r{self.res}-{hash}"
        print(ORANGE(f"Set string_id for LatLonGrid to {s}"))
        self.string_id = s
        
    def update(self):
        global levels_full
        assert '-' in self.source, f'source must be of the form "era5-28" or "hres-13", not {self.source}'
        #self.pressure_vars = ["129_z", "130_t", "131_u", "132_v", "135_w", "133_q", "075_crwc", "076_cswc", "248_cc", "246_clwc", "247_ciwc"]
        numlev = int(self.source.split('-')[1])
        if self.levels is None:
            self.levels = num2levels[numlev]
        if self.input_levels is None:
            self.input_levels = self.levels
        assert len(self.input_levels) == numlev, f'levels must be {numlev} long for {self.source}, not {len(self.input_levels)}'
        if self.source == 'hres-13':
            assert len(self.input_levels) == len(levels_tiny), f'levels must be {len(levels_tiny)} long for hres'
        self.pressure_vars = core_pressure_vars + self.extra_pressure_vars
        self.core_sfc_vars = core_sfc_vars
        self.sfc_vars = core_sfc_vars + self.extra_sfc_vars + ['zeropad']*self.extra_sfc_pad
        self.varlist = self.pressure_vars + self.sfc_vars
        self.full_varlist = []
        for v in self.pressure_vars:
            self.full_varlist = self.full_varlist + [v+"_"+str(l) for l in self.levels]

        self.all_var_files = ['base'] + self.extra_pressure_vars + self.extra_sfc_vars
        self.full_varlist = self.full_varlist + self.sfc_vars

        self.n_levels = len(self.levels)
        self.n_pr_vars = len(self.pressure_vars)
        self.wh_lev = [levels_full.index(x) for x in self.levels] #which_levels
        self.n_pr_vars = len(self.pressure_vars)
        self.n_pr = self.n_levels * self.n_pr_vars
        self.n_sfc = len(self.sfc_vars)
        self.n_sfc_vars = self.n_sfc
        self.n_vars = self.n_pr + self.n_sfc

    def get_zeros(self):
        return np.zeros((1, len(self.lats), len(self.lons), self.n_vars), dtype=np.float32)
    
    def summary_str(self):
        base_pr = f"0:{self.n_pr} base pr"; cur = self.n_pr
        base_sfc = f"{cur}:{cur + len(core_sfc_vars)} base sfc"; cur += len(core_sfc_vars)
        extra_sfc = f"{cur}:{cur + len(self.extra_sfc_vars)} extra sfc"; cur += len(self.extra_sfc_vars)
        zero_pad = f"{cur}:{cur + self.extra_sfc_pad} zero pad"
        if self.extra_sfc_pad == 0:
            zero_pad = "no zero pad"

        return f"source: {self.source}, {self.n_levels} lev [ {base_pr} | {base_sfc} | {extra_sfc} | {zero_pad} ]"

    def update_mesh(self):
        self.Lons, self.Lats = np.meshgrid(self.lons, self.lats)
        self.res = self.resolution * self.subsamp
        self.Lons /= 180
        self.Lats /= 90
        self.xpos = np.stack((self.Lats, self.Lons), axis=2)

        import utils
        self.state_norm,self.state_norm_stds,self.state_norm_means = utils.load_state_norm(self.wh_lev,self,with_means=True)

    def lon2i(self, lons):
        return np.argmin(np.abs(self.lons[:,np.newaxis] - lons),axis=0)
    
    def lat2i(self, lats):
        return np.argmin(np.abs(self.lats[:,np.newaxis] - lats),axis=0)
    
    def to_json(self, model_name=None):
        out = {}
        out['mesh_type'] = self.__class__.__name__
        out['pressure_vars'] = self.pressure_vars
        out['sfc_vars'] = self.sfc_vars
        out['full_varlist'] = self.full_varlist
        out['levels'] = self.levels
        out['lats'] = self.lats.tolist()
        out['lons'] = self.lons.tolist()
        out['res'] = self.res
        if model_name is not None: out['model_name'] = model_name
        st = json.dumps(out, indent=2)
        hash_24bit = hashlib.sha256(st.encode()).digest()[:3]
        base64_encoded = base64.b64encode(hash_24bit).decode()
        # replace + and / with a and b to because / in filenames are RIP. we could use urlsafe_b64encode but it would be incompatible with the old hashes
        base64_encoded = base64_encoded.replace('+', 'a').replace('/', 'b')
        return st, base64_encoded

    @staticmethod
    def from_json(js):
        out = LatLonGrid()
        out.__dict__ = js
        out.source = "unknown"
        out.lats = np.array(out.lats)
        out.lons = np.array(out.lons)
        out.wh_lev = [levels_full.index(x) for x in out.levels] #which_levels
        out.subsamp = 1; assert np.diff(out.lats)[1] == -0.25, f'lat diff is {np.diff(out.lats)}'
        out.update_mesh()
        return out