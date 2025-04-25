import torch
import numpy as np
import json
import hashlib
import base64
from utils import load_normalization

from utils import levels_full, core_pressure_vars, core_sfc_vars, num2levels, ORANGE

class Mesh():
    """
    Base Mesh class for all meshes. 
    
    This abstract class guarantees all WeatherMesh meshes have a set_string_id() function. 
    The shape() function is optional, though encouraged. 
    This class also declares helpful defaults (such as is_required).
    When building a new Mesh, you should inherit from this class and implement the set_string_id (not-optional) and shape() (optional) functions.
    To ensure that the set_string_id function is called, you should call super().__post__init__() in your Mesh's __init__ function as the final line.
    """
    def __init__(self):
        self.is_required = True
        self.resolution = 0.25
        self.all_var_files = ['base']
        
    def __post__init__(self):
        """
        Handles all post initialization for the Mesh class. 
        
        Currently only sets the string id for the mesh.
        """
        self.set_string_id() 

    def shape(self):
        """
        Returns the shape of the mesh. This function is optional, but encouraged.

        Returns:
            (int): Shape of the mesh (-1 if you want no shape to be defined)
        """
        return -1 
    
    def set_string_id(self):
        raise NotImplementedError("set_string_id not implemented for this Mesh. You are required to define a unique string id for your mesh")

class LatLonGrid(Mesh):
    def __init__(self, **kwargs):
        super().__init__()
        # Settings
        self.source = 'era5-28'
        self.load_locations = ['data/']
        self.input_levels = None
        self.levels = None
        self.extra_sfc_vars = []
        self.extra_sfc_pad = 0
        
        # Defaults
        self.lats, self.lons = self.get_latlons(self.resolution)
        self.pressure_vars = core_pressure_vars
        self.core_sfc_vars = core_sfc_vars
        
        for k,v in kwargs.items():
            assert hasattr(self,k), f"{k} is not a LatLonGrid attribute"
            setattr(self,k,v)
        
        self.update()
        self.update_mesh()
        super().__post__init__()

    def get_latlons(self, resolution):
        """
        Gets the latitudes and longitudes for the mesh at a given degree resolution
        
        Args:
            resolution (float): Resolution in degrees for the latitudes and longitudes
            
        Returns:
            lats (np.ndarray): Array of latitudes
            lons (np.ndarray): Array of longitudes
        """
        lats = np.arange(90, -90.01, -resolution)[:-1]
        lons = np.arange(0, 359.99, resolution)
        lons[lons >= 180] -= 360
        return lats, lons

    def update(self):
        assert '-' in self.source, f'Source must be of the form "era5-28" or "hres-13", not {self.source}'
        num_levels = int(self.source.split('-')[1])
        
        # Ease of life things
        if self.levels is None:
            self.levels = num2levels[num_levels]
        if self.input_levels is None:
            self.input_levels = self.levels
        assert len(self.input_levels) == num_levels, f'Levels must be {num_levels} long for {self.source}, not {len(self.input_levels)}'
        
        # Define varlists for the mesh
        self.sfc_vars = self.core_sfc_vars + self.extra_sfc_vars + ['zeropad'] * self.extra_sfc_pad
        self.varlist = self.pressure_vars + self.sfc_vars
        self.all_var_files = ['base'] + self.extra_sfc_vars
        
        # Prepares array of full variable list (pressure variables at front then surface variables at end)
        self.full_varlist = []
        for v in self.pressure_vars:
            self.full_varlist += [v + "_" + str(l) for l in self.levels]
        self.full_varlist += self.sfc_vars
        
        # Helpful variables to have in the mesh class
        self.n_levels = len(self.levels)
        self.n_pr_vars = len(self.pressure_vars)
        self.which_level = [levels_full.index(x) for x in self.levels] 
        self.n_pr_vars = len(self.pressure_vars)
        self.n_pr = self.n_levels * self.n_pr_vars
        self.n_sfc = len(self.sfc_vars)
        self.n_sfc_vars = self.n_sfc
        self.n_vars = self.n_pr + self.n_sfc

    def update_mesh(self):
        self.Lons, self.Lats = np.meshgrid(self.lons, self.lats)
        self.Lons /= 180
        self.Lats /= 90
        self.xpos = np.stack((self.Lats, self.Lons), axis=2)
        self.norm, self.normalization_matrix_std, self.normalization_matrix_mean = load_normalization(self, with_means=True)

    def shape(self):
        return torch.Size([len(self.lats), len(self.lons), self.n_vars])

    def set_string_id(self):
        _, hash = self.to_json()
        s = f"{self.source}>{self.n_levels}p{self.n_pr_vars}s{self.n_sfc_vars-self.extra_sfc_pad}z{self.extra_sfc_pad}r{self.resolution}-{hash}"
        print(ORANGE(f"Set string_id for LatLonGrid to {s}"))
        self.string_id = s
        
    def summary_str(self):
        base_pr = f"0:{self.n_pr} base pr"; cur = self.n_pr
        base_sfc = f"{cur}:{cur + len(core_sfc_vars)} base sfc"; cur += len(core_sfc_vars)
        extra_sfc = f"{cur}:{cur + len(self.extra_sfc_vars)} extra sfc"; cur += len(self.extra_sfc_vars)
        zero_pad = f"{cur}:{cur + self.extra_sfc_pad} zero pad"
        if self.extra_sfc_pad == 0:
            zero_pad = "no zero pad"

        return f"source: {self.source}, {self.n_levels} lev [ {base_pr} | {base_sfc} | {extra_sfc} | {zero_pad} ]"

    def to_json(self):
        out = {}
        out['mesh_type'] = self.__class__.__name__
        out['pressure_vars'] = self.pressure_vars
        out['sfc_vars'] = self.sfc_vars
        out['full_varlist'] = self.full_varlist
        out['levels'] = self.levels
        out['lats'] = self.lats.tolist()
        out['lons'] = self.lons.tolist()
        out['res'] = self.resolution
        
        output_json = json.dumps(out, indent=2)
        hash_24bit = hashlib.sha256(output_json.encode()).digest()[:3]
        hash = base64.b64encode(hash_24bit).decode()
        hash = hash.replace('+', 'a').replace('/', 'b')
        return output_json, hash