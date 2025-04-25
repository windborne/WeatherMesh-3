from meshes import LatLonGrid

class ForecastModelConfig():
    def __init__(self, inputs, **kwargs):
        self.inputs = inputs
        self.outputs = inputs
        self.checkpoint_type = "matepoint" # or "torch" or "none"
        
        # Model settings
        self.processor_dts = [6]
        self.parallel_encoders = False
        self.encoder_weights = [1,1,1,1,1,1]
        self.window_size = (5,7,7)
        self.pr_dims = [32,64,256]
        self.encdec_tr_depth = 2
        self.latent_size = 1024
        self.dims_per_head = 32
        self.affine = True
        self.tr_embedding = 'sincos'
        self.pr_depth = [8]
        self.patch_size = (4,8,8)
        self.use_pole_convs = True

        for k,v in kwargs.items():
            assert hasattr(self,k), f"{k} is not a ForecastModelConfig attribute"
            setattr(self,k,v)
            
        self.update()

    def update(self): 
        omesh = self.outputs[0]
        assert isinstance(omesh, LatLonGrid), "This is not yet open source"
        assert self.tr_embedding == 'sincos', "This is not yet open source"
        
        self.num_heads = self.latent_size // self.dims_per_head
        self.latent_lats = omesh.lats[::self.patch_size[1]]
        self.latent_lons = omesh.lons[::self.patch_size[2]]
        assert self.patch_size[1] == self.patch_size[2], f"patch_size must be square but got {self.patch_size}"
        
        self.latent_resolution = omesh.resolution * self.patch_size[1]
        self.latent_levels = omesh.levels[::-self.patch_size[0]][::-1]