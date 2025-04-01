from meshes import LatLonGrid

class ForecastModelConfig():
    def __init__(self,inputs,**kwargs):
        self.inputs = inputs
        self.outputs = inputs
        self.processor_dts = [6]
        self.parallel_encoders = False
        self.encoder_weights = [1,1,1,1,1,1]
        self.window_size = (5,7,7)
        self.oldpr = False
        self.pr_dims = [32,64,256]
        self.encdec_tr_depth = 2
        self.latent_size = 1024
        self.dims_per_head = 32
        self.affine = True
        self.tr_embedding = 'sincos'
        self.pr_depth = [8]
        self.checkpoint_type = "matepoint" # or "torch" or "none"
        self.nsight = False # used for profiling
        self.patch_size = (4,8,8)
        self.simple_decoder_patch_size = None
        self.use_pole_convs = True

        for k,v in kwargs.items():
            assert hasattr(self,k), f"{k} is not a ForecastModelConfig attribute"
            setattr(self,k,v)
        self.update()

    def update(self): 
        self.num_heads = self.latent_size // self.dims_per_head

        omesh = self.outputs[0]
        assert isinstance(omesh, LatLonGrid)
        self.latent_lats = omesh.lats[::self.patch_size[1]]
        self.latent_lons = omesh.lons[::self.patch_size[2]]
        assert self.patch_size[1] == self.patch_size[2], "patch_size must be square"
        if self.simple_decoder_patch_size is None:
            self.simple_decoder_patch_size = self.patch_size
        self.latent_res = omesh.res * self.patch_size[1]
        self.latent_levels = omesh.levels[::-self.patch_size[0]][::-1]

        # legacy
        self.processor_dt = self.processor_dts

        assert self.tr_embedding == 'sincos'
        self.embedding_module = None