import torch
import sys
import types
from meshes import LatLonGrid
from model_latlon.config import ForecastModelConfig
from model_latlon.encoder import ResConvEncoder
from model_latlon.decoder import ResConvDecoder
from model_latlon.top import ForecastModel
from utils import levels_gfs, levels_hres, levels_medium
from utils import MAGENTA

def load_weights(model, weights_path):
    """Load weights into model, handling checkpoint format and missing modules"""
    
    # Create temporary utils_lite module for checkpoint compatibility
    import utils
    utils_lite = types.ModuleType('utils_lite')
    for attr in dir(utils):
        if not attr.startswith('_'):
            setattr(utils_lite, attr, getattr(utils, attr))
    for class_name in ['WeatherTrainerConfig', 'LRScheduleConfig', 'DataConfig', 'OptimConfig']:
        setattr(utils_lite, class_name, type(class_name, (), {'__init__': lambda self, *a, **k: None}))
    sys.modules['utils_lite'] = utils_lite
    
    try:
        checkpoint = torch.load(weights_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
    finally:
        del sys.modules['utils_lite']

def get_WeatherMesh3(weights_path="WeatherMesh3.pt"):
    """
    Gets the WeatherMesh3 model with weights loaded.
    
    Args:
        weights_path (str, optional): Weights path for a WeatherMesh model. Defaults to "WeatherMesh3.pt".
        
    Returns:
        model (ForecastModel): A WeatherMesh model with loaded weights.
    """
    # Declare the extra variables we want to use
    extra_sfc_variables_input = [
        '45_tcc', 
        '168_2d', 
        '246_100u', 
        '247_100v'
    ]
    extra_sfc_variables_output = [
        '15_msnswrf', 
        '45_tcc', 
        '168_2d', 
        '246_100u', 
        '247_100v', 
        '142_lsp', 
        '143_cp', 
        '201_mx2t', 
        '202_mn2t', 
        '142_lsp-6h', 
        '143_cp-6h', 
        '201_mx2t-6h', 
        '202_mn2t-6h'
    ]
    
    # Declare all sources of data
    gfs_input_mesh = LatLonGrid(
        source='neogfs-25',
        extra_sfc_vars=extra_sfc_variables_input, 
        extra_sfc_pad=len(extra_sfc_variables_output) - len(extra_sfc_variables_input),
        input_levels=levels_gfs,
        levels=levels_medium,
    )
    hres_input_mesh = LatLonGrid(
        source='neohres-20', 
        extra_sfc_vars=extra_sfc_variables_input, 
        extra_sfc_pad=len(extra_sfc_variables_output) - len(extra_sfc_variables_input),
        input_levels=levels_hres,
        levels=levels_medium,
    )
    era_output_mesh = LatLonGrid(
        source='era5-28',
        extra_sfc_vars=extra_sfc_variables_output,
        levels=levels_medium,
    )
    
    config = ForecastModelConfig(
        inputs=[gfs_input_mesh, hres_input_mesh], 
        outputs=[era_output_mesh], 
        latent_size=1024,
        pr_dims=[48, 192, 512],
        parallel_encoders=True,
        encoder_weights=[0.1, 0.9],
        affine=True,
        encdec_tr_depth=2,
        checkpoint_type="none",
        use_pole_convs=False,
    )
    
    gfs_encoder = ResConvEncoder(gfs_input_mesh, config)
    hres_encoder = ResConvEncoder(hres_input_mesh, config)
    era_decoder = ResConvDecoder(era_output_mesh, config)
    
    model = ForecastModel(config, 
                          encoders=[gfs_encoder, hres_encoder],
                          decoders=[era_decoder])
    
    # Load weights
    print(MAGENTA(f"Loading weights from {weights_path}..."))
    load_weights(model, weights_path)
    print(MAGENTA("Weights loaded successfully!"))
    
    return model

if __name__ == "__main__":
    print(MAGENTA("Loading the WeatherMesh3 architecture..."))
    model = get_WeatherMesh3()
    print(model)
    print(MAGENTA("🎉🎉🎉 The WeatherMesh3 architecture has been loaded successfully! 🎉🎉🎉"))