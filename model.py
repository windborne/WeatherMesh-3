import torch
import numpy as np
from datetime import datetime, timezone
from meshes import LatLonGrid
from model_latlon.config import ForecastModelConfig
from model_latlon.encoder import ResConvEncoder
from model_latlon.decoder import ResConvDecoder
from model_latlon.top import ForecastModel
from model_latlon.data import DataConfig, WeatherDataset
from model_latlon.datasets import AnalysisDataset
from utils import levels_gfs, levels_hres, levels_medium, collate_fn, unnorm, save_instance

def get_WeatherMesh3(data_path="data/", weights_path="WeatherMesh3.pt"):
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
        load_locations=[data_path],
    )
    hres_input_mesh = LatLonGrid(
        source='neohres-20', 
        extra_sfc_vars=extra_sfc_variables_input, 
        extra_sfc_pad=len(extra_sfc_variables_output) - len(extra_sfc_variables_input),
        input_levels=levels_hres,
        levels=levels_medium,
        load_locations=[data_path]
    )
    era_output_mesh = LatLonGrid(
        source='era5-28',
        extra_sfc_vars=extra_sfc_variables_output,
        levels=levels_medium,
        load_locations=[data_path],
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
        oldpr=True,
        use_pole_convs=False,
    )
    
    gfs_encoder = ResConvEncoder(gfs_input_mesh,config)
    hres_encoder = ResConvEncoder(hres_input_mesh,config)
    era_decoder = ResConvDecoder(era_output_mesh,config)
    
    model = ForecastModel(config, 
                          encoders=[gfs_encoder, hres_encoder],
                          decoders=[era_decoder])
    
    # Load the weights
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    old_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(old_state_dict, strict=True)
    return model


def get_data(model, date):
    """
    Gets input data for a WeatherMesh model.
    
    Args:
        model (ForecastModel): A WeatherMesh model for which you want data for
        date (str): The date for which you want data for. Assumes you already have the data downloaded. Assumes date is in the format '%Y-%m-%d %H:00:00' (note the resolution is in hours)
        
    Returns:
        sample (dict): A dictionary containing the input data for the model. See model_latlon/data.py for more details.
    """
    date = datetime.strptime(date, '%Y-%m-%d %H:00:00').replace(tzinfo=timezone.utc)
    
    date = datetime.timestamp(1741305600) #JACK_CHANGE_LATER
    
    input_meshes = model.config.inputs
    output_meshes = model.config.outputs
    input_datasets = [AnalysisDataset(mesh) for mesh in input_meshes]
    output_datasets = [AnalysisDataset(mesh) for mesh in output_meshes]
    
    dataset = WeatherDataset(
        DataConfig(
            inputs=input_datasets,
            outputs=output_datasets,
            timesteps=[0],
            only_at_z=[0],
            clamp_output=np.inf,
            requested_train_dates = [date],
            realtime=True, 
        )
    )
    
    dataset.check_for_dates() 
    sample = collate_fn(dataset[0])
    return sample.get_x_t0(model.encoders)
    
def save_hour(output, hour, model):
    """
    Saves the output of the model to a file.
    
    Args:
        output (torch.Tensor): The output of the model.
        hour (int): The hour for which the output is for.
    """
    output = output[0].to('cpu')
    output_mesh = model.config.outputs[0]
    output = unnorm(output, output_mesh)
    save_instance(output, OUTPUT_DIR, round(hour), output_mesh)
    
if __name__ == "__main__":
    WEIGHTS_PATH = 'ignored/WeatherMesh3.pt' #JACK_CHANGE_LATER
    DATA_PATH = 'data/' # Location where the input data is stored #JACK_CHANGE_LATER
    EVAL_DATE = "2025-03-14 15:00:00"
    OUTPUT_DIR = f'outputs/{EVAL_DATE}/'
    TODO = [0, 24, 48, 72, 96, 120, 144] # Hours to run the model for
    
    model = get_WeatherMesh3(data_path=DATA_PATH, weights_path=WEIGHTS_PATH)
    print(model) #JACK_CHANGE_LATER
    
    # If you want to use on a GPU, change this to the appropriate CUDA device
    CUDA_DEVICE = 'cpu' # "cuda:0"
    model = model.to(CUDA_DEVICE)
    
    model.eval()
    x = get_data(model, EVAL_DATE)
    
    with torch.no_grad():
        with torch.autocast(enabled=True, device_type=CUDA_DEVICE, dtype=torch.float16):
            x_on_device = [x_iter.to(CUDA_DEVICE) for x_iter in x]
            model.forward(x_on_device, TODO, send_to_cpu=False, callback=save_hour)
        output = model(x)