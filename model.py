from meshes import LatLonGrid
from utils import levels_gfs, levels_hres, levels_medium

def get_wm3():
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    imesh1 = LatLonGrid(source='neogfs-25',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_gfs, levels=levels_medium)
    imesh2 = LatLonGrid(source='neohres-20',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_hres, levels=levels_medium)
    omesh = LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)

    from model_latlon.config import ForecastModelConfig
    from model_latlon.encoder import ResConvEncoder
    from model_latlon.decoder import ResConvDecoder
    from model_latlon.top import ForecastModel

    conf = ForecastModelConfig(inputs=[imesh1, imesh2], outputs=[omesh])
    conf.latent_size = 1024
    conf.pr_dims = [48, 192, 512]
    conf.parallel_encoders = True
    conf.encoder_weights = [0.1, 0.9]
    conf.affine = True
    conf.encdec_tr_depth = 2
    conf.checkpoint_type = "none"
    conf.oldpr = True
    conf.use_pole_convs = False
    conf.update()
    encoder1 = ResConvEncoder(imesh1,conf)
    encoder2 = ResConvEncoder(imesh2,conf)
    decoder = ResConvDecoder(omesh,conf)
    model = ForecastModel(conf, encoders=[encoder1, encoder2], decoders=[decoder])
    return model

if __name__ == "__main__":
    model = get_wm3()
    print(model)