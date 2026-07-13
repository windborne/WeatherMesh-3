"""Load WeatherMesh-3 and run a forecast; de-normalize outputs to physical units."""
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.chdir(REPO)

from . import outputs  # noqa: E402  (shared precip de-transform)

FIELDS = ["167_2t", "151_msl", "165_10u", "166_10v", "129_z_500", "130_t_850",
          "131_u_250", "132_v_250", "133_q_850"]


def load_model(weights="model/WeatherMesh3.pt", device="cuda"):
    from model import get_WeatherMesh3
    return get_WeatherMesh3(weights).to(device).eval()


def run(model, gx, hx, t0_unix, era_mesh, lead_hours=6, device="cuda"):
    x = [gx[None].to(device), hx[None].to(device), torch.tensor([t0_unix], device=device)]
    with torch.no_grad():
        out = model(x, [lead_hours])
    pred = out[lead_hours][0]
    if pred.ndim == 4:
        pred = pred[0]
    real = pred.float().cpu().numpy() * era_mesh.normalization_matrix_std + era_mesh.normalization_matrix_mean

    fields = {name: real[..., era_mesh.full_varlist.index(name)] for name in FIELDS}
    fields["precip6h_mm"] = outputs.total_precip_6h_mm(real, era_mesh, clamp=True)
    return real, fields, float(out["latent_l2"])
