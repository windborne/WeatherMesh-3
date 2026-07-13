"""Turn raw GFS fields into the two normalized encoder-input tensors WM-3 expects.

Per the assignment note, GFS feeds *both* encoders. The GFS encoder wants 25
levels and the HRES encoder wants 20; both level sets are sourced from GFS and
lifted to the model's 28 internal levels with the repo's own `interp_levels`.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from meshes import LatLonGrid                      # noqa: E402
from utils import (levels_gfs, levels_hres, levels_medium,   # noqa: E402
                   levels_full, interp_levels)

from . import config

NORM = json.load(open(REPO / "constants" / "normalization.json"))


EXTRA_SFC_OUTPUT = ["15_msnswrf", "45_tcc", "168_2d", "246_100u", "247_100v",
                    "142_lsp", "143_cp", "201_mx2t", "202_mn2t",
                    "142_lsp-6h", "143_cp-6h", "201_mx2t-6h", "202_mn2t-6h"]


def _mesh(source, input_levels):
    return LatLonGrid(source=source, extra_sfc_vars=config.EXTRA_SFC_INPUT,
                      extra_sfc_pad=13 - len(config.EXTRA_SFC_INPUT),
                      input_levels=input_levels, levels=levels_medium)


def build_meshes():
    return _mesh("neogfs-25", levels_gfs), _mesh("neohres-20", levels_hres)


def build_output_mesh():
    return LatLonGrid(source="era5-28", extra_sfc_vars=EXTRA_SFC_OUTPUT, levels=levels_medium)


def _norm_pr(vid, lev, a):
    k = levels_full.index(lev)
    return (a - NORM[vid]["mean"][k]) / NORM[vid]["std"][k]


def _norm_sfc(vid, a):
    return (a - NORM[vid]["mean"]) / NORM[vid]["std"]


def build_encoder_input(raw, mesh):
    lv = mesh.input_levels
    pr = np.zeros((config.NJ, config.NI, 5, len(lv)), np.float32)
    for vi, vid in enumerate(config.CORE_PRESSURE_VARS):
        for li, lev in enumerate(lv):
            a = raw["pressure"][vid].get(lev)
            if a is not None:
                pr[:, :, vi, li] = _norm_pr(vid, lev, a)

    sfc = np.zeros((config.NJ, config.NI, 4), np.float32)
    for si, vid in enumerate(config.CORE_SFC_VARS):
        sfc[:, :, si] = _norm_sfc(vid, raw["surface"][vid])

    ex = np.zeros((config.NJ, config.NI, 4), np.float32)
    for ei, vid in enumerate(config.EXTRA_SFC_INPUT):
        a = raw["extra"][vid]
        ex[:, :, ei] = np.clip(a / 100.0, 0, 1) if vid == "45_tcc" else _norm_sfc(vid, a)

    pr, sfc, ex = pr[:720], sfc[:720], ex[:720]
    pr_flat = pr.reshape(720, config.NI, 5 * len(lv))
    pad = np.zeros((720, config.NI, mesh.extra_sfc_pad), np.float32)
    x = torch.from_numpy(np.concatenate([pr_flat, sfc, ex, pad], axis=-1))
    return interp_levels(x, mesh, mesh.input_levels, mesh.levels)


def preprocess(raw):
    gfs_mesh, hres_mesh = build_meshes()
    gx = build_encoder_input(raw, gfs_mesh)
    hx = build_encoder_input(raw, hres_mesh)
    return gx, hx, gfs_mesh, hres_mesh
