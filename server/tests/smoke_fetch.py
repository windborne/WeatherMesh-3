"""Local smoke test: fetch a few GFS levels, preprocess, validate shapes/ranges.

Runs on CPU (no model). Fills only the fetched levels; the rest stay at the
normalized mean (0), so the assembled shape is exercised end to end while the
download stays small. Full-level runs happen in-region in the processing job.
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())

import numpy as np
from server.wm3pipe import gfs, preprocess

TEST_LEVELS = [850, 500, 250]

t0 = time.time()
date, hour = gfs.latest_available_cycle()
print(f"cycle: {date} {hour:02d}Z")
raw = gfs.fetch(date, hour, TEST_LEVELS)
print(f"fetch: {time.time()-t0:.1f}s")

print("\nraw sanity (physical units):")
print(f"  2m temp mean   : {raw['surface']['167_2t'].mean():.1f} K")
print(f"  mslp mean      : {raw['surface']['151_msl'].mean()/100:.1f} hPa")
print(f"  500hPa T mean  : {raw['pressure']['130_t'][500].mean():.1f} K")
print(f"  500hPa z mean  : {raw['pressure']['129_z'][500].mean()/9.80665:.0f} gpm")
print(f"  tcc range      : {raw['extra']['45_tcc'].min():.0f}..{raw['extra']['45_tcc'].max():.0f} %")
print(f"  levels fetched : {sorted(raw['pressure']['130_t'])}")

gx, hx, gfs_mesh, hres_mesh = preprocess.preprocess(raw)
print(f"\nencoder inputs: gfs {tuple(gx.shape)}  hres {tuple(hx.shape)}")
assert gx.shape == (720, 1440, gfs_mesh.n_vars)
assert hx.shape == (720, 1440, hres_mesh.n_vars)

i2t = gfs_mesh.n_pr + gfs_mesh.core_sfc_vars.index("167_2t")
imsl = gfs_mesh.n_pr + gfs_mesh.core_sfc_vars.index("151_msl")
it500 = gfs_mesh.pressure_vars.index("130_t") * gfs_mesh.n_levels + gfs_mesh.levels.index(500)
for name, idx in [("2t", i2t), ("msl", imsl), ("t@500", it500)]:
    ch = gx[..., idx].numpy()
    print(f"  normed {name:6}: mean {ch.mean():+.2f} std {ch.std():.2f} min {ch.min():+.2f} max {ch.max():+.2f}")

ok = -8 < gx[..., i2t].numpy().mean() < 8 and gx.shape[-1] == 157
print("\nSMOKE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
