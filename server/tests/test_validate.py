"""Tests for validate(): finiteness over the full output, physical bounds, and the
experimental-precip reporting. Run: `uv run python server/tests/test_validate.py`."""
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())

import numpy as np
from server.wm3pipe import preprocess, outputs
from server.wm3pipe.validate import validate

ERA = preprocess.build_output_mesh()
FV = ERA.full_varlist


def make_real():
    real = np.zeros((4, 4, ERA.n_vars), np.float32)
    def s(n, v): real[..., FV.index(n)] = v
    s("167_2t", 288); s("151_msl", 101300); s("165_10u", 5); s("166_10v", 5)
    s("131_u_250", 40); s("132_v_250", 10); s("168_2d", 285)
    for n in FV:
        if n.startswith("133_q_"):
            real[..., FV.index(n)] = 0.005
    s("142_lsp-6h", -10); s("143_cp-6h", -10)
    return real


def fields_of(real):
    return {n: real[..., FV.index(n)] for n in
            ["167_2t", "151_msl", "165_10u", "166_10v", "131_u_250", "132_v_250"]}


def check(name, cond):
    print(("PASS" if cond else "FAIL"), name)
    return cond


ok = True
# 1) clean output -> valid, zero non-finite; precip stays NOT reliable (transform unverified)
r = make_real(); m = validate(fields_of(r), r, ERA)
ok &= check("clean: valid", m["valid"])
ok &= check("clean: nonfinite==0", m["nonfinite_count"] == 0)
ok &= check("clean: precip_reliable False (unverified transform)", not m["precip_reliable"])

# 2) NaN in an UNSELECTED channel (humidity level not in the plotted subset) is caught
r = make_real(); r[..., FV.index("133_q_300")] = np.nan
m = validate(fields_of(r), r, ERA)
ok &= check("nan in unselected channel -> nonfinite>0", m["nonfinite_count"] > 0)
ok &= check("nan -> valid False", not m["valid"])

# 3) dewpoint > temperature everywhere -> invalid
r = make_real(); r[..., FV.index("168_2d")] = 320.0
m = validate(fields_of(r), r, ERA)
ok &= check("Td>T -> dewpoint_gt_temp_frac ~1", m["dewpoint_gt_temp_frac"] > 0.9)
ok &= check("Td>T -> valid False", not m["valid"])

# 4a) huge-but-finite precip de-transform -> flagged via capped fraction, not vacuously passed
r = make_real(); r[..., FV.index("142_lsp-6h")] = 50.0   # exp(50) ~ 5e21 (finite)
m = validate(fields_of(r), r, ERA)
ok &= check("precip huge -> capped_frac high", m["precip_capped_frac"] > 0.5)
ok &= check("precip huge -> precip_reliable False", not m["precip_reliable"])
# 4b) genuine +inf from overflow -> caught by finiteness
r = make_real(); r[..., FV.index("142_lsp-6h")] = 800.0   # exp(800) -> +inf
m = validate(fields_of(r), r, ERA)
ok &= check("precip +inf -> raw_finite False", not m["precip_raw_finite"])

print("\nRESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
