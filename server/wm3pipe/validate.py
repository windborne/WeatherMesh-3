"""Physical-plausibility checks on a forecast; returns a metrics dict + validity.

Validity gates on: finiteness of the FULL output (all channels), physical bounds
(temp/pressure/wind/jet), and bounded humidity/dewpoint artefacts. Precipitation is
treated as EXPERIMENTAL (uncertain log de-transform): its reliability is reported
separately via `precip_reliable`/`precip_capped_frac` rather than gating `valid`
vacuously.

Thresholds (documented, exercised by tests/test_validate.py):
  NEG_Q_FRAC_MAX / TD_GT_T_FRAC_MAX = 0.05  -> tolerate <=5% of the known small ML
     artefact cells (they are clamped in the written netCDF); more than that signals a
     genuinely broken field.
  PRECIP_CAPPED_FRAC_MAX = 0.01 -> >1% of cells hitting the precip cap means the
     de-transform is unreliable for that cycle.
"""
import numpy as np

from . import outputs

NEG_Q_FRAC_MAX = 0.05
TD_GT_T_FRAC_MAX = 0.05
PRECIP_CAPPED_FRAC_MAX = 0.0001
PRECIP_CAPPED_MASS_FRAC_MAX = 0.05
# The log-space precip inverse (exp) is NOT verified against training semantics or
# reference data, and a tiny capped tail can dominate the global mean. So precipitation
# is never labelled reliable until that verification exists — it stays experimental.
PRECIP_TRANSFORM_VERIFIED = False


def validate(fields, real=None, era_mesh=None):
    t2c = fields["167_2t"] - 273.15
    mslp = fields["151_msl"] / 100.0
    wind10 = np.sqrt(fields["165_10u"] ** 2 + fields["166_10v"] ** 2)
    jet250 = np.sqrt(fields["131_u_250"] ** 2 + fields["132_v_250"] ** 2)

    # finiteness over the FULL output (all channels), not just the plotted subset
    if real is not None:
        nonfinite = int((~np.isfinite(real)).sum())
    else:
        nonfinite = int(sum(int((~np.isfinite(v)).sum()) for v in fields.values()))

    m = {
        "t2m_min_C": float(np.nanmin(t2c)), "t2m_max_C": float(np.nanmax(t2c)),
        "t2m_mean_C": float(np.nanmean(t2c)),
        "mslp_min_hPa": float(np.nanmin(mslp)), "mslp_max_hPa": float(np.nanmax(mslp)),
        "wind10_max_ms": float(np.nanmax(wind10)), "jet250_max_ms": float(np.nanmax(jet250)),
        "nonfinite_count": nonfinite, "nan_count": nonfinite,
    }

    neg_q_frac = td_gt_t_frac = 0.0
    precip_raw_finite = True
    precip_capped_frac = 0.0
    precip_capped_mass_frac = 0.0
    precip_p99_mm = 0.0
    if real is not None and era_mesh is not None:
        fv = era_mesh.full_varlist
        qcols = [i for i, n in enumerate(fv) if n.startswith("133_q_")]
        if qcols:
            neg_q_frac = float((real[..., qcols] < 0).mean())
        td = real[..., fv.index("168_2d")]
        t2 = real[..., fv.index("167_2t")]
        td_gt_t_frac = float((td > t2 + 0.1).mean())
        raw = outputs.total_precip_6h_mm(real, era_mesh, clamp=False)
        precip_raw_finite = bool(np.isfinite(raw).all())
        clamped = np.clip(np.nan_to_num(raw, nan=0.0, posinf=outputs.PRECIP_CAP_MM, neginf=0.0),
                          0, outputs.PRECIP_CAP_MM)
        capped = clamped >= outputs.PRECIP_CAP_MM
        precip_capped_frac = float(capped.mean())
        precip_p99_mm = float(np.percentile(clamped, 99))
        total = float(clamped.sum())
        # how much of the total precip mass sits in the (unreliable) capped tail
        precip_capped_mass_frac = float(clamped[capped].sum() / total) if total > 0 else 0.0

    m.update({
        "neg_humidity_frac": neg_q_frac, "dewpoint_gt_temp_frac": td_gt_t_frac,
        "precip_raw_finite": precip_raw_finite, "precip_capped_frac": precip_capped_frac,
        "precip_capped_mass_frac": precip_capped_mass_frac, "precip_p99_mm": precip_p99_mm,
    })

    m["valid"] = bool(
        nonfinite == 0
        and -95 < m["t2m_min_C"] and m["t2m_max_C"] < 65
        and 850 < m["mslp_min_hPa"] and m["mslp_max_hPa"] < 1095
        and m["wind10_max_ms"] < 150 and 30 < m["jet250_max_ms"] < 200
        and neg_q_frac < NEG_Q_FRAC_MAX and td_gt_t_frac < TD_GT_T_FRAC_MAX
    )
    # precipitation reliability is reported, NOT folded into `valid` (experimental product).
    # It stays False until the inverse transform is verified AND the capped tail is
    # negligible in both count and mass contribution.
    m["precip_reliable"] = bool(
        PRECIP_TRANSFORM_VERIFIED and precip_raw_finite
        and precip_capped_frac < PRECIP_CAPPED_FRAC_MAX
        and precip_capped_mass_frac < PRECIP_CAPPED_MASS_FRAC_MAX
    )
    return m
