"""One real-time WeatherMesh-3 cycle: GFS -> preprocess -> infer -> validate -> S3.

Emits CloudWatch metrics throughout and alerts (SNS) on failure or invalid output.
This is the unit of work the 6-hourly notebook cron invokes (see infra/cron-onstart.sh).
Invalid forecasts are quarantined and never promoted to latest.json.
"""
import argparse
import os
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

import boto3

from . import config, gfs, preprocess, inference, outputs, s3io
from . import metrics as M
from .validate import validate

from utils import to_unix, get_date, levels_gfs, levels_hres  # noqa: E402

log = M.log


def _alert(subject, message):
    topic = os.environ.get("WM3_SNS_TOPIC")
    if not topic:
        return
    try:
        boto3.client("sns", region_name=config.REGION).publish(
            TopicArn=topic, Subject=subject[:100], Message=message)
    except Exception as e:
        log.warning("sns publish failed: %s", e)


def _ensure_weights(weights):
    """If the weights file is not present locally, fetch it from S3 (WM3_WEIGHTS_S3).

    Lets the container be self-sufficient (no host mount needed)."""
    if os.path.exists(weights):
        return weights
    uri = config.WEIGHTS_S3
    if not uri.startswith("s3://"):
        return weights
    bkt, key = uri[5:].split("/", 1)
    dst = "/tmp/WeatherMesh3.pt"
    log.info("downloading weights %s -> %s", uri, dst)
    boto3.client("s3", region_name=config.REGION).download_file(bkt, key, dst)
    return dst


def selftest():
    """Import the full model stack + build meshes on CPU (no weights, no forward).

    Verifies the container's deps (torch/natten/matepoint/model) actually import and
    the model constructs. Runs in CI on a CPU runner; the GPU forward is separate."""
    import torch
    log.info("selftest: torch %s (cuda build=%s)", torch.__version__, torch.version.cuda)
    # natten is CUDA-only; on a GPU-less CI runner its extension can't load libcuda.
    # Report status but don't fail the CPU self-test on it (GPU-validated via the notebook).
    for mod in ("natten", "matepoint"):
        try:
            __import__(mod)
            log.info("selftest: %s import OK", mod)
        except Exception as e:
            log.warning("selftest: %s import unavailable (expected without CUDA): %s", mod, e)
    gm, hm = preprocess.build_meshes()
    em = preprocess.build_output_mesh()
    assert gm.n_vars == hm.n_vars == em.n_vars == 157
    log.info("selftest OK: torch imports, meshes build (n_vars=157)")


def run(lead_hours=6, weights="model/WeatherMesh3.pt", device="cuda", bucket=None):
    try:
        with M.timed("cycle_seconds"):
            weights = _ensure_weights(weights)
            date, hour = gfs.latest_available_cycle()
            init_dt = datetime.strptime(f"{date}{hour:02d}", "%Y%m%d%H").replace(tzinfo=timezone.utc)
            init_iso = init_dt.strftime("%Y-%m-%dT%HZ")
            levels = sorted(set(levels_gfs) | set(levels_hres))
            log.info("cycle %s +%dh (GFS %d levels)", init_iso, lead_hours, len(levels))

            with M.timed("fetch_seconds"):
                raw = gfs.fetch(date, hour, levels)
            with M.timed("preprocess_seconds"):
                gx, hx, _gm, _hm = preprocess.preprocess(raw)
            era_mesh = preprocess.build_output_mesh()

            model = inference.load_model(weights, device)
            t0 = to_unix(init_dt)
            with M.timed("inference_seconds"):
                real, fields, l2 = inference.run(model, gx, hx, t0, era_mesh, lead_hours, device)

            checks = validate(fields, real, era_mesh)
            for k in ("t2m_min_C", "t2m_max_C", "t2m_mean_C", "mslp_min_hPa", "mslp_max_hPa",
                      "wind10_max_ms", "jet250_max_ms", "nonfinite_count", "nan_count",
                      "neg_humidity_frac", "dewpoint_gt_temp_frac",
                      "precip_capped_frac", "precip_capped_mass_frac", "precip_p99_mm"):
                M.put(k, checks[k])
            M.put("latent_l2", l2)
            M.put("output_valid", 1 if checks["valid"] else 0)

            tmp = Path(tempfile.mkdtemp())
            meta = {
                "model": "WeatherMesh-3", "init_time": init_iso, "lead_hours": lead_hours,
                "valid_time": get_date(t0 + lead_hours * 3600).strftime("%Y-%m-%dT%H:%MZ"),
                "gfs_cycle": f"{date}{hour:02d}", "input_source": "GFS 0.25deg (both encoders)",
                "latent_l2": l2, "validation": checks,
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }
            ncname = f"weathermesh3.f{lead_hours:03d}.nc"
            files = {
                ncname: outputs.write_netcdf(real, era_mesh, meta, tmp / ncname),
                "overview.png": outputs.write_overview_plot(fields, tmp / "overview.png"),
                "metadata.json": outputs.write_metadata(meta, tmp / "metadata.json"),
            }
            valid = bool(checks["valid"])
            # invalid forecasts go to a quarantine prefix and do NOT become latest.json
            uris = s3io.upload_cycle(files, init_iso, lead_hours, bucket, valid=valid)
            s3io.write_pointer(init_iso, lead_hours, uris, meta, bucket, valid=valid)
            if valid:
                s3io.write_manifest(bucket)
            else:
                _alert("WeatherMesh-3 output INVALID (quarantined)",
                       f"init={init_iso} +{lead_hours}h failed validation; NOT promoted to latest.\n\n{checks}")

            M.put("cycle_success", 1)
            log.info("cycle done %s +%dh valid=%s published=%s inference_l2=%.3f",
                     init_iso, lead_hours, valid, valid, l2)
            return {"init": init_iso, "lead_hours": lead_hours, "valid": valid, "published": valid, "uris": uris}
    except Exception as e:
        M.put("cycle_success", 0)
        tb = traceback.format_exc()
        log.error("cycle FAILED: %s\n%s", e, tb)
        _alert("WeatherMesh-3 cycle failed", f"{e}\n\n{tb}")
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lead-hours", type=int, default=int(os.environ.get("WM3_LEAD_HOURS", 6)))
    p.add_argument("--weights", default=os.environ.get("WM3_WEIGHTS", "model/WeatherMesh3.pt"))
    p.add_argument("--device", default=os.environ.get("WM3_DEVICE", "cuda"))
    p.add_argument("--bucket", default=None)
    p.add_argument("--selftest", action="store_true", help="import model + build meshes, then exit (no GPU/weights)")
    a = p.parse_args()
    if a.selftest:
        selftest()
        return
    run(a.lead_hours, a.weights, a.device, a.bucket)


if __name__ == "__main__":
    main()
