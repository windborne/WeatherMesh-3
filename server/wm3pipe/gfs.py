"""Fetch a GFS 0.25-degree initial condition from the NOAA Open Data bucket.

Only the fields the model needs are pulled, using the GRIB `.idx` catalog to
issue HTTP range requests instead of downloading the full ~500 MB file.
"""
import io
import logging
from datetime import datetime, timedelta, timezone

import boto3
import numpy as np
from botocore import UNSIGNED
from botocore.client import Config
from eccodes import codes_new_from_message, codes_get_values, codes_release

from . import config

log = logging.getLogger("wm3pipe.gfs")

_s3 = boto3.client("s3", region_name=config.REGION, config=Config(signature_version=UNSIGNED))


def _key(date, hour):
    return f"gfs.{date}/{hour:02d}/atmos/gfs.t{hour:02d}z.pgrb2.{config.GFS_RESOLUTION}.f000"


def _exists(key):
    try:
        _s3.head_object(Bucket=config.GFS_BUCKET, Key=key + ".idx")
        return True
    except Exception:
        return False


def latest_available_cycle(now=None):
    """Return (yyyymmdd, hour) of the most recent GFS cycle whose f000 is published."""
    now = now or datetime.now(timezone.utc)
    probe = now.replace(minute=0, second=0, microsecond=0)
    probe -= timedelta(hours=probe.hour % 6)
    for back in range(0, 6):
        c = probe - timedelta(hours=6 * back)
        if _exists(_key(c.strftime("%Y%m%d"), c.hour)):
            return c.strftime("%Y%m%d"), c.hour
    raise RuntimeError("no GFS cycle with a published f000 found in the last 36h")


def _parse_idx(text):
    recs = []
    for line in text.strip().splitlines():
        parts = line.split(":")
        recs.append({"n": int(parts[0]), "start": int(parts[1]), "var": parts[3], "level": parts[4]})
    for i, r in enumerate(recs):
        r["end"] = recs[i + 1]["start"] - 1 if i + 1 < len(recs) else None
    return recs


def _download(key, start, end):
    rng = f"bytes={start}-{end}" if end is not None else f"bytes={start}-"
    return _s3.get_object(Bucket=config.GFS_BUCKET, Key=key, Range=rng)["Body"].read()


def _values(msg):
    gid = codes_new_from_message(msg)
    try:
        return codes_get_values(gid).astype(np.float32).reshape(config.NJ, config.NI)
    finally:
        codes_release(gid)


def fetch(date, hour, levels):
    """Fetch model-relevant GFS fields for one cycle.

    Returns dict with:
      pressure[var_id][level_mb] -> (721,1440) in model units
      surface[var_id]            -> (721,1440)
      extra[var_id]              -> (721,1440)
    Missing pressure fields (e.g. high-altitude humidity absent in GFS) are omitted.
    """
    key = _key(date, hour)
    idx = _parse_idx(_download(key + ".idx", 0, None).decode())
    by_key = {(r["var"], r["level"]): r for r in idx}

    wanted = []
    for vid, (tok, _scale) in config.GFS_PRESSURE.items():
        for lev in levels:
            r = by_key.get((tok, f"{lev} mb"))
            if r:
                wanted.append(("P", vid, lev, r))
    for vid, (tok, lev) in config.GFS_SURFACE.items():
        r = by_key.get((tok, lev))
        if r:
            wanted.append(("S", vid, None, r))
    for vid, (tok, lev, _rep) in config.GFS_EXTRA.items():
        r = by_key.get((tok, lev))
        if r:
            wanted.append(("E", vid, None, r))

    out = {"pressure": {v: {} for v in config.GFS_PRESSURE}, "surface": {}, "extra": {}}
    log.info("GFS %s/%02dZ: fetching %d fields", date, hour, len(wanted))
    for kind, vid, lev, r in wanted:
        arr = _values(_download(key, r["start"], r["end"]))
        if kind == "P":
            out["pressure"][vid][lev] = arr * config.GFS_PRESSURE[vid][1]
        elif kind == "S":
            out["surface"][vid] = arr
        else:
            out["extra"][vid] = arr
    return out
