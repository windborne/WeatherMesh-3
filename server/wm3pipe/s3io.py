"""Upload forecast artifacts to S3 under a tidy, Hive-partitioned layout."""
import json
import logging

import boto3

from . import config

log = logging.getLogger("wm3pipe.s3io")
_s3 = boto3.client("s3", region_name=config.REGION)


def prefix(init_iso, lead_hours, valid=True):
    # invalid forecasts go to a quarantine tree and never become `latest.json`
    root = "forecasts" if valid else "quarantine"
    return f"{root}/init={init_iso}/lead={lead_hours:03d}h"


def upload_cycle(local_files, init_iso, lead_hours, bucket=None, valid=True):
    """local_files: dict s3-relative-name -> local path. Returns dict of s3 URIs."""
    bucket = bucket or config.OUTPUT_BUCKET
    base = prefix(init_iso, lead_hours, valid)
    uris = {}
    for name, path in local_files.items():
        key = f"{base}/{name}"
        _s3.upload_file(str(path), bucket, key)
        uris[name] = f"s3://{bucket}/{key}"
        log.info("uploaded %s", uris[name])
    return uris


def write_pointer(init_iso, lead_hours, uris, meta, bucket=None, valid=True):
    """Write the per-init metadata.json. Only a VALID cycle updates latest.json."""
    bucket = bucket or config.OUTPUT_BUCKET
    root = "forecasts" if valid else "quarantine"
    doc = {"init": init_iso, "lead_hours": lead_hours, "valid": valid,
           "artifacts": uris, "metadata": meta}
    body = json.dumps(doc, indent=2, default=str).encode()
    _s3.put_object(Bucket=bucket, Key=f"{root}/init={init_iso}/metadata.json", Body=body)
    if valid:
        _s3.put_object(Bucket=bucket, Key="latest.json", Body=body)
    return f"s3://{bucket}/latest.json" if valid else f"s3://{bucket}/{root}/init={init_iso}/metadata.json"


def write_manifest(bucket=None):
    bucket = bucket or config.OUTPUT_BUCKET
    text = (
        "WeatherMesh-3 forecast outputs\n\n"
        "Layout:\n"
        "  forecasts/init=<YYYY-MM-DDTHHZ>/lead=<NNNh>/weathermesh3.fNNN.nc   CF netCDF\n"
        "  forecasts/init=<...>/lead=<...>/overview.png                        proof plot\n"
        "  forecasts/init=<...>/metadata.json                                  run provenance\n"
        "  latest.json                                                         newest cycle pointer\n"
    )
    _s3.put_object(Bucket=bucket, Key="README.txt", Body=text.encode())
