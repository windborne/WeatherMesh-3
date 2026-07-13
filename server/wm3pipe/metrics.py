"""CloudWatch metrics + timing helpers for pipeline observability."""
import contextlib
import logging
import time

import boto3

from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("wm3pipe")

_cw = None


def _client():
    global _cw
    if _cw is None:
        _cw = boto3.client("cloudwatch", region_name=config.REGION)
    return _cw


def put(name, value, unit="None", dims=None):
    dims = dims or {"pipeline": "weathermesh3"}
    try:
        _client().put_metric_data(Namespace=config.CW_NAMESPACE, MetricData=[{
            "MetricName": name, "Value": float(value), "Unit": unit,
            "Dimensions": [{"Name": k, "Value": str(v)} for k, v in dims.items()],
        }])
    except Exception as e:  # never let telemetry break the pipeline
        log.warning("cloudwatch put %s failed: %s", name, e)


@contextlib.contextmanager
def timed(metric):
    t = time.time()
    yield
    dt = time.time() - t
    log.info("%s took %.1fs", metric, dt)
    put(metric, dt, "Seconds")
