# WeatherMesh-3 real-time inference container (SageMaker Processing / any GPU host).
# Build from the repo root:  docker build -f server/Dockerfile -t wm3-pipeline .
# Weights are NOT baked in (1.2 GB) — fetched from S3 at runtime.
# Base already ships Python 3.11 + torch 2.5.1 + CUDA 12.1, so no apt / no torch reinstall.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV WM3_DEVICE=cuda PIP_ROOT_USER_ACTION=ignore PIP_NO_CACHE_DIR=1
WORKDIR /opt/wm3

COPY pyproject.toml utils.py meshes.py model.py ./
COPY model_latlon ./model_latlon
COPY constants ./constants
COPY server ./server

# natten from a vendored wheel (server/wheels/, fetched from S3 before build) — the
# upstream shi-labs host has an expired cert and times out, so we don't build against it.
# matepoint (test.pypi) + pipeline deps come from reliable indexes.
RUN pip install server/wheels/natten-*.whl \
    && pip install "matepoint" \
        --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ \
    && pip install -r server/requirements-pipeline.txt

ENTRYPOINT ["python", "-m", "server.wm3pipe.run_cycle"]
