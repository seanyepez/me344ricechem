# syntax=docker/dockerfile:1.7

FROM python:3.11.9-slim-bookworm AS cpu
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /workspace
COPY requirements-common.txt /tmp/requirements-common.txt
RUN python -m pip install --upgrade pip==24.2 \
    && python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r /tmp/requirements-common.txt
COPY src/ /workspace/src/
RUN useradd --create-home --uid 10001 runner \
    && mkdir -p /data /models /outputs \
    && chown -R runner:runner /workspace /data /models /outputs
USER runner
ENTRYPOINT ["python", "src/serve_transformers.py"]

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS gpu
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /workspace
COPY requirements-common.txt requirements-gpu.txt /tmp/
RUN python -m pip install --upgrade pip==24.2 \
    && python -m pip install -r /tmp/requirements-gpu.txt
COPY src/ /workspace/src/
RUN useradd --create-home --uid 10001 runner \
    && mkdir -p /data /models /outputs \
    && chown -R runner:runner /workspace /data /models /outputs
USER runner
ENTRYPOINT ["python", "src/train_gpu_4b.py"]
