# syntax=docker/dockerfile:1.6   # enables `--mount=type=cache=` & friends
ARG CUDA_TAG=12.4.0
ARG PYTHON_VERSION=3.10
ARG TORCH_VERSION=2.2.0

############################
# 1️⃣  Build stage
############################
FROM nvidia/cuda:${CUDA_TAG}-devel-ubuntu22.04 AS builder
LABEL stage=builder

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# ――― basic OS tooling ―――――――――――――――――――――――――――――――――――――――――――――――――
RUN apt-get update && apt-get install -y --no-install-recommends \
         python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python3-pip \
         build-essential git wget curl ca-certificates cmake ninja-build \
         ffmpeg pkg-config && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    pip install --upgrade pip setuptools wheel

# ――― common Python libs (Torch wheels + BentoML CLI) ――――――――――――――――――――――
# Torch wheels for CUDA 12.4 come from the PyTorch nightly/RC index   🔗
RUN pip install --no-build-isolation \
      torch==${TORCH_VERSION}+cu124 torchvision==0.17.0+cu124 \
      --extra-index-url https://download.pytorch.org/whl/cu124  && \
    pip install bentoml packaging numpy pillow "opencv-python-headless>=4.9" \
               ninja  # ninja is needed by flash-attn builds

# mark the location of site-packages so we can copy it later
RUN python - <<'PY'
import site, pathlib, json, sysconfig
path = site.getsitepackages()[0]
pathlib.Path("/site.txt").write_text(path)
print("Site-packages:", path)
PY

############################
# 2️⃣  Runtime stage
############################
FROM nvidia/cuda:${CUDA_TAG}-runtime-ubuntu22.04 AS runtime
LABEL maintainer="you@example.com"

ARG PYTHON_VERSION
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1
ENV PATH="/home/vla/.local/bin:${PATH}"

# tiny runtime deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
         python${PYTHON_VERSION} python${PYTHON_VERSION}-distutils \
         python3-pip ffmpeg ca-certificates git && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    pip install --upgrade pip

# copy Python environment from builder without the entire rootfs
ARG SITE_PKGS_DIR
COPY --from=builder /site.txt /tmp/
RUN SITE=$(cat /tmp/site.txt) && \
    mkdir -p ${SITE%/*} && \
    rsync -a --delete --exclude='**/__pycache__' \
          --from=builder ${SITE}/ ${SITE}/

# non-root user for security
RUN useradd -m -u 1000 vla
USER vla
WORKDIR /workspace

CMD ["/bin/bash"]

############################
# 3️⃣  Export aliases
############################
FROM runtime AS vla-serving-base
FROM runtime AS vla-serving-base:runtime
