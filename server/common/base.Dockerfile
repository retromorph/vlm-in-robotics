ARG CUDA_TAG=12.4.0
ARG PYTHON_VERSION=3.10
ARG TORCH_VERSION=2.2.0

FROM nvidia/cuda:${CUDA_TAG}-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---- system packages ------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-distutils \
        python3-pip \
        ca-certificates && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    pip install --upgrade pip

# ---- core Python libs (Torch only) ----------------------------------------
RUN pip install \
      torch==${TORCH_VERSION}+cu124 \
      --extra-index-url https://download.pytorch.org/whl/cu124

# ---- clean up layer size --------------------------------------------------
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

CMD ["/bin/bash"]
