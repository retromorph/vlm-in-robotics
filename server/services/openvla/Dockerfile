# leverage common CUDA + python base to speed up multi-build
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    tree \
    less \
    vim \
    curl \
    wget \
    build-essential \
    python3-pip \
    mesa-utils \
    sudo \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# --- copy only what we need to leverage Docker cache -------------
COPY app/ ./app/
COPY openvla_inference.py ./app/

# --- build the Bento -------------------------------------------------------
RUN bentoml build -f bentofile.yaml

# ---- final runtime image (tiny) ------------------------------------------
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

COPY --from=0 /workspace/bentos/*.bento /bentos/
CMD ["bentoml", "serve", "/bentos/*.bento", "--production", "--workers", "1"]
