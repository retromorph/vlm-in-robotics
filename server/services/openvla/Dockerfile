# leverage common CUDA + python base to speed up multi-build
FROM vla-base:latest

WORKDIR /workspace

# --- copy only what we need to leverage Docker cache -------------
COPY app/ ./app/
COPY openvla_inference.py ./app/

# --- build the Bento -------------------------------------------------------
RUN bentoml build -f bentofile.yaml

# ---- final runtime image (tiny) ------------------------------------------
FROM vla-serving-base:runtime

COPY --from=0 /workspace/bentos/*.bento /bentos/
CMD ["bentoml", "serve", "/bentos/*.bento", "--production", "--workers", "1"]
