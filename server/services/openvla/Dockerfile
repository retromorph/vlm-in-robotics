FROM python:3.10-slim
WORKDIR /app

# Install system deps for OpenVLA (e.g., ffmpeg, libgl1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY openvla_service.py ./
COPY common/ ./common/

# BentoML serve command
CMD ["bentoml", "serve", "openvla_service:svc", "--port", "${BENTOML_PORT}", "--workers", "4"]
