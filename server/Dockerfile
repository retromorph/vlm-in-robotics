# Use official Python runtime
FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Install system deps for opencv
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Tell BentoML to build the archive
RUN bentoml build

# Expose BentoML’s default port
EXPOSE 3000

# Run the BentoML server with Gunicorn
CMD ["bentoml", "serve-gunicorn", "openvla_service:latest", "--workers", "1", "--timeout", "120"]
