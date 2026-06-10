# ---------------------------------------------------------------------------
# RunPod Serverless Dockerfile for TripoSR
# ---------------------------------------------------------------------------
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
WORKDIR /app

# Install PyTorch + CUDA (already present in base image, but pin versions if needed)
RUN pip install --no-cache-dir \
    runpod \
    omegaconf==2.3.0 \
    Pillow==10.1.0 \
    einops==0.7.0 \
    transformers==4.35.0 \
    trimesh==4.0.5 \
    rembg \
    huggingface-hub \
    imageio[ffmpeg] \
    xatlas==0.0.9 \
    moderngl==5.10.0 \
    numpy \
    requests \
    onnxruntime-gpu

# torchmcubes must be built from source (no wheel on PyPI)
RUN pip install --no-cache-dir git+https://github.com/tatsy/torchmcubes.git

# ---------------------------------------------------------------------------
# Install TripoSR
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir git+https://github.com/VAST-AI-Research/TripoSR.git

# ---------------------------------------------------------------------------
# Pre-download model weights so cold-start is faster
# ---------------------------------------------------------------------------
RUN python -c "from tsr.system import TSR; TSR.from_pretrained('stabilityai/TripoSR', config_name='config.yaml', weight_name='model.ckpt')"

# ---------------------------------------------------------------------------
# Copy handler
# ---------------------------------------------------------------------------
COPY rp_handler.py /rp_handler.py

# ---------------------------------------------------------------------------
# RunPod serverless entrypoint
# ---------------------------------------------------------------------------
CMD [ "python", "-u", "/rp_handler.py" ]
