FROM python:3.11-slim-bookworm

LABEL maintainer="netplexflix"
LABEL description="ULDAS - Unified Language Detection and Subtitle Processing (NVIDIA GPU with CPU fallback)"
LABEL org.opencontainers.image.source="https://github.com/netplexflix/ULDAS"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Runtime-only system deps. All Python packages ship prebuilt wheels, so no
# compiler or -dev headers are needed. libgomp1 is required by the ctranslate2
# wheel on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    xz-utils \
    mkvtoolnix \
    tesseract-ocr \
    tesseract-ocr-eng \
    gosu \
    tzdata \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install a patched static FFmpeg (>= 8.1.2) to fix CVE-2026-8461
ARG FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz
RUN curl -fsSL -o /tmp/ffmpeg.tar.xz "$FFMPEG_URL" \
    && mkdir -p /tmp/ffmpeg \
    && tar -xJf /tmp/ffmpeg.tar.xz -C /tmp/ffmpeg --strip-components=1 \
    && install -m 0755 /tmp/ffmpeg/bin/ffmpeg /usr/local/bin/ffmpeg \
    && install -m 0755 /tmp/ffmpeg/bin/ffprobe /usr/local/bin/ffprobe \
    && rm -rf /tmp/ffmpeg /tmp/ffmpeg.tar.xz \
    && /usr/local/bin/ffmpeg -version | head -n1

WORKDIR /app

# Python dependencies. faster-whisper runs on CTranslate2 (CPU / CUDA only);
# the two NVIDIA runtime libraries it needs for CUDA are pulled from pip so no
# CUDA base image is required. On hosts without an NVIDIA GPU / runtime the
# app auto-detects this and falls back to CPU.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install "nvidia-cublas-cu12" "nvidia-cudnn-cu12==9.*"

# Make the pip-provided CUDA libraries discoverable by CTranslate2
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib

# Copy application code
COPY ULDAS.py .
COPY uldas ./uldas

# Create directories and copy config
RUN mkdir -p /app/config /media
COPY config/config.example.yml /app/config.example.yml
COPY entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

# Keep the Whisper model cache on the config volume so it survives container
# recreation instead of living in the writable layer.
ENV HF_HOME=/app/config/.cache/huggingface

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PUID=0
ENV PGID=0

EXPOSE 2119

ENTRYPOINT ["/entrypoint.sh"]
