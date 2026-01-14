FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

LABEL maintainer="netplexflix"
LABEL description="ULDAS - Unified Language Detection and Subtitle Processing (NVIDIA GPU)"
LABEL org.opencontainers.image.source="https://github.com/netplexflix/ULDAS"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    ffmpeg \
    mkvtoolnix \
    tesseract-ocr \
    tesseract-ocr-eng \
    gosu \
    # Required for PyAV
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Upgrade pip first
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install PyAV (required by faster-whisper)
RUN pip install av

# Install faster-whisper and remaining dependencies
COPY requirements.txt .
RUN pip install faster-whisper>=1.2.0 && \
    pip install PyYAML>=6.0.2 requests>=2.32.4 packaging>=21.3 \
    psutil>=7.0.0 langdetect>=1.0.8 pytesseract>=0.3.13 pillow>=11.3.0

COPY ULDAS.py .
RUN mkdir -p /app/config /media
COPY config/config.example.yml /app/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PUID=0
ENV PGID=0

ENTRYPOINT ["/entrypoint.sh"]