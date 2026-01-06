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
    python3-pip \
    ffmpeg \
    mkvtoolnix \
    tesseract-ocr \
    tesseract-ocr-eng \
    gosu \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

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
