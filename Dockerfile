FROM ubuntu:24.04 AS ffmpeg-build
ARG DEBIAN_FRONTEND=noninteractive
ARG FFMPEG_REV=d3ad8a7fee6a647c6362e4a105d949282d50a98f
ARG NV_CODEC_HEADERS=n13.0.19.0
RUN sed -i 's|http://|https://|g' /etc/apt/sources.list.d/ubuntu.sources \
 && apt-get update && apt-get install -y --no-install-recommends \
    build-essential clang nasm pkg-config git ca-certificates libx264-dev libx265-dev \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /build
RUN git clone --depth=1 --branch ${NV_CODEC_HEADERS} https://github.com/FFmpeg/nv-codec-headers.git headers \
 && make -C headers PREFIX=/usr/local install
RUN git init ffmpeg && cd ffmpeg && git remote add origin https://github.com/FFmpeg/FFmpeg.git \
 && git fetch --depth=1 origin ${FFMPEG_REV} && git checkout --detach FETCH_HEAD
WORKDIR /build/ffmpeg
RUN ./configure --prefix=/opt/ffmpeg --disable-doc --disable-debug --disable-autodetect \
    --disable-ffplay --enable-gpl --enable-libx264 --enable-libx265 \
    --enable-ffnvcodec --enable-cuda-llvm --enable-cuvid --enable-nvdec --enable-nvenc \
 && make -j4 && make install \
 && mkdir -p /opt/ffmpeg/share/source \
 && git archive HEAD | gzip > /opt/ffmpeg/share/source/ffmpeg-source.tar.gz \
 && cp COPYING* /opt/ffmpeg/share/source/ \
 && git -C /build/headers archive HEAD | gzip > /opt/ffmpeg/share/source/nv-codec-headers-source.tar.gz

FROM ubuntu:24.04 AS runtime
ARG DEBIAN_FRONTEND=noninteractive
RUN sed -i 's|http://|https://|g' /etc/apt/sources.list.d/ubuntu.sources \
 && apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv ca-certificates libx264-164 libx265-199 libgomp1 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ffmpeg-build /opt/ffmpeg /opt/ffmpeg
ENV PATH="/opt/venv/bin:/opt/ffmpeg/bin:${PATH}" \
    PYTHONUNBUFFERED=1 MEDIA_ROOT=/media/photos DATA_DIR=/data
WORKDIR /app
COPY requirements.txt .
RUN python3 -m venv /opt/venv && pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY analysis ./analysis
COPY LICENSE README.md ./
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
