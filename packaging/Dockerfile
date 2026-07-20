# GS-IR - distributable, multi-GPU image (Turing -> Hopper).
#
# Build from the SOURCE ROOT:
#     docker build -f packaging/Dockerfile -t ghcr.io/costar/gs-ir:latest .
#
# NOTE: Blackwell (RTX 50-series, sm_120) is NOT covered - it needs a CUDA 12.8
#       base and driver 570+.
FROM nvcr.io/nvidia/pytorch:24.04-py3

ENV DEBIAN_FRONTEND=noninteractive

# Multi-GPU support:
#   7.5 Turing (20-series / T4) · 8.0 A100 · 8.6 30-series · 8.9 Ada (40-series) · 9.0 Hopper (H100/H200)
# +PTX lets a newer driver JIT for future minor archs. Does NOT reach Blackwell (12.0).
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX"

# Baked, world-readable model cache so lpips/torchvision weights are present offline
# and reachable no matter which UID `--user` runs the container as.
ENV TORCH_HOME=/workspace/.cache/torch

# System deps: colmap + ffmpeg drive the raw-images/video preprocessing;
# the rest are GL/X libs.
RUN set -xe \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
        curl gnupg2 wget unzip git dos2unix ffmpeg \
        libx11-6 libglib2.0-0 libxrender1 libxext6 libgl1 libgomp1 colmap \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# The build context is the source root, so this copies the whole source tree.
COPY . /workspace

# Python deps.
RUN pip install --no-cache-dir \
        plyfile==0.8.1 tqdm trimesh imageio kornia lpips

# Build the CUDA extensions once, for every arch in TORCH_CUDA_ARCH_LIST.
RUN cd gs-ir && python setup.py develop
# nvdiffrast PINNED to v0.4.0 via pip.
# --no-build-isolation: nvdiffrast's setup.py imports torch at build time, so it must see
# the already-installed torch rather than a fresh isolated build env.
RUN pip install --no-cache-dir --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0"
RUN cd submodules \
 && pip install --no-cache-dir --no-build-isolation ./simple-knn \
 && pip install --no-cache-dir --no-build-isolation ./diff-gaussian-rasterization

# Pre-download the lpips backbones so the run needs no network and no writable HOME.
RUN python -c "import lpips; lpips.LPIPS(net='vgg'); lpips.LPIPS(net='alex')" || true \
 && chmod -R a+rX /workspace/.cache

# Put the entrypoint + preprocessing scripts at the workspace root (normalise CRLF).
RUN cp packaging/run_pipeline.sh packaging/preprocess.sh /workspace/ \
 && dos2unix /workspace/run_pipeline.sh /workspace/preprocess.sh \
 && chmod +x /workspace/run_pipeline.sh /workspace/preprocess.sh

# ENTRYPOINT so `docker run <img> /data` passes /data as the scene arg.
# For a debug shell:  docker run --entrypoint bash -it <img>
ENTRYPOINT ["bash", "/workspace/run_pipeline.sh"]
