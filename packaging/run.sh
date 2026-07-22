#!/usr/bin/env bash
# Friendly launcher for the GS-IR one-command pipeline (Linux / macOS-shell / WSL2).
#
#   ./run.sh <scene_folder> <output_folder>
#
# It fills in the Docker flags (GPU access, volume mounts, file ownership) so you
# don't have to remember them. Override the image with:  GSIR_IMAGE=... ./run.sh ...
set -euo pipefail

IMAGE="${GSIR_IMAGE:-ghcr.io/costar/gs-ir:latest}"

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <scene_folder> <output_folder>"
    echo
    echo "  <scene_folder>   folder with your captured scene. One of:"
    echo "                     NeRF-OSR : has a  train/pose/  folder"
    echo "                     COLMAP   : has a  sparse/      folder"
    echo "                     Blender  : has a  transforms_train.json  file"
    echo "  <output_folder>  where the PLY, preview renders and checkpoints go"
    exit 1
fi

DATA="$(cd "$1" && pwd)"                 # Docker needs absolute host paths
mkdir -p "$2"; OUT="$(cd "$2" && pwd)"
SCENE="$(basename "$DATA")"              # name the output after YOUR folder, not the /data mount

echo "Scene : $DATA"
echo "Output: $OUT"
echo "Image : $IMAGE"
echo

# --gpus all            : hand the GPU(s) to the container
# --user $(id -u):$(id -g): write output files as YOU, not root
# --shm-size            : PyTorch dataloader shared memory
docker run --rm --gpus all \
    --user "$(id -u):$(id -g)" \
    --shm-size=16g \
    -e GSIR_SCENE="$SCENE" \
    -e GSIR_RESOLUTION -e GSIR_ITERS_S1 -e GSIR_ITERS_S2 \
    -e GSIR_OCCLU_RES -e GSIR_OCCLUSION \
    -e GSIR_TARGET_FRAMES -e GSIR_MATCHER -e GSIR_CAMERA \
    -e GSIR_COLMAP_MAX_IMAGE_SIZE -e GSIR_COLMAP_GPU \
    -v "$DATA":/data \
    -v "$OUT":/output \
    "$IMAGE" /data
