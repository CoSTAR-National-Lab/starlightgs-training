#!/usr/bin/env bash
# Recover camera poses from raw images or a video using COLMAP
# and writes a ready dataset back into the scene folder:
#
#   <scene>/input/     raw frames/photos (preserved, untouched)
#   <scene>/images/    undistorted images GS-IR actually trains on
#   <scene>/sparse/0/  COLMAP model (poses + intrinsics + sparse points)
#
# Run once; run_pipeline.sh auto-detects sparse/ on re-runs and skips this slow step.
#
#   bash preprocess.sh <scene_folder>
#
# Overrides (env):
#   GSIR_TARGET_FRAMES  target frame count when extracting from video (default 300)
#   GSIR_MATCHER        auto | exhaustive | sequential   (default auto)
#   GSIR_CAMERA         COLMAP camera model               (default OPENCV)
#   GSIR_COLMAP_GPU     1 to try GPU SIFT (needs a GL context; usually leave 0)
#   GSIR_COLMAP_MAX_IMAGE_SIZE  downscale cap for feature detection (default 2000;
#                       lower it if COLMAP gets "Killed" / OOMs on a low-RAM machine)
set -euo pipefail
export QT_QPA_PLATFORM=offscreen        # COLMAP runs headless in the container

DATA="$1"
TARGET_FRAMES="${GSIR_TARGET_FRAMES:-300}"
MATCHER="${GSIR_MATCHER:-auto}"
CAMERA="${GSIR_CAMERA:-OPENCV}"
USE_GPU="${GSIR_COLMAP_GPU:-0}"

# ---- 1. assemble input/ (raw frames live here; originals are never altered) ----
if [ ! -d "$DATA/input" ]; then
    mkdir -p "$DATA/input"
    shopt -s nullglob nocaseglob
    videos=( "$DATA"/*.mp4 "$DATA"/*.mov "$DATA"/*.avi )
    photos=( "$DATA"/*.jpg "$DATA"/*.jpeg "$DATA"/*.png )
    shopt -u nocaseglob

    if [ ${#videos[@]} -gt 0 ]; then
        # frames-per-second chosen so total frames ~= TARGET_FRAMES, regardless of length
        dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "${videos[0]}")
        r=$(python -c "import math; print(max(1, math.ceil($TARGET_FRAMES / max(1e-6,$dur))))")
        echo "    video ${dur}s -> extracting ~${TARGET_FRAMES} frames at ${r} fps"
        ffmpeg -hide_banner -loglevel error -i "${videos[0]}" \
            -vf "fps=${r}" -qscale:v 1 "$DATA/input/frame_%05d.jpg"
        [ "$MATCHER" = "auto" ] && MATCHER=sequential      # ordered frames
    elif [ ${#photos[@]} -gt 0 ]; then
        echo "    moving loose photos into input/"
        for f in "${photos[@]}"; do mv "$f" "$DATA/input/"; done
    elif [ -d "$DATA/images" ]; then
        echo "    using existing images/ as raw input"
        mv "$DATA/images" "$DATA/input"
    fi
    shopt -u nullglob
fi
[ "$MATCHER" = "auto" ] && MATCHER=exhaustive              # unordered photos

nimg=$(find "$DATA/input" -maxdepth 1 -type f | wc -l)
echo "    ${nimg} input images; matcher=${MATCHER}_matcher; camera=${CAMERA}; use_gpu=${USE_GPU}"
if [ "$nimg" -eq 0 ]; then
    echo "ERROR: no images or video found in '$DATA'." >&2
    echo "       Provide loose photos, an input/ folder of photos, or a video file." >&2
    exit 1
fi
if [ "$nimg" -lt 8 ]; then
    echo "ERROR: only ${nimg} images - COLMAP needs ~8+ overlapping views to reconstruct." >&2
    exit 1
fi

# ---- 2. COLMAP structure-from-motion (into distorted/) ------------------------
mkdir -p "$DATA/distorted"
DB="$DATA/distorted/database.db"

echo "    [colmap] feature extraction"
# max_image_size downscales images FOR FEATURE DETECTION ONLY (training still uses the
# full-res undistorted images). Without it, CPU SIFT holds a full image per thread (one
# per core) and can OOM on high-res photos. Lower GSIR_COLMAP_MAX_IMAGE_SIZE if it still
# gets "Killed" on a low-RAM machine.
colmap feature_extractor \
    --database_path "$DB" --image_path "$DATA/input" \
    --ImageReader.single_camera 1 --ImageReader.camera_model "$CAMERA" \
    --SiftExtraction.max_image_size "${GSIR_COLMAP_MAX_IMAGE_SIZE:-2000}" \
    --SiftExtraction.use_gpu "$USE_GPU"

echo "    [colmap] ${MATCHER} matching"
colmap "${MATCHER}_matcher" --database_path "$DB" --SiftMatching.use_gpu "$USE_GPU"

echo "    [colmap] mapping (structure from motion)"

mkdir -p "$DATA/distorted/sparse"
# Tighter BA tolerance than the default speeds up bundle adjustment.
colmap mapper --database_path "$DB" --image_path "$DATA/input" \
    --output_path "$DATA/distorted/sparse" \
    --Mapper.ba_global_function_tolerance=0.000001

if [ ! -f "$DATA/distorted/sparse/0/cameras.bin" ]; then
    echo "ERROR: COLMAP could not reconstruct this scene (too few matches / poor overlap)." >&2
    echo "       Try more photos with more overlap, or a higher GSIR_TARGET_FRAMES for video." >&2
    exit 1
fi

# ---- 3. undistort into ideal pinhole -> images/ + sparse/0/ -------------------
# GS-IR (like 3DGS) expects undistorted PINHOLE images.
echo "    [colmap] undistorting images"
colmap image_undistorter \
    --image_path "$DATA/input" --input_path "$DATA/distorted/sparse/0" \
    --output_path "$DATA" --output_type COLMAP

# image_undistorter writes sparse/ flat; GS-IR wants it under sparse/0/.
mkdir -p "$DATA/sparse/0"
for f in "$DATA"/sparse/*; do
    [ "$(basename "$f")" = "0" ] && continue
    mv "$f" "$DATA/sparse/0/"
done

echo "    COLMAP done -> $DATA/images/ + $DATA/sparse/0/"
