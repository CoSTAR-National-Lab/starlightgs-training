#!/usr/bin/env bash
# GS-IR one-command pipeline: Stage 1 -> bake -> Stage 2 -> preview render -> export PLY.
#
# This is the container ENTRYPOINT, so any user can run the whole pipeline with one command:
#     docker run --gpus all -v <scene>:/data -v <out>:/output ghcr.io/costar/gs-ir /data
# (or, more simply, via the run.sh / run.bat launchers).
#
# Everything is auto-detected/auto-sized; the GSIR_* env vars below are only for
# power users who want to override a default.
set -euo pipefail

cd /workspace
export HOME="${HOME:-/tmp}"   # some tools want a writable HOME (we run as an arbitrary --user)

# ---- inputs -------------------------------
DATA="${1:-${GSIR_DATA:-/data}}"
OUTPUT_BASE="${GSIR_OUTPUT:-/output}"       # Optional
SCENE="${GSIR_SCENE:-$(basename "$DATA")}"  # Optional
OUT="$OUTPUT_BASE/$SCENE"                   # Optional

RES="${GSIR_RESOLUTION:-1}"               # 1 = full resolution; 2 or 4 if VRAM is short
DATA_DEVICE="${GSIR_DATA_DEVICE:-cuda}"   # cpu keeps images in RAM: full quality, slower
ITERS_S1="${GSIR_ITERS_S1:-30000}"
ITERS_S2="${GSIR_ITERS_S2:-35000}"
# Preserve the stock GS-IR ratios by default: densify for the first half of Stage 1,
# and decay the position LR across all of Stage 1. Both remain overrideable.
DENSIFY_UNTIL="${GSIR_DENSIFY_UNTIL:-$(( ITERS_S1 / 2 ))}"
POSITION_LR_MAX_STEPS="${GSIR_POSITION_LR_MAX_STEPS:-$ITERS_S1}"
OCCLU_RES="${GSIR_OCCLU_RES:-256}"
OCCLUSION="${GSIR_OCCLUSION:-0.4}"
EVAL_FLAG=""                              # GSIR_EVAL=1 computes PSNR/SSIM/LPIPS metrics
[ -n "${GSIR_EVAL:-}" ] && EVAL_FLAG="--evaluate"

# Stage 2 --iterations is an absolute final iteration, not an additional count.
if [ "$ITERS_S2" -le "$ITERS_S1" ]; then
    echo "ERROR: GSIR_ITERS_S2 ($ITERS_S2) must be greater than GSIR_ITERS_S1 ($ITERS_S1)." >&2
    echo "       Example: Stage 1 = 150000 and 25000 PBR iterations -> GSIR_ITERS_S2=175000." >&2
    exit 1
fi

# ---- sanity checks ------------------------------------------------------------
if [ ! -d "$DATA" ]; then
    echo "ERROR: data folder '$DATA' not found inside the container." >&2
    echo "       Mount your scene with:  -v /path/to/scene:/data" >&2
    exit 1
fi
mkdir -p "$OUT"

# Keep a full transcript beside the results. The container is removed when it exits, so
# otherwise the only copy of the output is the user's terminal, which is no use when a
# remote tester needs to report a failure.
exec > >(tee -a "$OUT/run.log") 2>&1

echo "=================================================================="
echo " GS-IR run started $(date -u '+%Y-%m-%d %H:%M:%SZ')"
echo "   gpu         : $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | sed -n '1p')"
echo "   torch/cuda  : $(python -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda)' 2>/dev/null)"
echo "   settings    : -r $RES  iters $ITERS_S1/$ITERS_S2  densify_until $DENSIFY_UNTIL  pos_lr_steps $POSITION_LR_MAX_STEPS"
echo "                 occlu_res $OCCLU_RES  occlusion $OCCLUSION"
echo "=================================================================="

# ---- format detection ---------------------------------------------------------
# A calibrated scene (has camera poses) trains directly. Anything else is treated as
# a raw capture and handed to preprocess.sh, which extracts video frames / gathers
# photos and runs COLMAP. preprocess.sh errors clearly if nothing usable.
if [ -d "$DATA/train/pose" ] || [ -d "$DATA/sparse" ] || [ -f "$DATA/transforms_train.json" ]; then
    :  # NeRF-OSR / COLMAP / Blender — ready to train
else
    echo ""
    echo ">>> [0/5] No camera poses found - running COLMAP preprocessing (one-time; saved into your scene folder)"
    bash /workspace/preprocess.sh "$DATA"
fi

echo "=================================================================="
echo " GS-IR pipeline"
echo "   scene       : $SCENE"
echo "   data        : $DATA"
echo "   output      : $OUT"
echo "   resolution  : -r $RES   (set GSIR_RESOLUTION=2 if you run out of GPU memory)"
echo "=================================================================="

CKPT_S1="$OUT/chkpnt${ITERS_S1}.pth"
CKPT_S2="$OUT/chkpnt${ITERS_S2}.pth"

# Every training image is held on the GPU, so a large photo set is the usual cause of
# out-of-memory. Estimate it here rather than failing an hour into training.
if [ "$DATA_DEVICE" = "cuda" ]; then
    IMG_DIR="$DATA/images"
    [ -d "$DATA/train/rgb" ] && IMG_DIR="$DATA/train/rgb"
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sed -n '1p')
    python - "$IMG_DIR" "$RES" "${VRAM_MB:-0}" <<'PY' || true
import glob, sys
from PIL import Image
folder, res, vram_mb = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
files = [f for f in sorted(glob.glob(folder + "/*"))
         if f.lower().endswith((".jpg", ".jpeg", ".png"))]
if files and vram_mb:
    w, h = Image.open(files[0]).size
    if res in (1, 2, 4, 8):
        w, h = w / res, h / res
    elif res == -1 and w > 1600:
        w, h = 1600, h * 1600 / w
    needed, vram = len(files) * w * h * 3 * 4 / 1e9, vram_mb / 1024
    if needed > 0.6 * vram:
        print(f"WARNING: {len(files)} images at {int(w)}x{int(h)} need about {needed:.1f} GB of GPU")
        print(f"         memory and this GPU has {vram:.1f} GB. If training runs out of memory,")
        print(f"         retry with GSIR_DATA_DEVICE=cpu (full quality, slower) or")
        print(f"         GSIR_RESOLUTION=2 (faster, lower quality).")
PY
fi

echo ""; echo ">>> [1/5] Stage 1 - geometry + appearance (${ITERS_S1} iters)"
python train.py -m "$OUT" -s "$DATA" --iterations "$ITERS_S1" --eval -r "$RES" \
    --pbr_iteration "$ITERS_S1" \
    --densify_until_iter "$DENSIFY_UNTIL" \
    --position_lr_max_steps "$POSITION_LR_MAX_STEPS" \
    --data_device "$DATA_DEVICE"

echo ""; echo ">>> [2/5] Baking occlusion volumes (auto-bound)"
python baking.py -m "$OUT" --checkpoint "$CKPT_S1" \
    --occlu_res "$OCCLU_RES" --occlusion "$OCCLUSION"

echo ""; echo ">>> [3/5] Stage 2 - material / lighting decomposition (${ITERS_S2} iters)"
# Reuse the bound the occlusion was baked with, so the indirect-lighting volume covers the
# same cube. train.py otherwise falls back to its own default of 1.5, which is far too small
# for any real scene and collapses indirect light to a near-constant term.
BOUND=$(python -c "import torch; print(torch.load('$OUT/occlusion_volumes.pth', map_location='cpu', weights_only=False)['bound'])")
echo "    using baked bound: $BOUND"
python train.py -m "$OUT" -s "$DATA" \
    --start_checkpoint "$CKPT_S1" \
    --iterations "$ITERS_S2" --eval --gamma --indirect -r "$RES" \
    --pbr_iteration "$ITERS_S1" \
    --densify_until_iter "$DENSIFY_UNTIL" \
    --position_lr_max_steps "$POSITION_LR_MAX_STEPS" \
    --bound "$BOUND" --data_device "$DATA_DEVICE"

echo ""; echo ">>> [4/5] Preview renders (PBR + normals + material channels)"
python render.py -m "$OUT" -s "$DATA" \
    --checkpoint "$CKPT_S2" -r "$RES" --pbr --gamma --indirect \
    --data_device "$DATA_DEVICE" --skip_train $EVAL_FLAG

echo ""; echo ">>> [5/5] Exporting material PLY"
python extract_ply.py --checkpoint-file "$CKPT_S2" \
    --output-path "$OUT" --output-file "${SCENE}.ply"

echo ""
echo "=================================================================="
echo " DONE - $SCENE"
echo "   PLY            : $OUT/${SCENE}.ply"
echo "   Preview renders: $OUT/test/  (pbr/, normal/, renders/, gt/)"
echo "   Checkpoints    : $CKPT_S1 , $CKPT_S2"
echo "   Log            : $OUT/run.log  (send this if you hit a problem)"
echo "=================================================================="
