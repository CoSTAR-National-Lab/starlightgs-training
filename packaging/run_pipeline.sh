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
OCCLU_RES="${GSIR_OCCLU_RES:-256}"
OCCLUSION="${GSIR_OCCLUSION:-0.4}"

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
echo "   gpu         : $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "   torch/cuda  : $(python -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda)' 2>/dev/null)"
echo "   settings    : -r $RES  iters $ITERS_S1/$ITERS_S2  occlu_res $OCCLU_RES  occlusion $OCCLUSION"
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

echo ""; echo ">>> [1/5] Stage 1 - geometry + appearance (${ITERS_S1} iters)"
python train.py -m "$OUT" -s "$DATA" --iterations "$ITERS_S1" --eval -r "$RES" \
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
    --bound "$BOUND" --data_device "$DATA_DEVICE"

echo ""; echo ">>> [4/5] Preview renders (PBR + normals + material channels)"
python render.py -m "$OUT" -s "$DATA" \
    --checkpoint "$CKPT_S2" -r "$RES" --pbr --gamma --indirect \
    --data_device "$DATA_DEVICE"

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
