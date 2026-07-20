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

RES="${GSIR_RESOLUTION:-1}"          # NeRF-OSR REQUIRES -r 1; safe default elsewhere
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
python train.py -m "$OUT" -s "$DATA" --iterations "$ITERS_S1" --eval -r "$RES"

echo ""; echo ">>> [2/5] Baking occlusion volumes (auto-bound)"
python baking.py -m "$OUT" --checkpoint "$CKPT_S1" \
    --occlu_res "$OCCLU_RES" --occlusion "$OCCLUSION"

echo ""; echo ">>> [3/5] Stage 2 - material / lighting decomposition (${ITERS_S2} iters)"
python train.py -m "$OUT" -s "$DATA" \
    --start_checkpoint "$CKPT_S1" \
    --iterations "$ITERS_S2" --eval --gamma --indirect -r "$RES"

echo ""; echo ">>> [4/5] Preview renders (PBR + normals + material channels)"
python render.py -m "$OUT" -s "$DATA" \
    --checkpoint "$CKPT_S2" -r "$RES" --pbr --gamma --indirect

echo ""; echo ">>> [5/5] Exporting material PLY"
python extract_ply.py --checkpoint-file "$CKPT_S2" \
    --output-path "$OUT" --output-file "${SCENE}.ply"

echo ""
echo "=================================================================="
echo " DONE - $SCENE"
echo "   PLY            : $OUT/${SCENE}.ply"
echo "   Preview renders: $OUT/test/  (pbr/, normal/, renders/, gt/)"
echo "   Checkpoints    : $CKPT_S1 , $CKPT_S2"
echo "=================================================================="
