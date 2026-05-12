#!/bin/bash
# run_eval.sh - Simple script to launch model evaluation

set -e  # Exit on error

# Default configuration
KING_REPO="/dev/shm/model"
CHALLENGER_REPO="/dev/shm/teutonic/models/ClarenceDan/Teutonic-LXXX-5HBfvtZo-A5514"
# DATASET_DIR="/dev/shm/teutonic/datasets_eval"
JSON_FILE="/dev/shm/teutonic/datasets/sample_2.jsonl"
JSON_SAMPLE_NUMBER=800
N_SAMPLES=80
SEQ_LEN=2048
BATCH_SIZE=8
ALPHA=0.001
DELTA=0.0025
N_BOOTSTRAP=10000
GPUS="auto"
SEED="eval:102"

echo "🚀 Starting evaluation..."
echo "  King         : $KING_REPO"
echo "  Challenger   : $CHALLENGER_REPO"
echo "  Dataset      : $DATASET_DIR"
echo "  Json File    : $JSON_FILE"
echo "  Json Sample Number : $JSON_SAMPLE_NUMBER"
echo "  Samples      : $N_SAMPLES"
echo "  Seq len      : $SEQ_LEN"
echo "  Batch size   : $BATCH_SIZE"
echo "  Delta/Alpha  : $DELTA / $ALPHA"
echo "  GPUs         : $GPUS"
echo "  Seed         : $SEED"
echo ""

python3 eval.py \
    --king "$KING_REPO" \
    --challenger "$CHALLENGER_REPO" \
    --jsonl-file "$JSON_FILE" \
    --jsonl-sample-number "$JSON_SAMPLE_NUMBER" \
    --n "$N_SAMPLES" \
    --seq-len "$SEQ_LEN" \
    --batch-size "$BATCH_SIZE" \
    --alpha "$ALPHA" \
    --delta "$DELTA" \
    --n-bootstrap "$N_BOOTSTRAP" \
    --gpus "$GPUS" \
    --seed "$SEED"

echo "✅ Done! Verdict printed above."