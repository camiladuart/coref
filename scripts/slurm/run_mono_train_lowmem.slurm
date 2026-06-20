#!/bin/bash
#SBATCH -A f202600026aivlabdeucaliong
#SBATCH -p normal-a100-40
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -o /projects/F202600026AIVLABDEUCALION/up202000683/logs/%x-%j.out
#SBATCH -e /projects/F202600026AIVLABDEUCALION/up202000683/logs/%x-%j.err

set -euo pipefail

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load Transformers/4.39.3-gfbf-2023a

BASE=/projects/F202600026AIVLABDEUCALION/up202000683
REPO=$BASE/repos/coref

: "${DATASET:?Missing DATASET}"
: "${TAG:?Missing TAG}"
: "${ENCODER:?Missing ENCODER}"
: "${EPOCHS:?Missing EPOCHS}"

MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-384}"
TOP_SPAN_RATIO="${TOP_SPAN_RATIO:-0.03}"
MAX_SPAN_WIDTH="${MAX_SPAN_WIDTH:-10}"
MAX_TOP_SPANS="${MAX_TOP_SPANS:-80}"
MAX_TOP_ANTECEDENTS="${MAX_TOP_ANTECEDENTS:-10}"

DATA_DIR=$BASE/coref_data/$DATASET
SAVE_DIR=$BASE/checkpoints_mono_${TAG}_e${EPOCHS}

mkdir -p "$BASE/logs" "$SAVE_DIR"

export HF_HOME=$BASE/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TOP_SPAN_RATIO MAX_SPAN_WIDTH MAX_TOP_SPANS MAX_TOP_ANTECEDENTS

source "$BASE/venvs/coref_train/bin/activate"

cd "$REPO"

echo "===== TRAIN LOWMEM ====="
echo "DATASET=$DATASET"
echo "TAG=$TAG"
echo "ENCODER=$ENCODER"
echo "EPOCHS=$EPOCHS"
echo "MAX_SEQ_LENGTH=$MAX_SEQ_LENGTH"
echo "TOP_SPAN_RATIO=$TOP_SPAN_RATIO"
echo "MAX_SPAN_WIDTH=$MAX_SPAN_WIDTH"
echo "MAX_TOP_SPANS=$MAX_TOP_SPANS"
echo "MAX_TOP_ANTECEDENTS=$MAX_TOP_ANTECEDENTS"
echo "DATA_DIR=$DATA_DIR"
echo "SAVE_DIR=$SAVE_DIR"
date

echo "===== CHECK PATHS ====="
ls -lh "$DATA_DIR/train.jsonlines" "$DATA_DIR/dev.jsonlines" "$DATA_DIR/test.jsonlines"

python3 - <<'PY'
import torch, transformers, scipy, sentencepiece, sacremoses
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("scipy:", scipy.__version__)
print("sentencepiece OK")
print("sacremoses OK")
PY

python3 - <<'PY'
import os
from transformers import AutoTokenizer, AutoModel

model = os.environ["ENCODER"]
print("===== CHECK ENCODER CACHE =====")
print("ENCODER:", model)
AutoTokenizer.from_pretrained(model, local_files_only=True)
AutoModel.from_pretrained(model, local_files_only=True)
print("[OK] encoder available offline")
PY

HELP=$(python3 -m coref_loader.training.main_trainer --help 2>&1 || true)

EXTRA_ARGS=()

if echo "$HELP" | grep -q -- "--top_span_ratio"; then
  EXTRA_ARGS+=(--top_span_ratio "$TOP_SPAN_RATIO")
  echo "[OK] using --top_span_ratio $TOP_SPAN_RATIO"
else
  echo "[WARN] --top_span_ratio not found in trainer help"
fi

if echo "$HELP" | grep -q -- "--max_span_width"; then
  EXTRA_ARGS+=(--max_span_width "$MAX_SPAN_WIDTH")
  echo "[OK] using --max_span_width $MAX_SPAN_WIDTH"
else
  echo "[WARN] --max_span_width not found in trainer help"
fi

python3 -m coref_loader.training.main_trainer \
  --data_dir "$DATA_DIR" \
  --split train \
  --epochs "$EPOCHS" \
  --temperature_switch_epoch 10 \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --bert_lr 2e-5 \
  --task_lr 1e-4 \
  --eval_every 1 \
  --save_dir "$SAVE_DIR" \
  --encoder_name "$ENCODER" \
  "${EXTRA_ARGS[@]}"

echo "===== DONE TRAIN LOWMEM ====="
date
ls -lh "$SAVE_DIR" | tail -30
