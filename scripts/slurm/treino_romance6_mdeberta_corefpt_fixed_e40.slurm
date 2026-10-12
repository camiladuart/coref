#!/bin/bash
#SBATCH -A eehpc-dev-2026d06-183g
#SBATCH -p normal-a100-40
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -J romance6_mdeb_fixed_e40
#SBATCH -o /projects/EEHPC-DEV-2026D06-183/up202000683/logs/%x-%j.out
#SBATCH -e /projects/EEHPC-DEV-2026D06-183/up202000683/logs/%x-%j.err

set -euo pipefail

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load Transformers/4.39.3-gfbf-2023a

BASE=/projects/EEHPC-DEV-2026D06-183/up202000683
REPO=$BASE/repos/coref

DATA_DIR=$BASE/coref_data/romance6_mdeberta_corefpt_fixed
SAVE_DIR=$BASE/checkpoints_romance6_mdeberta_corefpt_fixed_e40

ENCODER=microsoft/mdeberta-v3-base
EPOCHS=40

export TOP_SPAN_RATIO=0.03
export MAX_TOP_SPANS=80
export MAX_TOP_ANTECEDENTS=10
export ENCODER_BATCH_SIZE=1
export GRADIENT_CHECKPOINTING=1

export HF_HOME=$BASE/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

source "$BASE/venvs/coref_train/bin/activate"

cd "$REPO"

echo "===== MULTILINGUAL MDEBERTA RETRAINING ====="
echo "DATE: $(date)"
echo "HOST: $(hostname)"
echo "DATASET=$DATA_DIR"
echo "SAVE_DIR=$SAVE_DIR"
echo "ENCODER=$ENCODER"
echo "EPOCHS=$EPOCHS"

echo "===== MEMORY CONFIGURATION ====="
echo "TOP_SPAN_RATIO=$TOP_SPAN_RATIO"
echo "MAX_TOP_SPANS=$MAX_TOP_SPANS"
echo "MAX_TOP_ANTECEDENTS=$MAX_TOP_ANTECEDENTS"
echo "ENCODER_BATCH_SIZE=$ENCODER_BATCH_SIZE"
echo "GRADIENT_CHECKPOINTING=$GRADIENT_CHECKPOINTING"

echo "===== GIT ====="
git rev-parse HEAD
git status -sb

echo "===== DATASET ====="
wc -l \
  "$DATA_DIR/train.jsonlines" \
  "$DATA_DIR/dev.jsonlines" \
  "$DATA_DIR/test.jsonlines"

echo "===== PYTHON AND CUDA ====="
python3 - <<'PY'
import torch
import transformers
from transformers import AutoTokenizer

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")

print("GPU:", torch.cuda.get_device_name(0))

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/mdeberta-v3-base",
    local_files_only=True
)
print("Tokenizer:", tokenizer.__class__.__name__)
PY

if [ -e "$SAVE_DIR" ]; then
    echo "ERROR: checkpoint directory already exists: $SAVE_DIR"
    exit 1
fi

mkdir "$SAVE_DIR"

echo "===== START TRAINING ====="
date

python3 -u -m coref_loader.training.main_trainer \
  --data_dir "$DATA_DIR" \
  --split train \
  --epochs "$EPOCHS" \
  --temperature_switch_epoch 10 \
  --max_seq_length 384 \
  --max_span_width 10 \
  --bert_lr 2e-5 \
  --task_lr 1e-4 \
  --warmup_ratio 0.1 \
  --eval_every 1 \
  --save_dir "$SAVE_DIR" \
  --encoder_name "$ENCODER"

echo "===== TRAINING FINISHED ====="
date

echo "===== CHECKPOINTS ====="
ls -lh "$SAVE_DIR"
