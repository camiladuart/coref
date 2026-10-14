#!/bin/bash
#SBATCH -A eehpc-dev-2026d06-183g
#SBATCH -p normal-a100-40
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -J romance6_thesis_fixed
#SBATCH -o /projects/EEHPC-DEV-2026D06-183/up202000683/logs/%x-%j.out
#SBATCH -e /projects/EEHPC-DEV-2026D06-183/up202000683/logs/%x-%j.err

set -euo pipefail

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load Transformers/4.39.3-gfbf-2023a

BASE=/projects/EEHPC-DEV-2026D06-183/up202000683
REPO=$BASE/repos/coref

DATA_DIR=$BASE/coref_data/romance6_mdeberta_corefpt_fixed
SAVE_DIR=$BASE/checkpoints_romance6_mdeberta_corefpt_fixed_thesisparams_e40

ENCODER=microsoft/mdeberta-v3-base
EPOCHS=40

# Effective parameters documented in thesis Table 5.1
export TOP_SPAN_RATIO=0.005
export MAX_SPAN_WIDTH=6
export MAX_TOP_SPANS=30
export MAX_TOP_ANTECEDENTS=4
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

echo "===== THESIS-CONFIGURATION MULTILINGUAL RETRAIN ====="
date
hostname

echo "===== GIT VERSION ====="
git rev-parse HEAD
git status --short

echo "===== DATASET ====="
echo "$DATA_DIR"
wc -l "$DATA_DIR"/{train,dev,test}.jsonlines

echo "===== PARAMETERS ====="
echo "ENCODER=$ENCODER"
echo "EPOCHS=$EPOCHS"
echo "MAX_SEQ_LENGTH=128"
echo "MAX_SEGMENT_LEN=128"
echo "TOP_SPAN_RATIO=$TOP_SPAN_RATIO"
echo "MAX_SPAN_WIDTH=$MAX_SPAN_WIDTH"
echo "MAX_TOP_SPANS=$MAX_TOP_SPANS"
echo "MAX_TOP_ANTECEDENTS=$MAX_TOP_ANTECEDENTS"
echo "ENCODER_BATCH_SIZE=$ENCODER_BATCH_SIZE"
echo "GRADIENT_CHECKPOINTING=$GRADIENT_CHECKPOINTING"
echo "BERT_LR=2e-5"
echo "TASK_LR=1e-4"
echo "WARMUP_RATIO=0.1"
echo "EVAL_EVERY=1"
echo "TEMPERATURE_SAMPLING=OFF (combined --data_dir)"

echo "===== HARDWARE ====="
python3 - <<'PY'
import torch, transformers
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("GPU unavailable")
print("GPU:", torch.cuda.get_device_name(0))
PY

echo "===== CHECK DATA ====="
test -s "$DATA_DIR/train.jsonlines"
test -s "$DATA_DIR/dev.jsonlines"
test -s "$DATA_DIR/test.jsonlines"

test "$(wc -l < "$DATA_DIR/train.jsonlines")" -eq 5120
test "$(wc -l < "$DATA_DIR/dev.jsonlines")" -eq 636
test "$(wc -l < "$DATA_DIR/test.jsonlines")" -eq 649

echo "===== PROTECT CHECKPOINTS ====="
if [ -e "$SAVE_DIR" ]; then
    echo "ERROR: output directory already exists"
    exit 1
fi

mkdir "$SAVE_DIR"

echo "===== START ====="
date

python3 -u -m coref_loader.training.main_trainer \
    --data_dir "$DATA_DIR" \
    --split train \
    --epochs "$EPOCHS" \
    --temperature_switch_epoch 10 \
    --max_seq_length 128 \
    --max_segment_len 128 \
    --max_span_width 6 \
    --bert_lr 2e-5 \
    --task_lr 1e-4 \
    --warmup_ratio 0.1 \
    --eval_every 1 \
    --save_dir "$SAVE_DIR" \
    --encoder_name "$ENCODER"

echo "===== TRAINING COMPLETED ====="
date
ls -lh "$SAVE_DIR"
