#!/bin/bash
#SBATCH -A eehpc-dev-2026d06-183g
#SBATCH -p normal-a100-40
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -J corefpt_fixed_bert_e40
#SBATCH -o /projects/EEHPC-DEV-2026D06-183/up202000683/logs/%x-%j.out
#SBATCH -e /projects/EEHPC-DEV-2026D06-183/up202000683/logs/%x-%j.err

set -euo pipefail

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load Transformers/4.39.3-gfbf-2023a

BASE=/projects/EEHPC-DEV-2026D06-183/up202000683
REPO=$BASE/repos/coref

DATA_DIR=$BASE/coref_data/coref_pt_sentseg_fixed
SAVE_DIR=$BASE/checkpoints_corefpt_bert_pt_sentseg_fixed_e40

export HF_HOME=$BASE/hf_cache
export HF_HUB_CACHE=$HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets

mkdir -p "$BASE/logs"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
mkdir -p "$SAVE_DIR"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

# Impede sobrescritas acidentais dos parâmetros de pruning.
unset TOP_SPAN_RATIO
unset MAX_TOP_SPANS
unset MAX_TOP_ANTECEDENTS
unset MAX_SPAN_WIDTH

cd "$REPO"

echo "============================================================"
echo "COREF-PT FIXED — BERTIMBAU — 40 EPOCHS"
echo "============================================================"
echo "HOST: $(hostname)"
echo "DATE: $(date)"
echo "REPO: $REPO"
echo "DATA: $DATA_DIR"
echo "SAVE: $SAVE_DIR"
echo

echo "=== GIT ==="
git rev-parse HEAD
git status -sb

echo
echo "=== DATA ==="
wc -l \
  "$DATA_DIR/train.jsonlines" \
  "$DATA_DIR/dev.jsonlines" \
  "$DATA_DIR/test.jsonlines"

echo
echo "=== PYTHON / CUDA ==="
which python3
python3 -V

python3 - <<'PY'
import torch
import transformers

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("ERRO: CUDA não disponível")

print("GPU:", torch.cuda.get_device_name(0))
PY

echo
echo "=== BERTIMBAU OFFLINE CHECK ==="

python3 - <<'PY'
from transformers import AutoTokenizer

name = "neuralmind/bert-base-portuguese-cased"

tok = AutoTokenizer.from_pretrained(
    name,
    local_files_only=True
)

print("Tokenizer:", tok.__class__.__name__)
print("Vocab:", tok.vocab_size)
PY

echo
echo "=== EFFECTIVE TRAINING RECIPE ==="
echo "encoder=neuralmind/bert-base-portuguese-cased"
echo "epochs=40"
echo "max_seq_length=512"
echo "max_span_width=20"
echo "max_segment_len=128"
echo "top_span_ratio=0.1 [trainer config]"
echo "max_top_spans=3900 [model fallback]"
echo "max_top_antecedents=50 [trainer config]"
echo "dropout_rate=0.3 [trainer config]"
echo "bert_lr=2e-5"
echo "task_lr=1e-4"
echo "warmup_ratio=0.1"
echo "eval_every=1"

echo
echo "=== START TRAINING ==="
date

python3 -u -m coref_loader.training.main_trainer \
  --data_dir "$DATA_DIR" \
  --split train \
  --epochs 40 \
  --max_seq_length 512 \
  --max_span_width 20 \
  --max_segment_len 128 \
  --bert_lr 2e-5 \
  --task_lr 1e-4 \
  --warmup_ratio 0.1 \
  --eval_every 1 \
  --save_dir "$SAVE_DIR" \
  --encoder_name neuralmind/bert-base-portuguese-cased

echo
echo "=== TRAINING FINISHED ==="
date

echo
echo "=== CHECKPOINTS ==="
ls -lh "$SAVE_DIR"
