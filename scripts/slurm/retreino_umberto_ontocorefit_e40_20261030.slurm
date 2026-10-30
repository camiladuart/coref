#!/bin/bash
#SBATCH --job-name=umberto_it80
#SBATCH --account=eehpc-dev-2026d06-183g
#SBATCH --partition=normal-a100-80
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/projects/EEHPC-DEV-2026D06-183/up202000683/logs/umberto_it80-%j.out
#SBATCH --error=/projects/EEHPC-DEV-2026D06-183/up202000683/logs/umberto_it80-%j.err

set -euo pipefail

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load Transformers/4.39.3-gfbf-2023a

BASE=/projects/EEHPC-DEV-2026D06-183/up202000683
REPO="$BASE/repos/coref"
DATA="$BASE/coref_data/OntoCorefIT"
SAVE="$BASE/checkpoints_mono_umberto_ontocorefit_retrain_a10080_20261030_e40"

ENCODER="$BASE/hf_cache/hub/models--Musixmatch--umberto-commoncrawl-cased-v1/snapshots/fe7b14808cccbbe2984b05e6fbfd71127f11008f"

# Configuracao normal: verificar contra o registo italiano original
export TOP_SPAN_RATIO=0.4
export MAX_SPAN_WIDTH=20
export MAX_TOP_SPANS=3900
export MAX_TOP_ANTECEDENTS=50
export ENCODER_BATCH_SIZE=1
export GRADIENT_CHECKPOINTING=1

export HF_HOME="$BASE/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

source "$BASE/venvs/coref_train/bin/activate"

echo "===== PRE-FLIGHT ====="

test ! -e "$SAVE" || { echo "ERRO: destino ja existe"; exit 1; }

for FILE in \
    "$DATA/train.jsonlines" \
    "$DATA/dev.jsonlines" \
    "$DATA/test.jsonlines" \
    "$ENCODER/config.json" \
    "$ENCODER/pytorch_model.bin"
do
    test -s "$FILE" || { echo "ERRO: ficheiro ausente: $FILE"; exit 1; }
done

FREE_KB=$(df -Pk "$BASE" | awk 'NR==2 {print $4}')
test "$FREE_KB" -ge $((150 * 1024 * 1024)) || {
    echo "ERRO: espaco insuficiente"
    exit 1
}

cd "$REPO"

python3 - <<'PY'
import torch
from transformers import AutoConfig
print("CUDA:", torch.cuda.is_available())
assert torch.cuda.is_available()
print("GPU:", torch.cuda.get_device_name(0))
PY

mkdir "$SAVE"

echo "===== CONFIGURACAO ITALIANA ====="
echo "DATA=$DATA"
echo "ENCODER=$ENCODER"
echo "SAVE=$SAVE"
echo "EPOCHS=40"
echo "MAX_SEQ_LENGTH=384"
echo "TOP_SPAN_RATIO=$TOP_SPAN_RATIO"
echo "MAX_SPAN_WIDTH=$MAX_SPAN_WIDTH"
echo "MAX_TOP_SPANS=$MAX_TOP_SPANS"
echo "MAX_TOP_ANTECEDENTS=$MAX_TOP_ANTECEDENTS"

python3 -m coref_loader.training.main_trainer \
    --data_dir "$DATA" \
    --split train \
    --epochs 40 \
    --max_seq_length 384 \
    --max_span_width "$MAX_SPAN_WIDTH" \
    --bert_lr 2e-5 \
    --task_lr 1e-4 \
    --warmup_ratio 0.1 \
    --eval_every 1 \
    --temperature_switch_epoch 10 \
    --encoder_name "$ENCODER" \
    --save_dir "$SAVE"

echo "===== TREINO ITALIANO CONCLUIDO ====="
date
df -h "$BASE"
