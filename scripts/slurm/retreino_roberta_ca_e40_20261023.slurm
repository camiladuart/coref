#!/bin/bash
#SBATCH --job-name=roberta_ca_80
#SBATCH --account=eehpc-dev-2026d06-183g
#SBATCH --partition=normal-a100-80
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/projects/EEHPC-DEV-2026D06-183/up202000683/logs/roberta_ca_80-%j.out
#SBATCH --error=/projects/EEHPC-DEV-2026D06-183/up202000683/logs/roberta_ca_80-%j.err

set -euo pipefail

BASE=/projects/EEHPC-DEV-2026D06-183/up202000683
REPO="$BASE/repos/coref"
DATA="$BASE/coref_data/catalan_ancora"
SAVE="$BASE/checkpoints_mono_roberta_ca_catalan_ancora_retrain_a10080_20261023_e40"

CATALAN="$BASE/hf_cache/hub/models--projecte-aina--roberta-base-ca-v2/snapshots/772dfdf344529cf40009372bf14915c4a8aa1152"

echo "===== PRE-FLIGHT ====="
date

# Nunca utilizar um diretorio de checkpoints existente
if [ -e "$SAVE" ]; then
    echo "ERRO: destino ja existe: $SAVE"
    exit 1
fi

# Confirmar ficheiros de entrada e encoder
for FILE in \
    "$DATA/train.jsonlines" \
    "$DATA/dev.jsonlines" \
    "$DATA/test.jsonlines" \
    "$CATALAN/config.json" \
    "$CATALAN/pytorch_model.bin"
do
    if [ ! -f "$FILE" ]; then
        echo "ERRO: ficheiro ausente: $FILE"
        exit 1
    fi
done

# Exigir 150 GiB livres antes do inicio
FREE_KB=$(df -Pk "$BASE" | awk 'NR==2 {print $4}')
MIN_KB=$((150 * 1024 * 1024))

if [ "$FREE_KB" -lt "$MIN_KB" ]; then
    echo "ERRO: espaco livre inferior a 150 GiB"
    exit 1
fi

df -h "$BASE"

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load Transformers/4.39.3-gfbf-2023a

source "$BASE/venvs/coref_train/bin/activate"

export HF_HOME="$BASE/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

export TOP_SPAN_RATIO=0.4
export MAX_SPAN_WIDTH=20
export MAX_TOP_SPANS=3900
export MAX_TOP_ANTECEDENTS=50
export ENCODER_BATCH_SIZE=1
export GRADIENT_CHECKPOINTING=1

cd "$REPO"

python3 - <<'PY'
import torch
import transformers

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("GPU CUDA indisponivel")

print("GPU:", torch.cuda.get_device_name(0))
print("Memoria GPU:", torch.cuda.get_device_properties(0).total_memory)
PY

# Criacao exclusiva: falha se outro processo criou o destino
mkdir "$SAVE"

echo "===== CONFIGURACAO ====="
echo "DATA=$DATA"
echo "SAVE=$SAVE"
echo "ENCODER=$CATALAN"
echo "EPOCHS=40"
echo "MAX_SEQ_LENGTH=384"
echo "TOP_SPAN_RATIO=$TOP_SPAN_RATIO"
echo "MAX_SPAN_WIDTH=$MAX_SPAN_WIDTH"
echo "MAX_TOP_SPANS=$MAX_TOP_SPANS"
echo "MAX_TOP_ANTECEDENTS=$MAX_TOP_ANTECEDENTS"
echo "GRADIENT_CHECKPOINTING=$GRADIENT_CHECKPOINTING"

echo "===== TRAIN ====="

python3 -m coref_loader.training.main_trainer \
    --data_dir "$DATA" \
    --split train \
    --epochs 40 \
    --max_seq_length 384 \
    --max_span_width 20 \
    --bert_lr 2e-5 \
    --task_lr 1e-4 \
    --warmup_ratio 0.1 \
    --eval_every 1 \
    --temperature_switch_epoch 10 \
    --encoder_name "$CATALAN" \
    --save_dir "$SAVE"

echo "===== TREINO TERMINADO ====="
date
df -h "$BASE"
