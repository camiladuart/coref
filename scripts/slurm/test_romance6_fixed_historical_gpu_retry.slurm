#!/bin/bash
#SBATCH -A eehpc-dev-2026d06-183g
#SBATCH -p normal-a100-40
#SBATCH --gres=gpu:1
#SBATCH --array=0-5%4
#SBATCH -t 04:00:00
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -J test_romance6_histgpu
#SBATCH -o /projects/EEHPC-DEV-2026D06-183/up202000683/logs/%x-%A_%a.out
#SBATCH -e /projects/EEHPC-DEV-2026D06-183/up202000683/logs/%x-%A_%a.err

set -euo pipefail

BASE=/projects/EEHPC-DEV-2026D06-183/up202000683
REPO="$BASE/repos/coref"
CKPT_DIR="$BASE/checkpoints_romance6_mdeberta_corefpt_fixed_thesisparams_e40"
SCORER="$REPO/corefud-scorer/corefud-scorer.py"

DATASETS=(
    coref_pt_sentseg_fixed
    spanish_ancora
    catalan_ancora
    french_ancor
    french_democrat
    OntoCorefIT
)

TAGS=(
    romance6_pt
    romance6_es
    romance6_cat
    romance6_fr_ancor
    romance6_fr_democrat
    romance6_it
)

EPOCHS=(26 14 14 31 39 3)
THRESHOLDS=(0.0 0.0 0.9 0.8 0.0 0.0)

I="${SLURM_ARRAY_TASK_ID:?}"

DATASET="${DATASETS[$I]}"
TAG="${TAGS[$I]}"
EPOCH="${EPOCHS[$I]}"
THR="${THRESHOLDS[$I]}"

DATA_DIR="$BASE/coref_data/$DATASET"
GOLD="$BASE/coref_data/${DATASET}_conllu_scorer/test.conllu"
CKPT="$CKPT_DIR/checkpoint_epoch_${EPOCH}.pt"

OUT="$BASE/predictions_romance6_historical_gpu_retry/$TAG/test"
PRED="$OUT/pred_epoch_${EPOCH}_thr_${THR}.conllu"
SCORE="$OUT/score_epoch_${EPOCH}_thr_${THR}.txt"

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load Transformers/4.39.3-gfbf-2023a

source "$BASE/venvs/coref_train/bin/activate"

export HF_HOME="$BASE/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

# Parametros efetivos utilizados no DEV e na tese
export TOP_SPAN_RATIO=0.008
export MAX_SPAN_WIDTH=6
export MAX_TOP_SPANS=25
export MAX_TOP_ANTECEDENTS=4
export ENCODER_BATCH_SIZE=1
export GRADIENT_CHECKPOINTING=1

cd "$REPO"

echo "===== TEST MULTILINGUAL CORRIGIDO ====="
date
echo "DATASET=$DATASET"
echo "TAG=$TAG"
echo "EPOCH=$EPOCH"
echo "THRESHOLD=$THR"
echo "CHECKPOINT=$CKPT"
echo "GOLD=$GOLD"
echo "GIT=$(git rev-parse HEAD)"

test -s "$DATA_DIR/test.jsonlines"
test -s "$GOLD"
test -s "$CKPT"
test -s "$SCORER"

# Nao sobrescrever resultados anteriores
if [ -e "$OUT" ]; then
    echo "ERRO: diretorio TEST ja existe: $OUT"
    exit 1
fi

mkdir -p "$OUT"

echo "===== PREDICAO ====="


echo "===== GPU OBRIGATORIA ====="

python3 - <<'CHECKCUDA'
import torch
print("CUDA:", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA indisponivel. TEST interrompido.")
print("GPU:", torch.cuda.get_device_name(0), flush=True)
CHECKCUDA

python3 predict_conllu.py \
    --data_dir "$DATA_DIR" \
    --gold_conllu "$GOLD" \
    --output_conllu "$PRED" \
    --save_dir "$CKPT_DIR" \
    --resume_from "$CKPT" \
    --split test \
    --mention_thresh "$THR"

test -s "$PRED"

echo "===== SCORER OFICIAL ====="

"$BASE/venvs/corefud_scorer/bin/python3" \
    "$SCORER" \
    -a exact \
    "$GOLD" \
    "$PRED" \
    -m muc bcub ceafe \
    | tee "$SCORE"

test -s "$SCORE"
grep -q '^CoNLL score:' "$SCORE"

echo "===== TEST CONCLUIDO ====="
date
