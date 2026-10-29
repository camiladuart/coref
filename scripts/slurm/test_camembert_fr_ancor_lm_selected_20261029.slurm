#!/bin/bash
#SBATCH -A eehpc-dev-2026d06-183g
#SBATCH -p normal-a100-80
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -o /projects/EEHPC-DEV-2026D06-183/up202000683/logs/test_camfrlm-%j.out
#SBATCH -e /projects/EEHPC-DEV-2026D06-183/up202000683/logs/test_camfrlm-%j.err

set -euo pipefail

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load Transformers/4.39.3-gfbf-2023a

BASE=/projects/EEHPC-DEV-2026D06-183/up202000683
REPO="$BASE/repos/coref"
DATA="$BASE/coref_data/french_ancor_lm"
GOLD="$BASE/coref_data/french_ancor_lm_conllu_scorer/test.conllu"
SCORER="$REPO/corefud-scorer"
SAVE="$BASE/checkpoints_mono_camembert_fr_ancor_lm_retrain_a10080_20261027_e40"
DEV="$BASE/predictions_camembert_fr_ancor_lm_retrain_a10080_dev_20261028_v3"
OUT="$BASE/predictions_camembert_fr_ancor_lm_test_selected_20261029"

EPOCH=$(cat "$DEV/best_epoch_crac.txt")
THRESHOLD=$(cat "$DEV/best_threshold_crac.txt")
CKPT="$SAVE/checkpoint_epoch_${EPOCH}.pt"

test "$EPOCH" = "35"
test "$THRESHOLD" = "0.0"
test -f "$CKPT"
test -f "$GOLD"
test -f "$DATA/test.jsonlines"

mkdir "$OUT"

export HF_HOME="$BASE/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export HF_HUB_DISABLE_TELEMETRY=1

export TOP_SPAN_RATIO=0.02
export MAX_SPAN_WIDTH=8
export MAX_TOP_SPANS=80
export MAX_TOP_ANTECEDENTS=10
export ENCODER_BATCH_SIZE=1
export GRADIENT_CHECKPOINTING=1

source "$BASE/venvs/coref_train/bin/activate"
cd "$REPO"

echo "===== TEST FRENCH ANCOR LM ====="
echo "EPOCH=$EPOCH"
echo "THRESHOLD=$THRESHOLD"
echo "CHECKPOINT=$CKPT"
date

python3 predict_conllu.py \
    --data_dir "$DATA" \
    --gold_conllu "$GOLD" \
    --output_conllu "$OUT/pred_test.conllu" \
    --save_dir "$SAVE" \
    --resume_from "$CKPT" \
    --split test \
    --mention_thresh "$THRESHOLD"

(
    cd "$SCORER"
    source "$BASE/venvs/corefud_scorer/bin/activate"

    python3 corefud-scorer.py -a exact \
        "$GOLD" \
        "$OUT/pred_test.conllu" \
        -m muc bcub ceafe \
        | tee "$OUT/test_score.txt"
)

echo "===== TEST TERMINADO ====="
date
df -h "$BASE"
