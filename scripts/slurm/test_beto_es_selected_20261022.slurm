#!/bin/bash
#SBATCH --job-name=test_beto_es
#SBATCH --account=eehpc-dev-2026d06-183g
#SBATCH --partition=normal-a100-80
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/projects/EEHPC-DEV-2026D06-183/up202000683/logs/test_beto_es-%j.out
#SBATCH --error=/projects/EEHPC-DEV-2026D06-183/up202000683/logs/test_beto_es-%j.err

set -euo pipefail

BASE=/projects/EEHPC-DEV-2026D06-183/up202000683
REPO="$BASE/repos/coref"
DATA="$BASE/coref_data/spanish_ancora"
GOLD="$BASE/coref_data/spanish_ancora_conllu_scorer/test.conllu"
SAVE="$BASE/checkpoints_mono_beto_spanish_ancora_retrain_a10080_20261021_e40"
DEV="$BASE/predictions_beto_spanish_ancora_retrain_a10080_dev_20261022"
OUT="$BASE/predictions_beto_spanish_ancora_retrain_a10080_test_20261022"
SCORER="$REPO/corefud-scorer"

BEST_EPOCH=$(cat "$DEV/best_epoch_crac.txt")
BEST_THRESHOLD=$(cat "$DEV/best_threshold_crac.txt")

CKPT="$SAVE/checkpoint_epoch_${BEST_EPOCH}.pt"
PRED="$OUT/pred_test_epoch_${BEST_EPOCH}_thr_${BEST_THRESHOLD}.conllu"
SCORE="$OUT/score_test_epoch_${BEST_EPOCH}_thr_${BEST_THRESHOLD}.txt"

echo "===== VERIFICACOES ====="

test "$BEST_EPOCH" = "29"
test "$BEST_THRESHOLD" = "0.0"

test -f "$CKPT"
test -f "$DATA/test.jsonlines"
test -f "$GOLD"
test -f "$SCORER/corefud-scorer.py"
test ! -e "$OUT"

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load Transformers/4.39.3-gfbf-2023a

source "$BASE/venvs/coref_train/bin/activate"

export HF_HOME="$BASE/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

echo "===== CONFIGURACAO ====="
echo "GPU:"
python3 -c 'import torch; print(torch.cuda.get_device_name(0))'
echo "CHECKPOINT=$CKPT"
echo "THRESHOLD=$BEST_THRESHOLD"
echo "DATA=$DATA"
echo "GOLD=$GOLD"

mkdir "$OUT"

cd "$REPO"

echo "===== PREDICAO TEST ====="
python3 predict_conllu.py \
  --data_dir "$DATA" \
  --gold_conllu "$GOLD" \
  --output_conllu "$PRED" \
  --save_dir "$SAVE" \
  --resume_from "$CKPT" \
  --split test \
  --mention_thresh "$BEST_THRESHOLD"

echo "===== COREFUD SCORER ====="
(
  cd "$SCORER"
  source "$BASE/venvs/corefud_scorer/bin/activate"

  python3 corefud-scorer.py -a exact \
    "$GOLD" "$PRED" \
    -m muc bcub ceafe | tee "$SCORE"
)

echo "===== RESULTADO TEST ====="
grep -E '^muc$|^bcub$|^ceafe$|^CoNLL score:' "$SCORE"

echo "===== TESTE TERMINADO ====="
date
