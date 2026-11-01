#!/bin/bash
#SBATCH --job-name=it_test_e38
#SBATCH --account=eehpc-dev-2026d06-183g
#SBATCH --partition=normal-a100-80
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/projects/EEHPC-DEV-2026D06-183/up202000683/logs/test_umberto_it-%j.out
#SBATCH --error=/projects/EEHPC-DEV-2026D06-183/up202000683/logs/test_umberto_it-%j.err

set -euo pipefail

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load Transformers/4.39.3-gfbf-2023a

BASE=/projects/EEHPC-DEV-2026D06-183/up202000683
REPO="$BASE/repos/coref"
DATA="$BASE/coref_data/OntoCorefIT"
GOLD="$BASE/coref_data/OntoCorefIT_conllu_scorer/test.conllu"
SCORER="$REPO/corefud-scorer"
SAVE="$BASE/checkpoints_mono_umberto_ontocorefit_retrain_a10080_20261030_e40"
CKPT="$SAVE/checkpoint_epoch_38.pt"
OUT="$BASE/predictions_umberto_ontocorefit_test_e38_20261101"

test -s "$DATA/test.jsonlines"
test -s "$GOLD"
test -s "$CKPT"
test -d "$SCORER"
test ! -e "$OUT" || { echo "ERRO: destino ja existe"; exit 1; }

mkdir "$OUT"

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

source "$BASE/venvs/coref_train/bin/activate"
cd "$REPO"

echo "===== ITALIAN TEST ====="
echo "Checkpoint: $CKPT"
echo "Threshold: 0.0"
date

python3 predict_conllu.py \
  --data_dir "$DATA" \
  --gold_conllu "$GOLD" \
  --output_conllu "$OUT/pred_test.conllu" \
  --save_dir "$SAVE" \
  --resume_from "$CKPT" \
  --split test \
  --mention_thresh 0.0

(
  cd "$SCORER"
  source "$BASE/venvs/corefud_scorer/bin/activate"

  python3 corefud-scorer.py -a exact \
    "$GOLD" \
    "$OUT/pred_test.conllu" \
    -m muc bcub ceafe \
    | tee "$OUT/test_score.txt"
)

echo "===== ITALIAN TEST COMPLETED ====="
date
