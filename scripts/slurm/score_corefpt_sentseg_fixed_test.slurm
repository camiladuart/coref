#!/bin/bash
#SBATCH -A eehpc-dev-2026d06-183g
#SBATCH -p normal-a100-40
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH --mem=40G
#SBATCH -c 8
#SBATCH -J SCORE_corefpt_fixed_test
#SBATCH -o /projects/EEHPC-DEV-2026D06-183/up202000683/logs/%x-%j.out
#SBATCH -e /projects/EEHPC-DEV-2026D06-183/up202000683/logs/%x-%j.err

set -euo pipefail

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load Transformers/4.39.3-gfbf-2023a

BASE=/projects/EEHPC-DEV-2026D06-183/up202000683
REPO=$BASE/repos/coref

DATA_DIR=$BASE/coref_data/coref_pt_sentseg_fixed
SCORER_CONLLU_DIR=$BASE/coref_data/coref_pt_sentseg_fixed_conllu_scorer
SAVE_DIR=$BASE/checkpoints_corefpt_bert_pt_sentseg_fixed_e40
PRED_DIR=$BASE/predictions_corefpt_sentseg_fixed_test_e40
SCORER_DIR=$REPO/corefud-scorer

BEST_EPOCH=$(cat "$SAVE_DIR/best_epoch_crac.txt")
BEST_THRESHOLD=$(cat "$SAVE_DIR/best_threshold_crac.txt")
CKPT=$SAVE_DIR/checkpoint_epoch_${BEST_EPOCH}.pt

PRED_CONLLU=$PRED_DIR/pred_test_epoch_${BEST_EPOCH}_thr_${BEST_THRESHOLD}.conllu
SCORE_TXT=$PRED_DIR/score_test_epoch_${BEST_EPOCH}_thr_${BEST_THRESHOLD}.txt

mkdir -p "$BASE/logs"
mkdir -p "$SCORER_CONLLU_DIR"
mkdir -p "$PRED_DIR"

export HF_HOME=$BASE/hf_cache
export HF_HUB_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export HF_HUB_DISABLE_TELEMETRY=1

cd "$REPO"

echo "============================================================"
echo "COREF-PT FIXED — FINAL TEST"
echo "============================================================"
echo "DATA: $DATA_DIR"
echo "CHECKPOINT: $CKPT"
echo "BEST EPOCH FROM DEV: $BEST_EPOCH"
echo "BEST THRESHOLD FROM DEV: $BEST_THRESHOLD"
echo "PREDICTION: $PRED_CONLLU"
echo

echo "=== GIT ==="
git rev-parse HEAD
git status -sb

echo
echo "=== CREATE SCORER-READY TEST GOLD ==="
python3 -m coref_loader.jsonlines_to_scorer_conllu \
  --input_dir "$DATA_DIR" \
  --output_dir "$SCORER_CONLLU_DIR" \
  --splits test

echo
echo "=== GOLD VS GOLD CHECK ==="
(
  cd "$SCORER_DIR"
  source "$BASE/venvs/corefud_scorer/bin/activate"

  python3 corefud-scorer.py -a exact \
    "$SCORER_CONLLU_DIR/test.conllu" \
    "$SCORER_CONLLU_DIR/test.conllu" \
    -m muc bcub ceafe
)

echo
echo "=== FINAL TEST PREDICTION ==="

cd "$REPO"

python3 predict_conllu.py \
  --data_dir "$DATA_DIR" \
  --gold_conllu "$SCORER_CONLLU_DIR/test.conllu" \
  --output_conllu "$PRED_CONLLU" \
  --save_dir "$SAVE_DIR" \
  --resume_from "$CKPT" \
  --split test \
  --mention_thresh "$BEST_THRESHOLD"

echo
echo "=== FINAL TEST SCORE ==="

(
  cd "$SCORER_DIR"
  source "$BASE/venvs/corefud_scorer/bin/activate"

  python3 corefud-scorer.py -a exact \
    "$SCORER_CONLLU_DIR/test.conllu" \
    "$PRED_CONLLU" \
    -m muc bcub ceafe \
    | tee "$SCORE_TXT"
)

echo
echo "============================================================"
echo "FINAL TEST COMPLETED"
echo "Epoch selected on DEV: $BEST_EPOCH"
echo "Threshold selected on DEV: $BEST_THRESHOLD"
echo "Score file: $SCORE_TXT"
echo "============================================================"
date
