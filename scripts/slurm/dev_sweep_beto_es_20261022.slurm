#!/bin/bash
#SBATCH -A eehpc-dev-2026d06-183g
#SBATCH -p normal-a100-80
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -o /projects/EEHPC-DEV-2026D06-183/up202000683/logs/dev_beto_es-%j.out
#SBATCH -e /projects/EEHPC-DEV-2026D06-183/up202000683/logs/dev_beto_es-%j.err

set -euo pipefail

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load Transformers/4.39.3-gfbf-2023a

BASE=/projects/EEHPC-DEV-2026D06-183/up202000683
REPO=$BASE/repos/coref
SCORER_DIR=$BASE/repos/coref/corefud-scorer

DATASET=spanish_ancora
TAG=beto_spanish_ancora_retrain_a10080_20261021
EPOCHS=40
THRESHOLDS="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"

DATA_DIR=$BASE/coref_data/$DATASET
SCORER_CONLLU_DIR=$BASE/coref_data/${DATASET}_conllu_scorer
SAVE_DIR=$BASE/checkpoints_mono_${TAG}_e${EPOCHS}
PRED_DIR=$BASE/predictions_beto_spanish_ancora_retrain_a10080_dev_20261022
RESULT_TSV=$PRED_DIR/dev_threshold_sweep_results.tsv

mkdir "$PRED_DIR"

export HF_HOME=$BASE/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export HF_HUB_DISABLE_TELEMETRY=1

export TOP_SPAN_RATIO=0.4
export MAX_SPAN_WIDTH=20
export MAX_TOP_SPANS=3900
export MAX_TOP_ANTECEDENTS=50
export ENCODER_BATCH_SIZE=1
export GRADIENT_CHECKPOINTING=1

source "$BASE/venvs/coref_train/bin/activate"

cd "$REPO"

echo "===== DEV SWEEP MONO ====="
echo "DATASET=$DATASET"
echo "TAG=$TAG"
echo "EPOCHS=$EPOCHS"
echo "THRESHOLDS=$THRESHOLDS"
echo "SAVE_DIR=$SAVE_DIR"
echo "PRED_DIR=$PRED_DIR"
echo "RESULT_TSV=$RESULT_TSV"
date

echo "===== CHECK PATHS ====="
ls -lh "$DATA_DIR/dev.jsonlines"
ls -lh "$SCORER_CONLLU_DIR/dev.conllu"
ls -lh "$SAVE_DIR"/checkpoint_epoch_*.pt | head

echo -e "epoch\tthreshold\tmuc_f1\tbcub_f1\tceafe_f1\tconll_f1" > "$RESULT_TSV"

for CKPT in $(ls "$SAVE_DIR"/checkpoint_epoch_*.pt | sort -V); do
  EPOCH=$(basename "$CKPT" .pt | sed 's/checkpoint_epoch_//')

  for THR in $THRESHOLDS; do
    echo
    echo "============================================================"
    echo "EPOCH=$EPOCH | THRESHOLD=$THR"
    echo "============================================================"

    PRED_CONLLU="$PRED_DIR/pred_dev_epoch_${EPOCH}_thr_${THR}.conllu"
    SCORE_TXT="$PRED_DIR/score_dev_epoch_${EPOCH}_thr_${THR}.txt"

    python3 predict_conllu.py \
      --data_dir "$DATA_DIR" \
      --gold_conllu "$SCORER_CONLLU_DIR/dev.conllu" \
      --output_conllu "$PRED_CONLLU" \
      --save_dir "$SAVE_DIR" \
      --resume_from "$CKPT" \
      --split dev \
      --mention_thresh "$THR"

    (
      cd "$SCORER_DIR"
      source "$BASE/venvs/corefud_scorer/bin/activate"

      python3 corefud-scorer.py -a exact \
        "$SCORER_CONLLU_DIR/dev.conllu" \
        "$PRED_CONLLU" \
        -m muc bcub ceafe \
        | tee "$SCORE_TXT"
    )

    METRICS=$(python3 - "$SCORE_TXT" <<'PY'
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace").splitlines()

current = None
vals = {"muc": "0.00", "bcub": "0.00", "ceafe": "0.00", "conll": "0.00"}

for line in text:
    s = line.strip()

    if s in {"muc", "bcub", "ceafe"}:
        current = s
        continue

    if current and "F1:" in s:
        vals[current] = s.split("F1:")[-1].strip()
        current = None
        continue

    if s.startswith("CoNLL score:"):
        vals["conll"] = s.split(":")[-1].strip()

print(vals["muc"], vals["bcub"], vals["ceafe"], vals["conll"], sep="\t")
PY
)

    echo -e "${EPOCH}\t${THR}\t${METRICS}" >> "$RESULT_TSV"
  done
done

echo
echo "===== SELECT BEST DEV RESULT ====="

export SAVE_DIR
export RESULT_TSV
export PRED_DIR

python3 - <<'PY'
import csv
import os
from pathlib import Path

result_path = Path(os.environ["RESULT_TSV"])
save_dir = Path(os.environ["PRED_DIR"])

rows = []
with open(result_path, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        row["conll_f1"] = float(row["conll_f1"])
        rows.append(row)

if not rows:
    raise SystemExit("No rows found in result TSV")

best = max(rows, key=lambda r: r["conll_f1"])

print("BEST DEV RESULT")
print("epoch:", best["epoch"])
print("threshold:", best["threshold"])
print("muc_f1:", best["muc_f1"])
print("bcub_f1:", best["bcub_f1"])
print("ceafe_f1:", best["ceafe_f1"])
print("conll_f1:", best["conll_f1"])

(save_dir / "best_epoch_crac.txt").write_text(str(best["epoch"]) + "\n", encoding="utf-8")
(save_dir / "best_threshold_crac.txt").write_text(str(best["threshold"]) + "\n", encoding="utf-8")

print("Saved:")
print(save_dir / "best_epoch_crac.txt")
print(save_dir / "best_threshold_crac.txt")
PY

echo
echo "===== DONE DEV SWEEP ====="
echo "Results table: $RESULT_TSV"
date
