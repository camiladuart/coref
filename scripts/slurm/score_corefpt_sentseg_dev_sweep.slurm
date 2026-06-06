#!/bin/bash
#SBATCH -A f202600026aivlabdeucaliong
#SBATCH -p normal-a100-40
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --mem=40G
#SBATCH -c 8
#SBATCH -J SCORE_corefpt_dev_sweep
#SBATCH -o /projects/F202600026AIVLABDEUCALION/up202000683/logs/%x-%j.out
#SBATCH -e /projects/F202600026AIVLABDEUCALION/up202000683/logs/%x-%j.err

set -euo pipefail

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load Transformers/4.39.3-gfbf-2023a

BASE=/projects/F202600026AIVLABDEUCALION/up202000683
REPO=$BASE/repos/coref
DATA_DIR=$BASE/coref_data/coref_pt_sentseg
SCORER_CONLLU_DIR=$BASE/coref_data/coref_pt_sentseg_conllu_scorer
SAVE_DIR=$BASE/checkpoints_corefpt_bert_pt_sentseg
PRED_DIR=$BASE/predictions_corefpt_sentseg_dev
SCORER_DIR=$BASE/repos/coref/corefud-scorer
RESULT_TSV=$PRED_DIR/dev_threshold_sweep_results.tsv

mkdir -p "$BASE/logs"
mkdir -p "$SCORER_CONLLU_DIR"
mkdir -p "$PRED_DIR"

export HF_HOME=$BASE/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export HF_HUB_DISABLE_TELEMETRY=1
cd "$REPO"

echo "=== CHECK PATHS ==="
ls -lh "$DATA_DIR/dev.jsonlines"
ls -lh "$SAVE_DIR"/checkpoint_epoch_*.pt | head

echo
echo "=== CREATE SCORER-READY DEV GOLD CONLLU ==="
python3 -m coref_loader.jsonlines_to_scorer_conllu \
  --input_dir "$DATA_DIR" \
  --output_dir "$SCORER_CONLLU_DIR" \
  --splits dev

echo
echo "=== GOLD VS GOLD CHECK ==="
(
  cd "$SCORER_DIR"
  source "$BASE/venvs/corefud_scorer/bin/activate"
  python3 corefud-scorer.py -a exact \
    "$SCORER_CONLLU_DIR/dev.conllu" \
    "$SCORER_CONLLU_DIR/dev.conllu" \
    -m muc bcub ceafe
)

echo -e "epoch\tthreshold\tmuc_f1\tbcub_f1\tceafe_f1\tconll_f1" > "$RESULT_TSV"

THRESHOLDS="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"

for CKPT in $(ls "$SAVE_DIR"/checkpoint_epoch_*.pt | sort -V); do
  EPOCH=$(basename "$CKPT" .pt | sed 's/checkpoint_epoch_//')

  for THR in $THRESHOLDS; do
    echo
    echo "============================================================"
    echo "EPOCH=$EPOCH | THRESHOLD=$THR"
    echo "============================================================"

    PRED_CONLLU="$PRED_DIR/pred_dev_epoch_${EPOCH}_thr_${THR}.conllu"
    SCORE_TXT="$PRED_DIR/score_dev_epoch_${EPOCH}_thr_${THR}.txt"

    cd "$REPO"

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

text = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()

current = None
vals = {"muc": "0.00", "bcub": "0.00", "ceafe": "0.00", "conll": "0.00"}

for line in text:
    line = line.strip()
    if line in {"muc", "bcub", "ceafe"}:
        current = line
        continue

    if current and "F1:" in line:
        vals[current] = line.split("F1:")[-1].strip()
        current = None
        continue

    if line.startswith("CoNLL score:"):
        vals["conll"] = line.split(":")[-1].strip()

print(vals["muc"], vals["bcub"], vals["ceafe"], vals["conll"], sep="\t")
PY
)

    echo -e "${EPOCH}\t${THR}\t${METRICS}" >> "$RESULT_TSV"
  done
done

echo
echo "=== SELECT BEST DEV RESULT ==="

export SAVE_DIR
export RESULT_TSV

python3 - <<'PY'
import csv
import os
from pathlib import Path

result_path = Path(os.environ["RESULT_TSV"])
save_dir = Path(os.environ["SAVE_DIR"])

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
echo "=== DONE ==="
echo "Results table: $RESULT_TSV"
date
