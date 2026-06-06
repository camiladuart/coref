#!/bin/bash
set -euo pipefail
module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load Transformers/4.39.3-gfbf-2023a
BASE=/projects/F202600026AIVLABDEUCALION/up202000683
REPO=$BASE/repos/coref
DATA_DIR=$BASE/coref_data/coref_pt_sentseg
SCORER_CONLLU_DIR=$BASE/coref_data/coref_pt_sentseg_conllu_scorer
SCORER_DIR=$BASE/repos/coref/corefud-scorer
OUT_DIR=$BASE/outputs/scorer_checks_sentseg

mkdir -p "$OUT_DIR"

cd "$REPO"

echo "=== JSONLINES DOC COUNTS ==="
wc -l "$DATA_DIR/train.jsonlines"
wc -l "$DATA_DIR/dev.jsonlines"
wc -l "$DATA_DIR/test.jsonlines"

echo
echo "=== CREATE SCORER-READY CONLLU FROM SENTSEG ==="
python3 -m coref_loader.jsonlines_to_scorer_conllu \
  --input_dir "$DATA_DIR" \
  --output_dir "$SCORER_CONLLU_DIR" \
  --splits dev test

echo
echo "=== CONLLU DOC COUNTS ==="
grep -c "^# newdoc" "$SCORER_CONLLU_DIR/dev.conllu"
grep -c "^# newdoc" "$SCORER_CONLLU_DIR/test.conllu"

echo
echo "=== BASIC CONLLU FORMAT CHECK ==="
python3 - <<'PY'
from pathlib import Path

base = Path("/projects/F202600026AIVLABDEUCALION/up202000683/coref_data/coref_pt_sentseg_conllu_scorer")

for split in ["dev", "test"]:
    path = base / f"{split}.conllu"
    bad = []

    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) != 10:
            bad.append((i, len(cols), line[:120]))

    if bad:
        print(f"[BAD] {split}: {len(bad)} malformed lines")
        for x in bad[:10]:
            print(x)
        raise SystemExit(1)

    print(f"[OK] {split}: all token lines have 10 columns")
PY

echo
echo "=== GOLD VS GOLD SCORER TEST ON DEV ==="
cd "$SCORER_DIR"
source "$BASE/venvs/corefud_scorer/bin/activate"

python3 corefud-scorer.py -a exact \
  "$SCORER_CONLLU_DIR/dev.conllu" \
  "$SCORER_CONLLU_DIR/dev.conllu" \
  -m muc bcub ceafe \
  | tee "$OUT_DIR/dev_gold_vs_gold.txt"

echo
echo "=== DONE ==="
echo "Check output: $OUT_DIR/dev_gold_vs_gold.txt"
