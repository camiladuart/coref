#!/bin/bash
#SBATCH -A eehpc-dev-2026d06-183g
#SBATCH -p normal-a100-40
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -o /projects/EEHPC-DEV-2026D06-183/up202000683/logs/%x-%j.out
#SBATCH -e /projects/EEHPC-DEV-2026D06-183/up202000683/logs/%x-%j.err

set -euo pipefail

BASE=/projects/EEHPC-DEV-2026D06-183/up202000683
REPO="$BASE/repos/coref"
SCORER="$REPO/corefud-scorer/corefud-scorer.py"

: "${DATASET:?DATASET nao informado}"
: "${TAG:?TAG nao informado}"

CKPT_DIR="$BASE/checkpoints_romance6_mdeberta_corefpt_fixed_thesisparams_e40"
DATA_DIR="$BASE/coref_data/$DATASET"
GOLD="$BASE/coref_data/${DATASET}_conllu_scorer/dev.conllu"
RESULT_DIR="$BASE/predictions_romance6_fixed_thesisparams/${TAG}/dev"
RESULT_TSV="$RESULT_DIR/dev_threshold_sweep_results.tsv"

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

# Parametros efetivos da Tabela 5.1
export TOP_SPAN_RATIO=0.005
export MAX_SPAN_WIDTH=6
export MAX_TOP_SPANS=30
export MAX_TOP_ANTECEDENTS=4
export ENCODER_BATCH_SIZE=1
export GRADIENT_CHECKPOINTING=0

cd "$REPO"

echo "===== DEV SWEEP ====="
echo "DATASET=$DATASET"
echo "TAG=$TAG"
echo "CHECKPOINTS=$CKPT_DIR"
echo "GOLD=$GOLD"
echo "GIT=$(git rev-parse HEAD)"
echo "TOP_SPAN_RATIO=$TOP_SPAN_RATIO"
echo "MAX_SPAN_WIDTH=$MAX_SPAN_WIDTH"
echo "MAX_TOP_SPANS=$MAX_TOP_SPANS"
echo "MAX_TOP_ANTECEDENTS=$MAX_TOP_ANTECEDENTS"

test -s "$DATA_DIR/dev.jsonlines"
test -s "$GOLD"
test -s "$SCORER"

for EPOCH in $(seq 1 40); do
    test -s "$CKPT_DIR/checkpoint_epoch_${EPOCH}.pt"
done

# Verificar correspondencia de tokens e documentos
python3 - "$DATA_DIR/dev.jsonlines" "$GOLD" <<'PY'
import json
import sys

json_path, gold_path = sys.argv[1:]

with open(json_path, encoding="utf-8") as f:
    docs = [json.loads(line) for line in f if line.strip()]

gold_docs = []
tokens = None

with open(gold_path, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if line.startswith("# newdoc"):
            if tokens is not None:
                gold_docs.append(tokens)
            tokens = []
        elif line and not line.startswith("#"):
            fields = line.split("\t")
            if len(fields) >= 10 and "-" not in fields[0] and "." not in fields[0]:
                if tokens is None:
                    raise RuntimeError("Gold sem marcador newdoc")
                tokens.append(fields[1])

if tokens is not None:
    gold_docs.append(tokens)

assert len(docs) == len(gold_docs), (
    f"Documentos: JSONL={len(docs)} gold={len(gold_docs)}"
)

for i, (doc, gold_tokens) in enumerate(zip(docs, gold_docs)):
    json_tokens = [
        token
        for sentence in doc["sentences"]
        for token in sentence
    ]
    assert json_tokens == gold_tokens, (
        f"Tokens desalinhados no documento {i}: {doc.get('doc_key')}"
    )

print(f"ALINHAMENTO OK: {len(docs)} documentos", flush=True)
PY

if [ -e "$RESULT_DIR" ]; then
    echo "ERRO: diretorio DEV ja existe: $RESULT_DIR"
    exit 1
fi

mkdir -p "$RESULT_DIR"

printf 'epoch\tthreshold\tmuc_f1\tbcub_f1\tceafe_f1\tconll_f1\n' \
    > "$RESULT_TSV"

for EPOCH in $(seq 1 40); do
    CKPT="$CKPT_DIR/checkpoint_epoch_${EPOCH}.pt"

    for THR in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do

        echo "===== EPOCH=$EPOCH THRESHOLD=$THR ====="

        PRED="$RESULT_DIR/pred_epoch_${EPOCH}_thr_${THR}.conllu"
        SCORE="$RESULT_DIR/score_epoch_${EPOCH}_thr_${THR}.txt"

        python3 predict_conllu.py \
            --data_dir "$DATA_DIR" \
            --gold_conllu "$GOLD" \
            --output_conllu "$PRED" \
            --save_dir "$CKPT_DIR" \
            --resume_from "$CKPT" \
            --split dev \
            --mention_thresh "$THR"

        "$BASE/venvs/corefud_scorer/bin/python3" \
            "$SCORER" -a exact \
            "$GOLD" "$PRED" \
            -m muc bcub ceafe > "$SCORE"

        python3 - "$EPOCH" "$THR" "$SCORE" "$RESULT_TSV" <<'PY'
import sys

epoch, threshold, score_file, result_file = sys.argv[1:]

values = {}
metric = None

for line in open(score_file, encoding="utf-8"):
    text = line.strip()
    low = text.lower()

    if low in {"muc", "bcub", "ceafe"}:
        metric = low
    elif metric and "f1:" in low:
        values[metric] = float(
            low.split("f1:")[-1].replace("%", "").strip()
        )
        metric = None

if set(values) != {"muc", "bcub", "ceafe"}:
    raise RuntimeError(f"Metricas incompletas: {values}")

conll = sum(values.values()) / 3

with open(result_file, "a", encoding="utf-8") as f:
    f.write(
        f"{epoch}\t{threshold}\t"
        f"{values['muc']:.4f}\t"
        f"{values['bcub']:.4f}\t"
        f"{values['ceafe']:.4f}\t"
        f"{conll:.4f}\n"
    )
PY

        # Conserva as pontuacoes; evita acumular centenas de predicoes
        rm -- "$PRED"
    done
done

python3 - "$RESULT_TSV" "$RESULT_DIR" <<'PY'
import csv
import sys
from pathlib import Path

tsv = Path(sys.argv[1])
out = Path(sys.argv[2])

with tsv.open(encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

assert len(rows) == 400, f"Esperadas 400 avaliacoes, obtidas {len(rows)}"

best = max(rows, key=lambda r: float(r["conll_f1"]))

(out / "best_epoch_crac.txt").write_text(best["epoch"] + "\n")
(out / "best_threshold_crac.txt").write_text(best["threshold"] + "\n")

print("===== MELHOR DEV =====")
print(best)
PY

echo "===== DEV COMPLETED ====="
date
