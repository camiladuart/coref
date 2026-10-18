#!/bin/bash
#SBATCH -A eehpc-dev-2026d06-183g
#SBATCH -p normal-a100-40
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -J diagnostico_test_pt_gpu
#SBATCH -o /projects/EEHPC-DEV-2026D06-183/up202000683/logs/%x-%j.out
#SBATCH -e /projects/EEHPC-DEV-2026D06-183/up202000683/logs/%x-%j.err

set -euo pipefail

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load Transformers/4.39.3-gfbf-2023a

BASE=/projects/EEHPC-DEV-2026D06-183/up202000683
cd "$BASE/repos/coref"

source "$BASE/venvs/coref_train/bin/activate"

export HF_HOME="$BASE/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

export TOP_SPAN_RATIO=0.008
export MAX_SPAN_WIDTH=6
export MAX_TOP_SPANS=25
export MAX_TOP_ANTECEDENTS=4
export ENCODER_BATCH_SIZE=1
export GRADIENT_CHECKPOINTING=1

DATA="$BASE/coref_data/coref_pt_sentseg_fixed"
GOLD="$BASE/coref_data/coref_pt_sentseg_fixed_conllu_scorer/test.conllu"
CKPT="$BASE/checkpoints_romance6_mdeberta_corefpt_fixed_thesisparams_e40"
OUT="$BASE/predictions_romance6_historical_gpu_diagnostic"

mkdir -p "$OUT"


echo "===== VERIFICACAO OBRIGATORIA DA GPU ====="

python3 - <<'CHECKCUDA'
import torch
assert torch.cuda.is_available(), "CUDA indisponivel"
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
CHECKCUDA

echo "===== INFERENCIA TEST PORTUGUES ====="
date

python3 predict_conllu.py \
  --data_dir "$DATA" \
  --gold_conllu "$GOLD" \
  --output_conllu "$OUT/test_pt_epoch26_historical_gpu.conllu" \
  --save_dir "$CKPT" \
  --resume_from "$CKPT/checkpoint_epoch_26.pt" \
  --split test \
  --mention_thresh 0.0

echo "===== SCORER ====="

"$BASE/venvs/corefud_scorer/bin/python3" \
  "$BASE/repos/coref/corefud-scorer/corefud-scorer.py" \
  -a exact \
  "$GOLD" \
  "$OUT/test_pt_epoch26_historical_gpu.conllu" \
  -m muc bcub ceafe \
  | tee "$OUT/test_pt_epoch26_historical_gpu_score.txt"

echo "===== TESTE CONCLUIDO ====="
date
