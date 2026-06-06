#!/bin/bash
#SBATCH -A f202600026aivlabdeucaliong
#SBATCH -p normal-a100-40
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -J treino_corefpt_bert_pt_sentseg
#SBATCH -o /projects/F202600026AIVLABDEUCALION/up202000683/logs/%x-%j.out
#SBATCH -e /projects/F202600026AIVLABDEUCALION/up202000683/logs/%x-%j.err

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load Transformers/4.39.3-gfbf-2023a

BASE=/projects/F202600026AIVLABDEUCALION/up202000683
REPO=$BASE/repos/coref
DATA_DIR=$BASE/coref_data/coref_pt_sentseg
SAVE_DIR=$BASE/checkpoints_corefpt_bert_pt_sentseg

export HF_HOME=$BASE/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" "$SAVE_DIR"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

cd "$REPO"

echo "=== ENV ==="
hostname
date
pwd
which python3
python3 -V

echo "=== CHECK DATA ==="
ls -lh "$DATA_DIR/train.jsonlines"
ls -lh "$DATA_DIR/dev.jsonlines"

echo "=== CHECK CUDA ==="
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
python3 -c "import scipy; print('scipy OK:', scipy.__version__)"

echo "=== START TRAINING COREF-PT BERT ==="

python3 -u -m coref_loader.training.main_trainer \
  --data_dir "$DATA_DIR" \
  --split train \
  --epochs 20 \
  --max_seq_length 512 \
  --max_span_width 20 \
  --max_segment_len 128 \
  --bert_lr 2e-5 \
  --task_lr 1e-4 \
  --warmup_ratio 0.1 \
  --eval_every 1 \
  --save_dir "$SAVE_DIR" \
  --encoder_name neuralmind/bert-base-portuguese-cased

echo "=== DONE ==="
date
ls -lh "$SAVE_DIR"
