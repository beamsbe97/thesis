#!/bin/bash -l

#SBATCH --job-name=vm_spatial_extract
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user="s4561341@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=100G
#SBATCH --cpus-per-task=60
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu-mig-40g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4g.40gb:1

# Fan-out across nodes: submit several copies of this job, each handling a
# disjoint block of the global shards. Override with e.g.
#   sbatch --export=ALL,NPROC=6,GLOBAL_SHARDS=24,SHARD_BASE=6 \
#          --job-name=vm_spatial_extract_b6 extract_videomae_spatial.slurm
NPROC=${NPROC:-6}
GLOBAL_SHARDS=${GLOBAL_SHARDS:-$NPROC}
SHARD_BASE=${SHARD_BASE:-0}

module purge
module load ALICE/default
module load slurm
module load CUDA/12.3.2
module load GCC/11.3.0
module load Miniforge3

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

conda init
conda activate thesis_env
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

ROOT=$HOME/vm_spatial
DATA=$HOME/thesis/actionformer/data/epic_kitchens
OUT=$DATA/features_videomae_verb_spatial
JSON=$DATA/annotations/epic_kitchens_100_verb.json
VIDEO_ROOT=/zfsstore/courses/2025-2026/4343SADL6/g10/EPIC-KITCHENS
CKPT=$HOME/thesis/pretrained/videomae_l_epic_verb.pth

echo "NPROC=$NPROC GLOBAL_SHARDS=$GLOBAL_SHARDS SHARD_BASE=$SHARD_BASE"
cd $ROOT

for SPLIT in training validation; do
  echo "=== split $SPLIT : launching $NPROC shards (global base $SHARD_BASE / $GLOBAL_SHARDS) ==="
  for k in $(seq 0 $((NPROC-1))); do
    idx=$((SHARD_BASE + k))
    python -u tools/extract_videomae_spatial.py \
      --video-root $VIDEO_ROOT \
      --json-file $JSON \
      --output $OUT \
      --ckpt $CKPT \
      --split $SPLIT \
      --num-crops 16 --crop-scale 0.25 1.0 \
      --num-shards $GLOBAL_SHARDS --shard-index $idx \
      --batch-size 4 --decode-threads 8 \
      > $ROOT/log_${SPLIT}_${idx}.log 2>&1 &
  done
  wait
  echo "=== split $SPLIT : shards done ==="
done

echo "FINISHED vm_spatial_extract (base $SHARD_BASE)"
