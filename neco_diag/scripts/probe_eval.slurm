#!/bin/bash -l

#SBATCH --job-name=neco_probe_eval
#SBATCH --output=/zfsstore/user/s4561341/neco_diag/logs/%x_%j.out
#SBATCH --error=/zfsstore/user/s4561341/neco_diag/logs/%x_%j.err
#SBATCH --mail-user="s4561341@vuw.leidenuniv.nl"
#SBATCH --mail-type="FAIL"
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --partition=gpu-a100-80g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

module purge
module load ALICE/default
module load slurm
module load Miniforge3

set -e
conda init >/dev/null
conda activate neco_diag
unset LD_LIBRARY_PATH

export PYTHONPATH=/zfsstore/user/s4561341/neco_repo
D=/zfsstore/user/s4561341/neco_diag

ckpt_for() {
  if [ "$1" = "dinov2" ]; then echo "$D/ckpt/dinov2_vits14_pretrain.pth";
  else echo "$D/ckpt/neco_dinov2_vits14_student.ckpt"; fi
}

# Evaluate head X on backbone Y: the frozen trained-head transfer test.
for pair in dinov2:dinov2 neco:neco dinov2:neco neco:dinov2; do
  train_bb=${pair%%:*}
  eval_bb=${pair##*:}
  python -u $D/linear_probe.py \
      --backbone $eval_bb \
      --ckpt $(ckpt_for $eval_bb) \
      --data voc \
      --data-root $D/data/voc/VOCSegmentation \
      --batch-size 64 --workers 8 \
      --eval-only \
      --head $D/results/probe_${train_bb}_voc_head.pt \
      --out $D/results/xfer_${train_bb}head_${eval_bb}.json
done

echo "FINISHED neco_probe_eval"
