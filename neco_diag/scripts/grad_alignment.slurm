#!/bin/bash -l

#SBATCH --job-name=neco_grad_${BACKBONE}
#SBATCH --output=/zfsstore/user/s4561341/neco_diag/logs/%x_%j.out
#SBATCH --error=/zfsstore/user/s4561341/neco_diag/logs/%x_%j.err
#SBATCH --mail-user="s4561341@vuw.leidenuniv.nl"
#SBATCH --mail-type="FAIL"
#SBATCH --mem=64G
#SBATCH --time=1:30:00
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

if [ "$BACKBONE" = "dinov2" ]; then
  S=$D/ckpt/dinov2_vits14_pretrain.pth
  T=$D/ckpt/dinov2_vits14_pretrain.pth
else
  S=$D/ckpt/neco_dinov2_vits14_student.ckpt
  T=$D/ckpt/neco_dinov2_vits14_teacher.ckpt
fi

echo "gradient alignment backbone=$BACKBONE"
nvidia-smi

python -u $D/grad_alignment.py \
    --backbone $BACKBONE \
    --student-ckpt $S \
    --teacher-ckpt $T \
    --head $D/results/probe_${BACKBONE}_voc_head.pt \
    --data-root $D/data/voc/VOCSegmentation \
    --iters 20 --batch-size 8 \
    --out $D/results/grad_align_${BACKBONE}.json

echo "FINISHED neco_grad_${BACKBONE}"
