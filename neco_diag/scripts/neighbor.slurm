#!/bin/bash -l

#SBATCH --job-name=neco_knn_${BACKBONE}
#SBATCH --output=/zfsstore/user/s4561341/neco_diag/logs/%x_%j.out
#SBATCH --error=/zfsstore/user/s4561341/neco_diag/logs/%x_%j.err
#SBATCH --mail-user="s4561341@vuw.leidenuniv.nl"
#SBATCH --mail-type="FAIL"
#SBATCH --mem=48G
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

if [ "$BACKBONE" = "dinov2" ]; then
  C=$D/ckpt/dinov2_vits14_pretrain.pth
else
  C=$D/ckpt/neco_dinov2_vits14_student.ckpt
fi

echo "neighbour/label agreement backbone=$BACKBONE"
nvidia-smi

python -u $D/neighbor_label_agreement.py \
    --backbone $BACKBONE \
    --ckpt $C \
    --data-root $D/data/voc/VOCSegmentation \
    --n-images 400 --k 10 \
    --out $D/results/neighbor_agreement_${BACKBONE}.json

echo "FINISHED neco_knn_${BACKBONE}"
