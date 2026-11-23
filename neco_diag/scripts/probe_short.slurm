#!/bin/bash -l

#SBATCH --job-name=neco_probe
#SBATCH --output=/zfsstore/user/s4561341/neco_diag/logs/%x_%j.out
#SBATCH --error=/zfsstore/user/s4561341/neco_diag/logs/%x_%j.err
#SBATCH --mail-user="s4561341@vuw.leidenuniv.nl"
#SBATCH --mail-type="FAIL"
#SBATCH --mem=32G
#SBATCH --time=3:00:00
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

echo "probe backbone=$BACKBONE dataset=$DATASET"
nvidia-smi

python -u $D/linear_probe.py \
    --backbone $BACKBONE \
    --ckpt $CKPT \
    --data $DATASET \
    --data-root $DATA_ROOT \
    --epochs ${EPOCHS:-25} \
    --batch-size ${BS:-128} \
    --workers 8 \
    --out $D/results/probe_${BACKBONE}_${DATASET}.json

echo "FINISHED probe_${BACKBONE}_${DATASET}"
