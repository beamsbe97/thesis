#!/bin/bash -l

#SBATCH --job-name=ssl_neco_rate
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user="s4561341@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=70G
#SBATCH --time=3:00:00
#SBATCH --partition=gpu-mig-40g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4g.40gb:1

module purge

module load ALICE/default
module load slurm
module load CUDA/12.3.2
module load GCC/11.3.0
module load Miniforge3

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo $CUDA_VISIBLE_DEVICES
nvidia-smi

export CWD=$(pwd)
echo "[$SHELL] CWD: "$CWD

echo "NeCo SSL from random init, temporal crops + per-view rate augmentation [0.5, 2] (SlowFast)"
conda init
conda activate thesis_env
# --ckpt-dir ./ckpt_ssl has no epic_slowfast_verb_rate* folder -> random init
python -u ./train_ssl.py ./configs/ssl_epic_slowfast_verb_rate.yaml --output neco --ckpt-dir ./ckpt_ssl

echo "FINISHED ssl_neco_rate"
