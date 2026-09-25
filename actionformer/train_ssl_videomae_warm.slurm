#!/bin/bash -l

#SBATCH --job-name=ssl_neco_videomae_warm
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

echo "NeCo SSL post-training on VideoMAE-L, warm-started from the supervised baseline (pipeline 2)"
conda init
conda activate thesis_env
python -u ./train_ssl.py ./configs/ssl_epic_videomae_verb.yaml --output neco_warm \
    --init-encoder ./ckpt/epic_videomae_verb_reproduce/epoch_021.pth.tar

echo "FINISHED ssl_neco_videomae_warm"
