#!/bin/bash -l

#SBATCH --job-name=train_verb_from_ssl
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user="s4561341@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=70G
#SBATCH --time=7:00:00
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

echo "running train.py for verbs (SlowFast, encoder warm-started from untrained-init NeCo SSL)"
conda init
conda activate thesis_env
python ./train.py ./configs/epic_slowfast_verb.yaml \
    --init-encoder ./ckpt_ssl/ssl_epic_slowfast_verb_neco/epoch_012.pth.tar \
    --output from_ssl_neco

echo "FINISHED training (from SSL init)"
