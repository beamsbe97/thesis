#!/bin/bash -l

#SBATCH --job-name=ssl_neco_vjepa2
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user="s4561341@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=70G
#SBATCH --time=10:00:00
#SBATCH --partition=gpu-mig-40g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4g.40gb:1

module purge
module load slurm
module load CUDA/12.3.2
module load GCC/11.3.0
module load Miniforge3

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo $CUDA_VISIBLE_DEVICES
nvidia-smi

export CWD=$(pwd)
echo "[$SHELL] CWD: "$CWD

echo "running train_ssl.py (NeCo-style SSL post-training, V-JEPA2 features)"
conda init
conda activate thesis_env
python ./train_ssl.py ./configs/ssl_epic_vjepa2_verb.yaml --output neco_vjepa2 --ckpt-dir ./ckpt

echo "FINISHED self-supervised training (V-JEPA2)"
