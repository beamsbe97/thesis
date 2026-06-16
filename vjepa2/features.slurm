#!/bin/bash -l

#SBATCH --job-name=vjepa2_features
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user="s4561341@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=70G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-mig-40g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3g.40gb:1

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

echo "extracting VJEPA2 feaures"


apptainer exec --nv ~/thesis/vjepa2/configv6.sif python features.py -i ~/kitchen-2X -o ~/EPIC-KITCHENS/embeddings --clip-frame-step 1 \
  --snippet-stride 16

echo "FINISHED extracting"