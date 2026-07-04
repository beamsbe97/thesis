#!/bin/bash -l

#SBATCH --job-name=ffmpeg_cpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user="s4561341@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=70G
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00
#SBATCH --partition=cpu-zen4
#SBATCH --ntasks=1

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

echo "re-encoding videos to 60fps"


apptainer exec --bind /zfsstore --nv ~/thesis/vjepa2/configv6.sif python ffmpeg.py
ls 

echo "FINISHED extracting"