#!/bin/bash
#SBATCH --job-name=epic_jpg_extract
#SBATCH --output=extract_jpg_%j.log
#SBATCH --error=extract_jpg_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8        # Matches the MAX_IO_WORKERS in the Python script
#SBATCH --mem=24G                # Balanced RAM allocation
#SBATCH --time=04:00:00          # Standard JPG encoding is fast, 4 hours should be plenty


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


apptainer exec --bind /zfsstore --nv ~/thesis/vjepa2/configv6.sif python verify.py
ls 

echo "FINISHED extracting"