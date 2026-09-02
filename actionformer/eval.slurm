#!/bin/bash -l

#SBATCH --job-name=eval_ssl_slowfast
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user="s4561341@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=70G
#SBATCH --time=3:00:00
#SBATCH --partition=gpu-2080ti-11g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2080_ti:4 

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

echo "running eval.py"
conda init
conda activate thesis_env
python ./eval.py ./configs/epic_slowfast_verb.yaml ./ckpt/ssl_epic_slowfast_verb_neco

echo "FINISHED Evaluation"