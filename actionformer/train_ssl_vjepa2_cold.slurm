#!/bin/bash -l

#SBATCH --job-name=ssl_neco_vjepa2_cold
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

# pipeline 3, stage A for V-JEPA2: NeCo SSL on an UNTRAINED encoder.
# --ckpt-dir points at ckpt_ssl/, which has no epic_vjepa2_verb* folder, so
# train_ssl.py's supervised-checkpoint auto-discovery finds nothing and the
# encoder stays at random init (same trick as the SlowFast run, job 4796515)
echo "running train_ssl.py (NeCo SSL pretraining, V-JEPA2, untrained init)"
conda init
conda activate thesis_env
python ./train_ssl.py ./configs/ssl_epic_vjepa2_verb.yaml --output neco_vjepa2_cold --ckpt-dir ./ckpt_ssl

echo "FINISHED self-supervised pretraining (V-JEPA2, untrained init)"
