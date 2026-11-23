#!/bin/bash -l
#SBATCH --job-name=ssl_fixed
#SBATCH --time=3:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user="s4561341@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=70G
#SBATCH --partition=gpu-mig-40g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4g.40gb:1

module purge
module load ALICE/default
module load slurm
module load CUDA/12.3.2
module load GCC/11.3.0
module load Miniforge3
nvidia-smi
conda init
conda activate thesis_env
SEEDARG=""; [ -n "$SEED" ] && SEEDARG="--seed $SEED"

# Bug-fixed NeCo SSL (2026-11-22). Submit with
#   sbatch --export=ALL,CONFIG=configs/ssl_epic_slowfast_verb_fixed_v2.yaml,OUT=fixed_v2_s0[,SEED=1][,WARM=<sup ckpt>] ssl_fixed.slurm
# No WARM -> from scratch (pipeline 3 stage A); WARM -> post-training from the
# supervised checkpoint's EMA weights (pipeline 2).
: "${CONFIG:?}" "${OUT:?}"
INIT="--from-scratch"; [ -n "$WARM" ] && INIT="--init-encoder $WARM"
echo "SSL fixed: $CONFIG out=$OUT seed=${SEED:-default} init=$INIT"
python -u ./train_ssl.py $CONFIG $SEEDARG $INIT --output $OUT
