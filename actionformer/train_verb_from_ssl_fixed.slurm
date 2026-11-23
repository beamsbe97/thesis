#!/bin/bash -l
#SBATCH --job-name=verb_from_ssl_fixed
#SBATCH --time=5:00:00
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

# Pipeline 3 stage B from a bug-fixed SSL checkpoint. Submit with
#   sbatch --export=ALL,SSL_CKPT=<ckpt_ssl/.../epoch_NNN.pth.tar>,OUT=<name>[,SEED=1][,SUP_CONFIG=configs/epic_videomae_verb.yaml] train_verb_from_ssl_fixed.slurm
: "${SSL_CKPT:?}" "${OUT:?}"
echo "stage B from $SSL_CKPT seed=${SEED:-default}"
python -u ./train.py ${SUP_CONFIG:-./configs/epic_slowfast_verb.yaml} $SEEDARG \
    --init-encoder $SSL_CKPT --output $OUT
