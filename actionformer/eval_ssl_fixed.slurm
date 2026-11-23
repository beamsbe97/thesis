#!/bin/bash -l
#SBATCH --job-name=eval_ssl_fixed
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

# SSL encoder spliced onto the frozen supervised EMA heads (pipeline 2 eval). Submit with
#   sbatch --export=ALL,SSL_CKPT=<ckpt>[,EVAL_FLAGS=--no-ema] eval_ssl_fixed.slurm
: "${SSL_CKPT:?}"
echo "eval: encoder from $SSL_CKPT $EVAL_FLAGS, heads from SlowFast baseline (EMA)"
python -u ./eval_ssl.py ./configs/epic_slowfast_verb.yaml $SSL_CKPT $EVAL_FLAGS \
    --head-from ./ckpt/epic_slowfast_verb_reproduce/epoch_021.pth.tar
