#!/bin/bash -l
#SBATCH --job-name=knn_label
#SBATCH --time=1:30:00
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
# TAL neighbour-label test (tools/knn_label_agreement.py) on the bug-fixed encoders.
python -u tools/knn_label_agreement.py configs/epic_slowfast_verb.yaml \
    input=input random=random \
    ssl=./ckpt_ssl/ssl_epic_slowfast_verb_fixed_v2_s0/epoch_020.pth.tar \
    supervised=./ckpt/epic_slowfast_verb_reproduce/epoch_021.pth.tar \
    ssl_then_sup=./ckpt/epic_slowfast_verb_from_ssl_fixed_v2_s0/epoch_021.pth.tar \
    --out figs/knn_label_slowfast.json
python -u tools/knn_label_agreement.py configs/epic_videomae_verb.yaml \
    input=input random=random \
    ssl=./ckpt_ssl/ssl_epic_videomae_verb_fixed_v2_s0/epoch_020.pth.tar \
    supervised=./ckpt/epic_videomae_verb_reproduce/epoch_021.pth.tar \
    ssl_then_sup=./ckpt/epic_videomae_verb_from_ssl_fixed_v2_s0/epoch_021.pth.tar \
    --excl 32 --out figs/knn_label_videomae.json
