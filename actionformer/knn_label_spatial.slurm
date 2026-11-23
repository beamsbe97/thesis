#!/bin/bash -l
#SBATCH --job-name=knn_label_spatial
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
# neighbour-label + kitchen probe for the bug-fixed SPATIAL-crop VideoMAE SSL encoder
python -u tools/knn_label_agreement.py configs/epic_videomae_verb.yaml \
    ssl_spatial=./ckpt_ssl/ssl_epic_videomae_verb_spatial_fixed_v2_s0/epoch_020.pth.tar \
    --excl 32 --out figs/knn_label_videomae_spatial.json
