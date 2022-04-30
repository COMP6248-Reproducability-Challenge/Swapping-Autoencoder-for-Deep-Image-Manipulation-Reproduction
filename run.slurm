#!/bin/bash

#SBATCH --partition=ecsstudents
#SBATCH --nodes=1
#SBATCH --tasks=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64gb
#SBATCH --output=/ECShome/cn2g18/6248/slurm-out/%j.out

#export TORCH_CUDA_ARCH_LIST="7.5"

module unload python
module load conda/py3-latest
module load gcc/8.5.0
module load cuda/11.1
cd $HOME/6248/Swapping-Autoencoder-for-Deep-Image-Manipulation-Reproduction

echo "==================Copy of Running script========================="
cat run.slurm
echo "================================================================"

eval "$(conda shell.bash hook)"
conda activate 6248
python --version
echo "running training.py"
python training.py
