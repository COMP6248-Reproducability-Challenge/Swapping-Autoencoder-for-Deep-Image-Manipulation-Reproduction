#!/bin/bash

#SBATCH --partition=ecsall
#SBATCH --gpus-per-node=4
#SBATCH --output=/ECShome/cn2g18/6248/slurm-out/%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cn2g18@soton.ac.uk
#SBATCH --time=4-23:59

#export TORCH_CUDA_ARCH_LIST="7.5"

module unload python
module load conda/py3-latest
module load gcc/8.5.0
module load cuda/11.1
cd $HOME/6248/Swapping-Autoencoder-for-Deep-Image-Manipulation-Reproduction

echo "==================Copy of Running script========================="
cat run.slurm
echo "================================================================"

mkdir /ECSssd/data-sets
cp -r ../CwkData/lsun/* /ECSssd/data-sets

eval "$(conda shell.bash hook)"
conda activate 6248_39
python --version
echo "running training.py"
python training.py 20550
