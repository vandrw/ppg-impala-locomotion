#!/bin/bash
#SBATCH --job-name=myleg_ppg
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1

module purge
module load PyTorch/1.10.0-fosscuda-2020b CMake/3.20.1-GCCcore-10.2.0 Eigen/3.3.8-GCCcore-10.2.0 SWIG/4.0.2-GCCcore-10.2.0 OpenBLAS/0.3.12-GCC-10.2.0

conda activate opensim

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/adol-c/lib64/:$CONDA_PREFIX/ipopt/lib/:$LD_LIBRARY_PATH

cd rug-locomotion-ppg

python -m src.ppg_impala

conda deactivate