#!/bin/bash
#SBATCH --job-name=myleg_ppg_mpi
#SBATCH --time=24:00:00
#SBATCH --output=healthy_mpi_%j.out
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=24
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH --partition=regular

module purge
module load PyTorch/1.10.0-fosscuda-2020b

source /data/$USER/.envs/osim/bin/activate
export LD_LIBRARY_PATH=/data/$USER/.libs/opensim_dependencies/ipopt/lib:/data/$USER/.libs/opensim_dependencies/adol-c/lib64:$LD_LIBRARY_PATH

# Change the number after -n to increase or lower the number of workers used.
mpirun -n 64 --mca opal_warn_on_missing_libcuda 0 python -m src.train_mpi -c configs/healthy.yml --run-name healthy_mpi

deactivate