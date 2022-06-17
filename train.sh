#!/bin/bash
#SBATCH --job-name=myleg_ppg_mpi
#SBATCH --time=2-23:59
#SBATCH --output=output/mpi_%j.out
#SBATCH --ntasks=51
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH --partition=regular
#SBATCH --signal=USR1@60

module purge
module load PyTorch/1.10.0-fosscuda-2020b

source /home/$USER/.envs/osim/bin/activate
export LD_LIBRARY_PATH=/home/$USER/.libs/opensim_dependencies/ipopt/lib:/home/$USER/.libs/opensim_dependencies/adol-c/lib64:$LD_LIBRARY_PATH

OMPI_MCA_opal_warn_on_missing_libcuda=0 srun python -m src.train_mpi -c configs/healthy.yml

deactivate
