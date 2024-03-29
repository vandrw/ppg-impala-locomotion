#!/bin/bash
#SBATCH --job-name=myleg_ppg_mpi
#SBATCH --time=12:00:00
#SBATCH --output=output/sweep_mpi_%j.out
#SBATCH --ntasks=11
#SBATCH --partition=regular

module purge
module load PyTorch/1.10.0-fosscuda-2020b

source /home/$USER/.envs/osim/bin/activate
export LD_LIBRARY_PATH=/home/$USER/.libs/opensim_dependencies/ipopt/lib:/home/$USER/.libs/opensim_dependencies/adol-c/lib64:$LD_LIBRARY_PATH

while read -r line; do
    if ! [[ -f "$(dirname "$line")/agent.pth" ]]; then
        echo "Starting run using $line."
        mpirun --mca opal_warn_on_missing_libcuda 0 python -m src.sweep_mpi -c $line < /dev/null
    else
        echo "Config $line was already used for a sweep. Continuing..."
    fi
done < output/sweep/sweeps.info

deactivate
