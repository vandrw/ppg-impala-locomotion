#!/bin/bash
#SBATCH --job-name=myleg_ppg_mpi
#SBATCH --time=12:00:00
#SBATCH --output=output/sweep_mpi_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=11
#SBATCH --mem=16GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH --partition=regular

module purge
module load PyTorch/1.10.0-fosscuda-2020b

source /data/$USER/.envs/osim/bin/activate
export LD_LIBRARY_PATH=/data/$USER/.libs/opensim_dependencies/ipopt/lib:/data/$USER/.libs/opensim_dependencies/adol-c/lib64:$LD_LIBRARY_PATH

while read -r line; do
    if ! [[ -f "$(dirname "$line")/agent.pth" ]]; then
        echo "Starting run using $line."
        mpirun --mca opal_warn_on_missing_libcuda 0 python -m src.sweep_mpi -c $line < /dev/null
    else
        echo "Config $line was already used for a sweep. Continuing..."
    fi
done < output/sweep/sweeps.info

deactivate