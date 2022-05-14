#!/bin/bash
#SBATCH --job-name=myleg_ppg_ray
#SBATCH --output=output/healthy_ray_%j.out
#SBATCH --time=192:00:00
#SBATCH --nodes=1
#SBATCH --mem=8GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH --partition=regular

module purge
module load PyTorch/1.10.0-fosscuda-2020b

source /data/$USER/.envs/osim/bin/activate
export LD_LIBRARY_PATH=/data/$USER/.libs/opensim_dependencies/ipopt/lib:/data/$USER/.libs/opensim_dependencies/adol-c/lib64:$LD_LIBRARY_PATH

python -m src.train_ppg_impala -c configs/healthy.yml --run-name test_test

deactivate