#!/bin/bash
#SBATCH --job-name=myleg_ppg
#SBATCH --time=192:00:00
#SBATCH --nodes=2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH --partition=regular
#SBATCH --gres=gpu:1

module purge
module load PyTorch/1.10.0-fosscuda-2020b

source /data/$USER/.envs/osim/bin/activate
export LD_LIBRARY_PATH=/data/$USER/.libs/opensim_dependencies/ipopt/lib:/data/$USER/.libs/opensim_dependencies/adol-c/lib64:$LD_LIBRARY_PATH

cd rug-locomotion-ppg

python -m src.train_ppg_impala -c configs/default.yml --run-name test_healthy_ppg

conda deactivate