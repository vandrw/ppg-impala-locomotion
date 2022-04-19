# Locomotion on Uneven Terrain of Musculoskeletal Models using Phasic Policy Gradient 

## How to install

First, install dependencies. On Ubuntu 18.04, open terminal:

```
sudo apt install libxi-dev libxmu-dev liblapack-dev libadolc2 coinor-libipopt1v5 
```

For Arch-based distributions:

```
sudo pacman -S gcc gcc-fortran
conda install -c conda-forge libgfortran4

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/adol-c/lib64/:$CONDA_PREFIX/ipopt/lib/:$LD_LIBRARY_PATH
```

Install OpenSim using conda:
```
conda create -n opensim -c vbotics opensim=4.2 python=3.7 numpy
conda activate opensim
git clone https://github.com/vbotics/rug-opensim-rl.git
cd rug-opensim-rl
git checkout tags/v3.0
pip install -e .
```

Install PPG:
```
git clone https://github.com/openai/phasic-policy-gradient.git
conda env update --file phasic-policy-gradient/environment.yml
pip install -e phasic-policy-gradient
```

To run a training process, run the following:
```
python3 -m src.ppg.train --env-name healthy
```