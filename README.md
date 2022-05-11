# Locomotion on Uneven Terrain of Musculoskeletal Models using Phasic Policy Gradient 

## How to install

First, install dependencies. On Ubuntu 18.04, open terminal:

```
sudo apt install libxi-dev libxmu-dev liblapack-dev libadolc2 coinor-libipopt1v5 
```

Install OpenSim using conda:
```
conda create -n opensim -c vbotics opensim=4.3 python=3.7 numpy
conda activate opensim
git clone https://github.com/vbotics/rug-opensim-rl.git
cd rug-opensim-rl
git checkout tags/v3.0
pip install -e .
```

For non-Ubuntu distributions, you will have to find a way to build OpenSim-core. See [this](https://github.com/opensim-org/opensim-core) or below for more information.

After activating the virtual environment, install additional dependencies:
```
pip install python-dateutil pytz ray==1.12.0

git clone https://github.com/rug-my-leg/opensim-env.git
cd opensim-env/
pip install -e .
```

## Running

Before running, update your LD_LIBRARY_PATH. This needs to be done only when you open a new terminal:
```
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/adol-c/lib64/:$CONDA_PREFIX/ipopt/lib/:$LD_LIBRARY_PATH
```

To run a training process, run the following:
```
python -m src.ppg_impala --env-name healthy
```


## Building OpenSim from source on Peregrine
```
module load PyTorch/1.10.0-fosscuda-2020b CMake/3.20.1-GCCcore-10.2.0 Eigen/3.3.8-GCCcore-10.2.0 SWIG/4.0.2-GCCcore-10.2.0 OpenBLAS/0.3.12-GCC-10.2.0

mkdir /data/$USER/.libs
mkdir/data/$USER/.envs

python -m venv /data/$USER/.envs/osim
source /data/$USER/.envs/osim/bin/activate

mkdir software
mkdir software/opensim
cd software/opensim

git clone https://github.com/opensim-org/opensim-core.git

mkdir build_deps/
cd build_deps/

cmake ../opensim-core/dependencies/ -LAH \
      -DCMAKE_INSTALL_PREFIX=/data/$USER/.libs/opensim_dependencies \
      -DCMAKE_BUILD_TYPE=Release \
      -DSUPERBUILD_ezc3d=ON \
      -DOPENSIM_WITH_TROPTER=ON \
      -DOPENSIM_WITH_CASADI=ON

make -j8

cd ..
mkdir build/
cd build/

export docopt_DIR=/data/$USER/.libs/opensim_dependencies/docopt/lib64/cmake

cmake ../opensim-core -LAH \
      -DCMAKE_INSTALL_PREFIX=/data/$USER/.libs/opensim-core \
      -DCMAKE_BUILD_TYPE=Release \
      -DOPENSIM_DEPENDENCIES_DIR=/data/$USER/.libs/opensim_dependencies \
      -DOPENSIM_C3D_PARSER=ezc3d \
      -DBUILD_PYTHON_WRAPPING=ON \
      -DSWIG_DIR=/software/software/SWIG/4.0.2-GCCcore-10.2.0/share/swig \
      -DSWIG_EXECUTABLE=/software/software/SWIG/4.0.2-GCCcore-10.2.0/bin/swig \
      -DOPENSIM_INSTALL_UNIX_FHS=OFF \
      -DOPENSIM_DOXYGEN_USE_MATHJAX=OFF \
      -DOPENSIM_SIMBODY_DOXYGEN_LOCATION="https://simbody.github.io/simtk.org/api_docs/simbody/latest/" \
      -DCMAKE_CXX_FLAGS="-Wno-error"

make -j8

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/$USER/.libs/opensim_dependencies/simbody/lib
ctest --parallel 8 --output-on-failure

make -j8 install

export LD_LIBRARY_PATH=/data/$USER/.libs/opensim_dependencies/ipopt/lib:/data/$USER/.libs/opensim_dependencies/adol-c/lib64:$LD_LIBRARY_PATH

cd /data/$USER/.libs/opensim-core/sdk/Python/
pip install .
```