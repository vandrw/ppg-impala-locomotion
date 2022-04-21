# Locomotion on Uneven Terrain of Musculoskeletal Models using Phasic Policy Gradient 

## How to install

First, install dependencies. On Ubuntu 18.04, open terminal:

```
sudo apt install libxi-dev libxmu-dev liblapack-dev libadolc2 coinor-libipopt1v5 
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

For non-Ubuntu distributions, you will have to find a way to build OpenSim-core. See [this](https://github.com/opensim-org/opensim-core) or below for more information.
```
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/adol-c/lib64/:$CONDA_PREFIX/ipopt/lib/:$LD_LIBRARY_PATH
```

## Running

To run a training process, run the following:
```
python -m src.ppg_impala --env-name healthy
```


## Building OpenSim from source on Peregrine
```
module load PyTorch/1.10.0-fosscuda-2020b CMake/3.20.1-GCCcore-10.2.0 Eigen/3.3.8-GCCcore-10.2.0 SWIG/4.0.2-GCCcore-10.2.0 OpenBLAS/0.3.12-GCC-10.2.0

mkdir software
mkdir software/opensim
cd software/opensim

git clone https://github.com/opensim-org/opensim-core.git

mkdir build_deps/
cd build_deps/

cmake ../opensim-core/dependencies/ -LAH \
      -DCMAKE_INSTALL_PREFIX=~/software/opensim/opensim_dependencies_install \
      -DCMAKE_BUILD_TYPE=Release \
      -DSUPERBUILD_ezc3d=ON \
      -DOPENSIM_WITH_TROPTER=ON \
      -DOPENSIM_WITH_CASADI=ON

make -j8

cd ..
mkdir build/
cd build/

export docopt_DIR=/home/$USER/software/opensim/opensim_dependencies_install/docopt/lib64/cmake

cmake ../opensim-core -LAH \
      -DCMAKE_INSTALL_PREFIX=~/software/opensim/opensim-core-install \
      -DCMAKE_BUILD_TYPE=Release \
      -DOPENSIM_DEPENDENCIES_DIR=~/software/opensim/opensim_dependencies_install \
      -DOPENSIM_C3D_PARSER=ezc3d \
      -DBUILD_PYTHON_WRAPPING=ON \
      -DSWIG_DIR=/software/software/SWIG/4.0.2-GCCcore-10.2.0/share/swig \
      -DSWIG_EXECUTABLE=/software/software/SWIG/4.0.2-GCCcore-10.2.0/bin/swig \
      -DOPENSIM_INSTALL_UNIX_FHS=OFF \
      -DOPENSIM_DOXYGEN_USE_MATHJAX=off \
      -DOPENSIM_SIMBODY_DOXYGEN_LOCATION="https://simbody.github.io/simtk.org/api_docs/simbody/latest/" \
      -DCMAKE_CXX_FLAGS="-Werror"

make -j8
ctest --parallel 4 --output-on-failure

# make doxygen
make --j8 install
```