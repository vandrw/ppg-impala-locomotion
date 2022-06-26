# Locomotion on Uneven Terrain of Musculoskeletal Models using Phasic Policy Gradient 

## How to install

First, install dependencies. On Ubuntu 18.04, open terminal:

```
sudo apt install libxi-dev libxmu-dev liblapack-dev libadolc2 coinor-libipopt1v5 
```

Install OpenSim using conda:
```
conda create -n osim -c vbotics opensim=4.3 python=3.7 numpy
conda activate osim
git clone https://github.com/vbotics/rug-opensim-rl.git
cd rug-opensim-rl
git checkout tags/v3.0
pip install -e .

# For the visualizer, you will also need this.
conda install -c conda-forge libstdcxx-ng
```

For non-Ubuntu distributions, you will have to find a way to build OpenSim-core. See [this](https://github.com/opensim-org/opensim-core) or below for more information.

## Dependencies

After activating the virtual environment, install additional dependencies. Run the following in a local library folder:
```
# Gym 0.24.0/1 worked well for me.
pip install python-dateutil pytz wandb gym

git clone https://github.com/rug-my-leg/opensim-env.git
cd opensim-env/
pip install -e .
```

## Running

Before running, update your LD_LIBRARY_PATH. This needs to be done only when you open a new terminal:
```
# For Conda environments
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/adol-c/lib64/:$CONDA_PREFIX/ipopt/lib/:$LD_LIBRARY_PATH
```

```
# For environments built from source. Make sure you replace /home/$USER with the correct path to the libraries.
export LD_LIBRARY_PATH=/home/$USER/.libs/opensim_dependencies/ipopt/lib:/home/$USER/.libs/opensim_dependencies/adol-c/lib64:$LD_LIBRARY_PATH
```

To use MPI, run the following:
```
# Add the argument '-n' to use a specific amount of workers. If running with SLURM, disregard this argument.
mpirun python -m src.train_mpi -c configs/healthy.yml
```

If you encounter issues when running the MPI version, try changing `start_method="fork"` from `src/ppg/logging.py` to `start_method="thread"`.

## Hyperparameter Search
To find a good set of hyperparameters for the model, `src/sweep_mpi.py` makes use of the `wandb` sweep function. Currently this is only implemented for the MPI version.

To begin a hyperparameter sweep, first initialize the configurations using the command below. This will create a number of new folders under `output/sweep` that contain a config file, as well as a `sweeps.info` file that contains the path of these configs. Before running it, make sure you create a sweep in your project on wandb and provide the given ID to the `src/utils/init_sweep.py` script. You can also adjust the number of generated configurations by providing the `runs` parameter.
```
wandb init
python -m src.utils.init_sweep --id <SWEEP_ID> --runs 25
```

Finally, to train all the models, one can use the script mentioned below. If you're running the project on Peregrine, simply submit `run_sweep.sh` to the job handler.
```
#!/bin/bash

while read -r line; do
    if ! [[ -f "$(dirname "$line")/agent.pth" ]]; then
        echo $line
        mpirun --mca opal_warn_on_missing_libcuda 0 python -m src.sweep_mpi -c $line < /dev/null
    else
        echo "Config $line was already used for a sweep. Continuing..."
    fi
done < output/sweep/sweeps.info
```

## Visualizing a model
If you are running on your own machine and have a graphical user interface available, you can use the command below to visualize your trained model. This will instantiate the Python OpenSim visualizer.
```
python -m src.utils.visualize_model output/example_run_name
```

However, if you are running the code on the cluster or you do not have a graphical user interface, you can run the command below to generate a motion file. This file can then be imported in OpenSim, along with the model you used. Please take a look first at the command help to see what it can do.
```
python -m src.utils.generate_motion output/example_run_name
```

If you do not have access to the OpenSim visualizer, you can also generate a `.csv` file and visualize it using the playback script.
```
python -m src.utils.generate_motion output/example_run_name --csv
python -m src.utils.playback output/example_run_name/episode.csv --speed 0.2

```

# Building OpenSim from source on Peregrine
If you already have access to the Peregrine cluster, you can run the commands below on one of the nodes to have access to the latest OpenSim version. Try to perform them one by one, rather than running the following as a script.

```
module load PyTorch/1.10.0-fosscuda-2020b CMake/3.20.1-GCCcore-10.2.0 Eigen/3.3.8-GCCcore-10.2.0 SWIG/4.0.2-GCCcore-10.2.0 OpenBLAS/0.3.12-GCC-10.2.0

mkdir /home/$USER/.libs
mkdir /home/$USER/.envs

python -m venv /home/$USER/.envs/osim
source /home/$USER/.envs/osim/bin/activate
pip install --upgrade pip
pip install --upgrade wheel

mkdir software
mkdir software/opensim
cd software/opensim

git clone https://github.com/opensim-org/opensim-core.git

mkdir build_deps/
cd build_deps/

cmake ../opensim-core/dependencies/ -LAH \
      -DCMAKE_INSTALL_PREFIX=/home/$USER/.libs/opensim_dependencies \
      -DCMAKE_BUILD_TYPE=Release \
      -DSUPERBUILD_ezc3d=ON \
      -DOPENSIM_WITH_TROPTER=ON \
      -DOPENSIM_WITH_CASADI=ON

make -j8

cd ..
mkdir build/
cd build/

export docopt_DIR=/home/$USER/.libs/opensim_dependencies/docopt/lib64/cmake
export ezc3d_DIR=/home/$USER/.libs/opensim_dependencies/ezc3d/lib64/cmake

cmake ../opensim-core -LAH \
      -DCMAKE_INSTALL_PREFIX=/home/$USER/.libs/opensim-core \
      -DCMAKE_BUILD_TYPE=Release \
      -DOPENSIM_DEPENDENCIES_DIR=/home/$USER/.libs/opensim_dependencies \
      -DOPENSIM_C3D_PARSER=ezc3d \
      -DBUILD_PYTHON_WRAPPING=ON \
      -DSWIG_DIR=/software/software/SWIG/4.0.2-GCCcore-10.2.0/share/swig \
      -DSWIG_EXECUTABLE=/software/software/SWIG/4.0.2-GCCcore-10.2.0/bin/swig \
      -DOPENSIM_INSTALL_UNIX_FHS=OFF \
      -DOPENSIM_DOXYGEN_USE_MATHJAX=OFF \
      -DOPENSIM_SIMBODY_DOXYGEN_LOCATION="https://simbody.github.io/simtk.org/api_docs/simbody/latest/" \
      -DCMAKE_CXX_FLAGS="-Wno-error"

make -j8

# Ignore the python_example failure
# If other errors occur, run ctest using the --rerun-failed flag
# If the errors persist after rerunning the tests, try making the build again.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$USER/.libs/opensim_dependencies/simbody/lib
ctest --parallel 8 --output-on-failure

# Ignore issues with doxygen
make -j8 install

export LD_LIBRARY_PATH=/home/$USER/.libs/opensim_dependencies/ipopt/lib:/home/$USER/.libs/opensim_dependencies/adol-c/lib64:$LD_LIBRARY_PATH

cd /home/$USER/.libs/opensim-core/sdk/Python/
pip install .

# We do not need the build files anymore.
rm -rvf /home/$USER/software
```

You can now continue installing the other dependencies mentioned above.