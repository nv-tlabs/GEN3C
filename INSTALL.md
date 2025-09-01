## Environment setup

Cosmos runs only on Linux systems. We have tested the installation with Ubuntu 24.04, 22.04, and 20.04.
Cosmos requires the Python version to be `3.10.x`. Please also make sure you have `conda` installed ([instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)).

### Inference

The below commands creates the `cosmos-predict1` conda environment and installs the dependencies for inference:
```bash
# Create the cosmos-predict1 conda environment.
conda env create --file cosmos-predict1.yaml
# Activate the cosmos-predict1 conda environment.
conda activate cosmos-predict1
# Install the dependencies.
pip install -r requirements.txt

# Install cosmos_predict1 for Nvidia's diffusers-like models
pip install -e .

# Patch Transformer engine linking issues in conda environments.
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.10
# Install Transformer engine.
pip install transformer-engine[pytorch]==1.12.0
# Install Apex for inference.
git clone https://github.com/NVIDIA/apex
# CUDA_HOME=$CONDA_PREFIX pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./apex
# CUDA_HOME=/usr/local/cuda-12.6 uv pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation ./apex --config-settings=build-args="--cpp_ext --cuda_ext"
CUDA_HOME=/usr/local/cuda-12.6 uv pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation ./apex --config-settings="--build-option=--cpp_ext" --config-settings="--build-option=--cuda_ext"

# Install MoGe for inference.
pip install git+https://github.com/microsoft/MoGe.git
```

* Alternatively, if you are more familiar with a containerized environment, you can build the dockerfile and run it to get an environment with all the packages pre-installed.
    This requires docker to be already present on your system with the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.

    ```bash
    docker build -f Dockerfile . -t nvcr.io/$USER/cosmos-predict1:latest
    ```

    Note: In case you encounter permission issues while mounting local files inside the docker, you can share the folders from your current directory to all users (including docker) using this helpful alias `alias share='sudo chown -R ${USER}:users $PWD && sudo chmod g+w $PWD'` before running the docker.


You can test the environment setup for inference with
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/test_environment.py
```

### Post-training


🛠️ *Under construction* 👷

Stay tuned!
