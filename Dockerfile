# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Use NVIDIA PyTorch container as base image
FROM nvcr.io/nvidia/pytorch:24.10-py3

# Install basic tools
RUN apt-get update && apt-get install -y git tree ffmpeg wget
RUN rm /bin/sh && ln -s /bin/bash /bin/sh && ln -s /lib64/libcuda.so.1 /lib64/libcuda.so

# Copy the cosmos-predict1.yaml and requirements.txt files to the container
COPY ./cosmos-predict1.yaml /cosmos-predict1.yaml
COPY ./requirements.txt /requirements.txt

# Install cosmos-predict1 dependencies. This will take a while.
RUN echo "Installing dependencies. This will take a while..." && \
    mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
    rm ~/miniconda3/miniconda.sh && \
    source ~/miniconda3/bin/activate && \
    conda env create --file /cosmos-predict1.yaml && \
    conda activate cosmos-predict1 && \
    pip install --no-cache-dir -r /requirements.txt && \
    ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/ && \
    ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.10 && \
    ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/triton/backends/nvidia/include/* $CONDA_PREFIX/include/ && \
    pip install transformer-engine[pytorch]==1.12.0 && \
    git clone https://github.com/NVIDIA/apex && cd apex && \
    CUDA_HOME=$CONDA_PREFIX pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" . && \
    pip install git+https://github.com/microsoft/MoGe.git && \
    echo "Environment setup complete"


# Default command
CMD ["/bin/bash"]
