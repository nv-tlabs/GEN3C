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

import argparse
import os

import torch

from cosmos_predict1.autoregressive.configs.base.model_config import create_video2world_model_config
from cosmos_predict1.autoregressive.utils.checkpoint import merge_tensor_parallel_state_dicts
from cosmos_predict1.utils import log


def merge_sharded_checkpoints(checkpoint_path, output_path, tensor_parallel_size, model_size, model_family):
    assert checkpoint_path.endswith(".pt"), "Checkpoint path must end with .pt"
    assert model_family == "cosmos", "Only cosmos model family is currently supported"
    assert model_size == "4b", "Only 4B model size is currently supported"
    model_config, _ = create_video2world_model_config(
        model_ckpt_path=checkpoint_path,
        model_family=model_family,
        model_size=model_size,
        tensor_model_parallel_size=tensor_parallel_size,
        tokenizer_ckpt_path="checkpoints/Cosmos-Tokenize1-DV8x16x16-720p/ema.jit",
    )
    log.info(f"Merging sharded checkpoints from {checkpoint_path.replace('.pt', '_model_mp_*.pt')} into {output_path}")

    checkpoint_paths = [checkpoint_path.replace(".pt", f"_model_mp_{rank}.pt") for rank in range(tensor_parallel_size)]
    for path in checkpoint_paths:
        assert os.path.exists(path), f"Checkpoint path {path} does not exist"
        log.info(f"Found checkpoint {path}")
    sharded_state_dicts = [torch.load(path, map_location="cpu") for path in checkpoint_paths]
    merged_state_dict = merge_tensor_parallel_state_dicts(sharded_state_dicts, model_config)
    torch.save(merged_state_dict, output_path)
    log.info(f"Merged checkpoint saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Merge Cosmos-Predict1-4B autoregressive checkpoints")
    parser.add_argument(
        "--checkpoint_path",
        "-c",
        type=str,
        required=True,
        help="Path to the checkpoint to merge. Must end with .pt and be colocated with the sharded checkpoints ending in _model_mp_{rank}.pt",
    )
    parser.add_argument("--output_path", "-o", type=str, required=True, help="Path to the output merged checkpoint")
    parser.add_argument("--tensor_parallel_size", "-t", type=int, required=True, help="Tensor parallel size")
    parser.add_argument("--model_size", "-s", type=str, required=True, help="Model size")
    parser.add_argument("--model_family", "-f", type=str, required=False, default="cosmos", help="Model family")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_sharded_checkpoints(
        args.checkpoint_path, args.output_path, args.tensor_parallel_size, args.model_size, args.model_family
    )
