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
import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download

from scripts.download_guardrail_checkpoints import download_guardrail_checkpoints


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download NVIDIA Cosmos Predict1 autoregressive models from Hugging Face"
    )
    parser.add_argument(
        "--model_sizes",
        nargs="*",
        default=[
            "4B",
            "5B",
            "12B",
            "13B",
        ],  # Download all by default
        choices=["4B", "5B", "12B", "13B"],
        help="Which model sizes to download. Possible values: 4B, 5B, 12B, 13B.",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Directory to save the downloaded checkpoints."
    )
    args = parser.parse_args()
    return args


MD5_CHECKSUM_LOOKUP = {
    "Cosmos-Predict1-12B/guardrail/video_content_safety_filter/safety_filter.pt": "b46dc2ad821fc3b0d946549d7ade19cf",
    "Cosmos-Predict1-12B/model.pt": "ed748fabcb544d30d35385a8c28efb4d",
    "Cosmos-Predict1-13B-Video2World/guardrail/video_content_safety_filter/safety_filter.pt": "b46dc2ad821fc3b0d946549d7ade19cf",
    "Cosmos-Predict1-13B-Video2World/model.pt": "21a9fb02c61fbebc96c1af1fcaa5893f",
    "Cosmos-Predict1-4B/guardrail/video_content_safety_filter/safety_filter.pt": "b46dc2ad821fc3b0d946549d7ade19cf",
    "Cosmos-Predict1-4B/model.pt": "5fdc62fc87fbf470dbcc2288589b7942",
    "Cosmos-Predict1-5B-Video2World/guardrail/video_content_safety_filter/safety_filter.pt": "b46dc2ad821fc3b0d946549d7ade19cf",
    "Cosmos-Predict1-5B-Video2World/model.pt": "2a48a854bb6e04abb6b7c72979f1a69b",
    "Cosmos-Predict1-7B-Decoder-DV8x16x16ToCV8x8x8-720p/aux_vars.pt": "29e450d81839e82bb4bdbf12e43a74f1",
    "Cosmos-Predict1-7B-Decoder-DV8x16x16ToCV8x8x8-720p/model.pt": "a30149cc3730f3142b01fd374b6076f8",
    "Cosmos-Predict1-7B-Decoder-DV8x16x16ToCV8x8x8-720p/guardrail/video_content_safety_filter/safety_filter.pt": "b46dc2ad821fc3b0d946549d7ade19cf",
    "Cosmos-Tokenize1-CV8x8x8-720p/autoencoder.jit": "7f658580d5cf617ee1a1da85b1f51f0d",
    "Cosmos-Tokenize1-CV8x8x8-720p/decoder.jit": "ff21a63ed817ffdbe4b6841111ec79a8",
    "Cosmos-Tokenize1-CV8x8x8-720p/encoder.jit": "f5834d03645c379bc0f8ad14b9bc0299",
    "Cosmos-Tokenize1-CV8x8x8-720p/mean_std.pt": "f07680ad7eefae57d698778e2a0c7c96",
    "Cosmos-Tokenize1-CV8x8x8-720p/image_mean_std.pt": "9f19fd3312fc1198e4905ada02e68bce",
    "Cosmos-Tokenize1-DV8x16x16-720p/autoencoder.jit": "606b8585b637f06057725cbb67036ae6",
    "Cosmos-Tokenize1-DV8x16x16-720p/decoder.jit": "f0c8a9d992614a43e7ce24ebfc901e26",
    "Cosmos-Tokenize1-DV8x16x16-720p/encoder.jit": "95186b0410346a3f0cf250b76daec452",
    "google-t5/t5-11b/pytorch_model.bin": "f890878d8a162e0045a25196e27089a3",
    "google-t5/t5-11b/tf_model.h5": "e081fc8bd5de5a6a9540568241ab8973",
}


def get_md5_checksum(checkpoints_dir, model_name):
    print("---------------------")
    for key, value in MD5_CHECKSUM_LOOKUP.items():
        if key.startswith(model_name + "/"):
            print(f"Verifying checkpoint {key}...")
            file_path = checkpoints_dir.joinpath(key)
            # File must exist
            if not Path(file_path).exists():
                print(f"Checkpoint {key} does not exist.")
                return False
            # File must match give MD5 checksum
            with open(file_path, "rb") as f:
                file_md5 = hashlib.md5(f.read()).hexdigest()
            if file_md5 != value:
                print(f"MD5 checksum of checkpoint {key} does not match.")
                return False
    print(f"Model checkpoints for {model_name} exist with matched MD5 checksums.")
    return True


def main(args):
    ORG_NAME = "nvidia"

    # Mapping from size argument to Hugging Face repository name
    model_map = {
        "4B": "Cosmos-Predict1-4B",
        "5B": "Cosmos-Predict1-5B-Video2World",
        "12B": "Cosmos-Predict1-12B",
        "13B": "Cosmos-Predict1-13B-Video2World",
    }

    # Additional models that are always downloaded
    extra_models = [
        "Cosmos-Predict1-7B-Decoder-DV8x16x16ToCV8x8x8-720p",
        "Cosmos-Tokenize1-CV8x8x8-720p",
        "Cosmos-Tokenize1-DV8x16x16-720p",
        "google-t5/t5-11b",
    ]

    # Create local checkpoints folder
    checkpoints_dir = Path(args.checkpoint_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    download_kwargs = dict(
        allow_patterns=["README.md", "model.pt", "image_mean_std.pt", "mean_std.pt", "config.json", "*.jit"]
    )

    # Download the requested Autoregressive models
    for size in args.model_sizes:
        model_name = model_map[size]
        repo_id = f"{ORG_NAME}/{model_name}"
        local_dir = checkpoints_dir.joinpath(model_name)

        if not get_md5_checksum(checkpoints_dir, model_name):
            local_dir.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {repo_id} to {local_dir}...")
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                **download_kwargs,
            )

    # Download the always-included models
    for model_name in extra_models:
        if model_name == "google-t5/t5-11b":
            repo_id = model_name
        else:
            repo_id = f"{ORG_NAME}/{model_name}"
        local_dir = checkpoints_dir.joinpath(model_name)

        if not get_md5_checksum(checkpoints_dir, model_name):
            local_dir.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {repo_id} to {local_dir}...")
            # Download all files
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )

    download_guardrail_checkpoints(args.checkpoint_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
