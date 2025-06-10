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
import pickle
from typing import Tuple

import numpy as np
import torch
from transformers import T5EncoderModel, T5TokenizerFast

"""example command
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/get_t5_embeddings_from_waymo.py --dataset_path datasets/waymo
"""

PREFIX_PROMPTS = {
    "pinhole_front": "The video is captured from a camera mounted on a car. The camera is facing forward.",
    "pinhole_front_left": "The video is captured from a camera mounted on a car. The camera is facing forward and slightly to the left.",
    "pinhole_front_right": "The video is captured from a camera mounted on a car. The camera is facing forward and slightly to the right.",
    "pinhole_side_left": "The video is captured from a camera mounted on a car. The camera is facing to the left.",
    "pinhole_side_right": "The video is captured from a camera mounted on a car. The camera is facing to the right.",
}


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute T5 embeddings for text prompts")
    parser.add_argument("--dataset_path", type=str, default="datasets/waymo", help="Root path to the dataset")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length of the text embedding")
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, default="google-t5/t5-11b", help="T5 model name or the local path"
    )
    parser.add_argument("--prompt", type=str, default="A video of sks teal robot.", help="Text prompt for the dataset")
    parser.add_argument("--cache_dir", type=str, default="checkpoints", help="Directory to cache the T5 model")
    return parser.parse_args()


def init_t5(
    pretrained_model_name_or_path: str = "google-t5/t5-11b", max_length: int = 512, cache_dir: str = "~/.cache"
) -> Tuple[T5TokenizerFast, T5EncoderModel]:
    """Initialize and return the T5 tokenizer and text encoder."""
    tokenizer = T5TokenizerFast.from_pretrained(
        pretrained_model_name_or_path, model_max_length=max_length, cache_dir=cache_dir
    )
    text_encoder = T5EncoderModel.from_pretrained(pretrained_model_name_or_path, cache_dir=cache_dir)
    text_encoder.to("cuda")
    text_encoder.eval()
    return tokenizer, text_encoder


@torch.inference_mode()
def encode_for_batch(tokenizer, encoder, prompts: list[str], max_length=512) -> list:
    """
    Encode a batch of text prompts to a batch of T5 embeddings.
    Parameters:
        tokenizer: T5 embedding tokenizer.
        encoder: T5 embedding text encoder.
        prompts: A batch of text prompts.
        max_length: Sequence length of text embedding (defaults to 512).
    """

    batch_encoding = tokenizer.batch_encode_plus(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_length=True,
        return_offsets_mapping=False,
    )

    # We expect all the processing is done on GPU.
    input_ids = batch_encoding.input_ids.cuda()
    attn_mask = batch_encoding.attention_mask.cuda()

    outputs = encoder(input_ids=input_ids, attention_mask=attn_mask)
    encoded_text = outputs.last_hidden_state

    lengths = attn_mask.sum(dim=1).cpu()
    for batch_id in range(encoded_text.shape[0]):
        encoded_text[batch_id][lengths[batch_id] :] = 0

    encoded_text = encoded_text.cpu().numpy().astype(np.float16)
    encoded_text = encoded_text[:, :max_length]

    # trim zeros to save space
    encoded_text = [encoded_text[batch_id][: lengths[batch_id]] for batch_id in range(encoded_text.shape[0])]

    return encoded_text


def main(args) -> None:
    videos_dir = os.path.join(args.dataset_path, "videos")

    metas_dir = os.path.join(args.dataset_path, "metas")
    os.makedirs(metas_dir, exist_ok=True)
    metas_list = [
        os.path.join(metas_dir, viewname, filename.replace(".mp4", ".txt"))
        for viewname in sorted(os.listdir(videos_dir))
        for filename in sorted(os.listdir(videos_dir + "/" + viewname))
        if filename.endswith(".mp4")
    ]

    # Write txt files to match other dataset formats.
    for meta_filename in metas_list:
        if not os.path.exists(meta_filename):
            with open(meta_filename, "w") as fp:
                fp.write(args.prompt)

    t5_xxl_dir = os.path.join(args.dataset_path, "t5_xxl")
    os.makedirs(t5_xxl_dir, exist_ok=True)

    # Initialize T5
    tokenizer, text_encoder = init_t5(cache_dir=args.cache_dir)

    # Extract T5 embeddings for prefix prompt
    for view_name, prefix_prompt in PREFIX_PROMPTS.items():
        t5_xxl_filename = os.path.join(args.dataset_path, "cache", f"prefix_t5_embeddings_{view_name}.pickle")
        os.makedirs(os.path.dirname(t5_xxl_filename), exist_ok=True)
        if os.path.exists(t5_xxl_filename):
            # Skip if the file already exists
            continue

        # Compute T5 embeddings
        encoded_text = encode_for_batch(tokenizer, text_encoder, [prefix_prompt])

        # Save T5 embeddings as pickle file
        with open(t5_xxl_filename, "wb") as fp:
            pickle.dump(encoded_text, fp)

    for meta_filename in metas_list:
        t5_xxl_filename = os.path.join(
            t5_xxl_dir, meta_filename.split("/")[-2], os.path.basename(meta_filename).replace(".txt", ".pickle")
        )
        os.makedirs(os.path.dirname(t5_xxl_filename), exist_ok=True)
        if os.path.exists(t5_xxl_filename):
            # Skip if the file already exists
            continue

        with open(meta_filename, "r") as fp:
            prompt = fp.read().strip()

        # Compute T5 embeddings
        encoded_text = encode_for_batch(tokenizer, text_encoder, [prompt])

        # Save T5 embeddings as pickle file
        with open(t5_xxl_filename, "wb") as fp:
            pickle.dump(encoded_text, fp)


if __name__ == "__main__":
    args = parse_args()
    main(args)
