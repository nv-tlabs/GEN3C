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
import json
import os

import ffmpeg
from pytubefix import YouTube

"""example command
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_tokenizer_example_data.py --dataset_path datasets/hdvila --N_videos 128 --do_download --do_clip
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download example (hdvila) data for posttraining")
    parser.add_argument("--dataset_path", type=str, default="datasets/hdvila", help="Root path to the dataset")
    parser.add_argument("--N_videos", type=int, default=128, help="Number of videos to download")
    parser.add_argument("--do_download", action="store_true", help="Download the videos")
    parser.add_argument("--do_clip", action="store_true", help="Clip the videos")
    return parser.parse_args()


def convert_time_to_seconds(time_str) -> int:
    h, m, s = map(float, time_str.split(":"))
    ms = int(time_str.split(".")[-1]) if "." in time_str else 0
    return int(h * 3600 + m * 60 + s) + ms / 1000


def download_data(args) -> None:
    urls_set = set()
    download_count = 0

    videos_orig_dir = os.path.join(args.dataset_path, "videos_orig")
    os.makedirs(videos_orig_dir, exist_ok=True)
    videos_dir = os.path.join(args.dataset_path, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    metas_dir = os.path.join(args.dataset_path, "metas")
    os.makedirs(metas_dir, exist_ok=True)

    hdvila_jsonl_path = os.path.join(args.dataset_path, "hdvila-100M.jsonl")
    with open(hdvila_jsonl_path, "r") as fp:
        for line in fp:
            json_object = json.loads(line)
            url = json_object["url"]
            if url not in urls_set:  # download videos with unique urls
                yt = YouTube(json_object["url"])
                try:
                    # Download a video
                    yt.streams.get_highest_resolution().download(
                        output_path=videos_orig_dir, filename=json_object["video_id"] + ".mp4"
                    )
                    download_count += 1
                    urls_set.add(url)
                    print(f"Downloaded videos: {download_count}/{args.N_videos}")

                    # Save metadata - caption and whole metadata
                    meta_txt_name = os.path.join(metas_dir, json_object["clip_id"].replace(".mp4", ".txt"))
                    with open(meta_txt_name, "w") as fp:
                        fp.write(json_object["caption"])
                    meta_json_name = os.path.join(metas_dir, json_object["clip_id"].replace(".mp4", ".json"))
                    with open(meta_json_name, "w") as fp:
                        json.dump(json_object, fp)
                except Exception as e:
                    print(e)
                    continue

            if len(urls_set) >= args.N_videos:
                break


def clip_data(args) -> None:
    videos_orig_dir = os.path.join(args.dataset_path, "videos_orig")
    videos_dir = os.path.join(args.dataset_path, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    metas_dir = os.path.join(args.dataset_path, "metas")

    metas_list = [
        os.path.join(metas_dir, filename) for filename in sorted(os.listdir(metas_dir)) if filename.endswith(".json")
    ]
    videos_orig_list = [
        os.path.join(videos_orig_dir, filename)
        for filename in sorted(os.listdir(videos_orig_dir))
        if filename.endswith(".mp4")
    ]

    for meta_filename, video_orig_filename in zip(metas_list, videos_orig_list):
        with open(meta_filename, "r") as fp:
            metadata = json.load(fp)

        # Convert time strings to seconds
        start_time = convert_time_to_seconds(metadata["span_start"])
        end_time = convert_time_to_seconds(metadata["span_end"])
        # Clip the video
        clip_name = os.path.join(videos_dir, metadata["clip_id"])
        ffmpeg.input(video_orig_filename, ss=start_time, t=end_time - start_time).output(clip_name).run()


def main(args) -> None:
    if args.do_download:
        download_data(args)
    if args.do_clip:
        clip_data(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
