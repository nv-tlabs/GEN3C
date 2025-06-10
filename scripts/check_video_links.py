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

import os
import re

import requests


def find_md_files(root="."):
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(".md"):
                yield os.path.join(dirpath, f)


def extract_video_urls(md_file):
    with open(md_file, "r", encoding="utf-8") as f:
        content = f.read()
    return re.findall(r'<video\s+src="([^"]+)"', content)


def check_video_url(url):
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            ),
            "Accept": "*/*",
        }
        response = requests.get(url, headers=headers, stream=True, timeout=10)
        content_type = response.headers.get("Content-Type", "")
        if response.status_code == 200 and "video" in content_type.lower():
            print(f"✅ Video OK: {url}")
            return True
        else:
            print(f"❌ Invalid video: {url} (Status: {response.status_code}, Type: {content_type})")
            return False
    except Exception as e:
        print(f"❌ Error checking {url}: {e}")
        return False


def main():
    all_passed = True
    for md_file in find_md_files():
        video_urls = extract_video_urls(md_file)
        for url in video_urls:
            if not check_video_url(url):
                all_passed = False
    if not all_passed:
        exit(1)


if __name__ == "__main__":
    main()
