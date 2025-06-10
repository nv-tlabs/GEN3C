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
import sys

import termcolor

parser = argparse.ArgumentParser(description="Cosmos IP header checker/fixer")
parser.add_argument("--fix", action="store_true", help="apply the fixes instead of checking")
args, files_to_check = parser.parse_known_args()


def get_header(ext: str = "py", old: str | bool = False) -> list[str]:
    header = [
        "SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "SPDX-License-Identifier: Apache-2.0",
        "",
        'Licensed under the Apache License, Version 2.0 (the "License");',
        "you may not use this file except in compliance with the License.",
        "You may obtain a copy of the License at",
        "",
        "http://www.apache.org/licenses/LICENSE-2.0",
        "",
        "Unless required by applicable law or agreed to in writing, software",
        'distributed under the License is distributed on an "AS IS" BASIS,',
        "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
        "See the License for the specific language governing permissions and",
        "limitations under the License.",
    ]
    if ext == ".py" and old:
        if old == "single":
            header = ["'''"] + header + ["'''"]
        elif old == "double":
            header = ['"""'] + header + ['"""']
        else:
            raise NotImplementedError
    elif ext in (".py", ".yaml"):
        header = [("# " + line if line else "#") for line in header]
    elif ext in (".c", ".cpp", ".cu", ".h", ".cuh"):
        header = ["/*"] + [(" * " + line if line else " *") for line in header] + [" */"]
    else:
        raise NotImplementedError
    return header


def apply_file(file: str, results: dict[str, int], fix: bool = False) -> None:
    if file.endswith("__init__.py"):
        return
    ext = os.path.splitext(file)[1]
    content = open(file).read().splitlines()
    header = get_header(ext=ext)
    if fix:
        if _check_header(content, header):
            return
        print(f"fixing: {file}")
        while len(content) > 0 and not content[0]:
            content.pop(0)
        content = header + [""] + content
        with open(file, "w") as file_obj:
            for line in content:
                file_obj.write(line + "\n")
    else:
        if not _check_header(content, header):
            bad_header = colorize("BAD HEADER", color="red", bold=True)
            print(f"{bad_header}: {file}")
            results[file] = 1
        else:
            results[file] = 0


def traverse_directory(path: str, results: dict[str, int], fix: bool = False, substrings_to_skip=[]) -> None:
    files = os.listdir(path)
    for file in files:
        full_path = os.path.join(path, file)
        if os.path.isdir(full_path):
            traverse_directory(full_path, results, fix=fix, substrings_to_skip=substrings_to_skip)
        elif os.path.isfile(full_path):
            ext = os.path.splitext(file)[1]
            to_skip = any(substr in full_path for substr in substrings_to_skip)
            if not to_skip and ext in (".py", ".yaml", ".c", ".cpp", ".cu", ".h", ".cuh"):
                apply_file(full_path, results, fix=fix)
        else:
            raise NotImplementedError


def _check_header(content: list[str], header: list[str]) -> bool:
    if content[: len(header)] != header:
        return False

    i = len(header)
    blank_line_count = 0

    while i < len(content) and content[i].strip() == "":
        blank_line_count += 1
        i += 1

    # Allow at most two blank lines
    if blank_line_count > 2:
        return False

    # Must have at least one non-empty line after the blank lines
    return i < len(content)


def colorize(x: str, color: str, bold: bool = False) -> str:
    return termcolor.colored(str(x), color=color, attrs=("bold",) if bold else None)  # type: ignore


if __name__ == "__main__":
    if not files_to_check:
        files_to_check = [
            "cosmos_predict1/auxiliary",
            "cosmos_predict1/diffusion",
            "cosmos_predict1/callbacks",
            "cosmos_predict1/checkpointer",
            "cosmos_predict1/autoregressive",
            "cosmos_predict1/tokenizer",
            "cosmos_predict1/utils",
        ]

    for file in files_to_check:
        assert os.path.isfile(file) or os.path.isdir(file), f"{file} is neither a directory or a file!"

    substrings_to_skip = ["prompt_upsampler"]
    results = dict()
    for file in files_to_check:
        if os.path.isfile(file):
            apply_file(file, results, fix=args.fix)
        elif os.path.isdir(file):
            traverse_directory(file, results, fix=args.fix, substrings_to_skip=substrings_to_skip)
        else:
            raise NotImplementedError

    if any(results.values()):
        sys.exit(1)
