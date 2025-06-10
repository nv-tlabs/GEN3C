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

import asyncio
from io import BytesIO
import pickle
import time
from typing import Callable

import httpx
from tqdm import tqdm


def content_with_progress(content, chunk_size=1024, desc="Upload",
						  progress_callback: Callable[[str, float, tqdm], None] | None = None):
	total = len(content)
	with tqdm(total=total, unit_scale=True, unit_divisor=1024, unit="B", desc=desc) as progress:
		for i in range(0, total, chunk_size):
			chunk = content[i:i + chunk_size]
			yield chunk
			report_progress("upload", len(chunk), progress, callback=progress_callback)

async def async_content_with_progress(*args, **kwargs):
	for chunk in content_with_progress(*args, **kwargs):
		yield chunk


def streaming_response_to_response(response: httpx.Response, content_bytes: BytesIO) -> httpx.Response:
	"""
	Convert a streaming response to a non-streaming response.
	"""
	# TODO: is there a nicer way to get a non-streaming-style Response object, despite
	# having used the streaming API above? (for uniform consumption by the caller).
	to_remove = set(["is_stream_consumed", "next_request", "is_closed", "content", "stream"] + [
		k for k in response.__dict__ if k.startswith("_")
	])
	kwargs = { k: v for k, v in response.__dict__.items() if k not in to_remove }

	content_bytes.seek(0)
	kwargs["content"] = content_bytes.read()
	return httpx.Response(**kwargs)


def report_progress(direction: str, progress_absolute: int | float,
					bar: tqdm, callback: Callable[[str, float, tqdm], None] | None = None):
	bar.update(progress_absolute)
	if callback is not None:
		progress_percent = bar.n / bar.total
		callback(direction=direction, progress=progress_percent, bar=bar)



def httpx_request(method: str,
				  *args,
				  progress: bool = False,
				  progress_direction: str = "auto",
				  desc: str | None = None,
				  async_client: httpx.AsyncClient | None = None,
				  callback: Callable | None = None,
				  progress_callback: Callable[[str, float, tqdm], None] | None = None,
				  **kwargs) -> httpx.Response | asyncio.Task[httpx.Response]:
	is_async = async_client is not None

	progress_download = progress and (
		progress_direction in ("both", "download")
		or (progress_direction == "auto" and method.lower() == "get")
	)
	progress_upload = progress and (
		progress_direction in ("both", "upload")
		or (progress_direction == "auto" and method.lower() == "post")
	)

	if progress_upload:
		for key in ("content", "data"):
			if key in kwargs:
				upload_desc = f"{desc} (upload)" if desc else "Upload"
				wrapper = async_content_with_progress if is_async else content_with_progress
				kwargs[key] = wrapper(kwargs[key], desc=upload_desc, progress_callback=progress_callback)

	if progress_download:
		# Progress bar requested for download, need to use streaming API

		if async_client is None:
			content_bytes = BytesIO()
			with httpx.stream(method, *args, **kwargs) as response:
				total = int(response.headers["Content-Length"])
				with tqdm(total=total, unit_scale=True, unit_divisor=1024, unit="B", desc=desc) as progress:
					num_bytes_downloaded = response.num_bytes_downloaded
					for chunk in response.iter_bytes():
						report_progress("download", response.num_bytes_downloaded - num_bytes_downloaded,
										progress, callback=progress_callback)

						num_bytes_downloaded = response.num_bytes_downloaded
						content_bytes.write(chunk)
			response = streaming_response_to_response(response, content_bytes)
			if callback is not None:
				callback(response)
			return response

		else:
			async def inner():
				content_bytes = BytesIO()
				async with async_client.stream(method, *args, **kwargs) as response:
					total = int(response.headers["Content-Length"])
					with tqdm(total=total, unit_scale=True, unit_divisor=1024, unit="B", desc=desc) as progress:
						num_bytes_downloaded = response.num_bytes_downloaded
						async for chunk in response.aiter_bytes():
							report_progress("download", response.num_bytes_downloaded - num_bytes_downloaded,
											progress, callback=progress_callback)
							num_bytes_downloaded = response.num_bytes_downloaded
							content_bytes.write(chunk)
				response = streaming_response_to_response(response, content_bytes)
				return response

			task = asyncio.create_task(inner())
			if callback is not None:
				task.add_done_callback(callback)
			return task

	else:
		# No download progress bar needed, use standard httpx methods
		if is_async:
			task = asyncio.create_task(
				async_client.request(method, *args, **kwargs)
			)
			if callback is not None:
				task.add_done_callback(callback)
			return task
		else:
			res = httpx.request(method, *args, **kwargs)
			if callback is not None:
				callback(res)
			return res


def benchmark_requests(host, port, n=100):
	url = f"http://{host}:{port}/image"

	t0 = time.time()
	for i in range(n):
		res = httpx.get(url)
		loaded = pickle.loads(res.content)
		assert "image" in loaded

	elapsed = time.time() - t0
	print(f"Took {elapsed} s = {1000 * elapsed / n} ms/it")
