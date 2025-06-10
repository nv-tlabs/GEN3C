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

from abc import abstractmethod
import asyncio
from os.path import realpath, dirname, join

from loguru import logger as log
import numpy as np

from api_types import InferenceRequest, InferenceResult, SeedingRequest


ROOT_DIR = realpath(dirname(dirname(dirname(__file__))))
DATA_DIR = join(ROOT_DIR, "data")


class InferenceModel():
	"""
	Base class for models that can be served by the inference server
	defined in `server.py`.
	"""

	def __init__(self, data_path: str | None = None, checkpoint_path: str | None = None,
				 fake_delay_ms: float = 0, inference_cache_size: int = 15,
				 compress_inference_results: bool = True) -> None:

		# These paths may be unused by certain inference server types.
		self.data_path = data_path
		self.checkpoint_path = checkpoint_path

		self.fake_delay_ms = fake_delay_ms
		self.inference_cache_size = inference_cache_size

		self.inference_tasks: dict[str, asyncio.Task] = {}
		self.inference_results: dict[str, InferenceResult] = {}
		self.request_history: set[str] = set()

		# If supported by the model and relevant, compress inference results,
		# e.g. as MP4 video, before returning from the server.
		self.compress_inference_results: bool = compress_inference_results

		# Can be acquired before starting inference
		# if the model can only handle one request at a time
		self.inference_lock = asyncio.Lock()

		# The generative model may need to be seeded with one or more initial frames.
		self.model_seeded = False


	# ----------- Inference model interface

	@abstractmethod
	async def make_test_image(self):
		"""Evaluate one default inference request, if possible.
		Helps ensuring that the model has been loaded correctly."""
		raise NotImplementedError("make_test_image")

	async def seed_model(self, req: SeedingRequest) -> None:
		"""By default, no seeding is required so the default implementation just returns."""
		self.model_seeded = True

	@abstractmethod
	async def run_inference(self, req: InferenceRequest) -> InferenceResult:
		"""Evaluate the actual inference model to produce an inference result."""
		raise NotImplementedError("run_inference")

	@abstractmethod
	def metadata(self) -> dict:
		"""Returns metadata about this inference server."""
		raise NotImplementedError("metadata")

	@abstractmethod
	def min_frames_per_request(self) -> int:
		"""Minimum number of frames that can be produced in one inference batch."""
		raise NotImplementedError("min_frames_per_request")

	@abstractmethod
	def max_frames_per_request(self) -> int:
		"""Maximum number of frames that can be produced in one inference batch."""
		raise NotImplementedError("max_frames_per_request")

	@abstractmethod
	def inference_time_per_frame(self) -> int:
		"""Estimated average inference time per frame (not per batch!) in seconds."""
		raise NotImplementedError("inference_time_per_frame")

	def inference_resolution(self) -> list[tuple[int, int]] | None:
		"""
		The supported inference resolutions (width, height) in pixels,
		or None if any resolution is supported.
		"""
		return None

	def default_framerate(self) -> float | None:
		"""
		The model's preferred framerate when generating video.
		Returns None when not applicable.
		"""
		return None

	@abstractmethod
	def requires_seeding(self) -> int:
		"""Whether or not this model requires to be seeded with images before inference."""
		return False

	# ----------- Requests handling

	def request_inference(self, req: InferenceRequest) -> asyncio.Task:
		if not self.model_seeded:
			raise ValueError(f"Received request id '{req.request_id}', but the model was not seeded.")
		if (req.request_id in self.inference_tasks) or (req.request_id in self.inference_results):
			raise ValueError(f"Invalid request id '{req.request_id}': request already exists.")
		self.check_valid_request(req)

		task = asyncio.create_task(self.run_inference(req))
		self.inference_tasks[req.request_id] = task
		self.request_history.add(req.request_id)
		return task


	async def request_inference_sync(self, req: InferenceRequest) -> InferenceResult:
		await self.request_inference(req)
		result = self.inference_result_or_none(req.request_id)
		assert isinstance(result, InferenceResult)
		return result


	def inference_result_or_none(self, request_id: str) -> InferenceResult | None:
		if request_id in self.inference_tasks:
			task = self.inference_tasks[request_id]
			if task.done():
				try:
					# Inference result ready, cache it and return it
					result = task.result()
					self.inference_results[request_id] = result
					del self.inference_tasks[request_id]
					self.evict_results()
					return result
				except Exception as e:
					# Inference failed
					log.error(f"Task for request '{request_id}' failed with exception {e}")
					raise e
			else:
				# Inference result not ready yet
				return None

		elif request_id in self.inference_results:
			# Inference result was ready and cached, return it directly
			return self.inference_results[request_id]

		elif request_id in self.request_history:
			raise KeyError(f"Request with id '{request_id}' was known, but does not have any result. Perhaps it was evicted from the cache or failed.")

		else:
			raise KeyError(f"Invalid request id '{request_id}': request not known.")


	def evict_results(self, keep_max: int | None = None):
		"""
		Evict all results that were added before the last `keep_max` entries.
		"""
		keep_max = keep_max if (keep_max is not None) else self.inference_cache_size

		to_evict = []
		for i, k in enumerate(reversed(self.inference_results)):
			if i < keep_max:
				continue
			to_evict.append(k)
		for k in to_evict:
			del self.inference_results[k]


	def get_latest_rgb(self) -> np.ndarray | None:
		"""Returns the latest generated RGB image, if any. Useful for debugging."""
		if not self.inference_results:
			return None
		last_key = next(reversed(self.inference_results.keys()))
		return self.inference_results[last_key].images[-1, ...]

	def check_valid_request(self, req: InferenceRequest):
		if len(req) not in range(self.min_frames_per_request(), self.max_frames_per_request() + 1):
			raise ValueError(f"This model can produce between {self.min_frames_per_request()} and"
							 f" {self.max_frames_per_request()} frames per request, but the request"
							 f" specified {len(req)} camera poses.")

		return True


	# ----------- Resource management
	def cleanup(self):
		pass
