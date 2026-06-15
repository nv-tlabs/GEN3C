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

import numpy as np

from api_types import InferenceRequest, InferenceResult, SeedingRequest, SeedingResult
from server_base import InferenceModel


class DebugInferenceModel(InferenceModel):
	"""
	Small deterministic model for exercising the API server without loading
	checkpoints, CUDA, or the Cosmos inference stack.
	"""

	def __init__(self, *args, gpu_count: int = 0, **kwargs) -> None:
		super().__init__(*args, compress_inference_results=False, **kwargs)
		self.model_seeded = True
		self.aabb_min = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
		self.aabb_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)

	async def make_test_image(self):
		req = InferenceRequest(
			request_id="debug-startup",
			timestamps=np.array([0.0], dtype=np.float32),
			cameras_to_world=np.zeros((1, 3, 4), dtype=np.float32),
			focal_lengths=np.ones((1, 2), dtype=np.float32),
			principal_points=np.full((1, 2), 0.5, dtype=np.float32),
			resolutions=np.array([[16, 8]], dtype=np.int32),
			return_depths=True,
		)
		result = await self.run_inference(req)
		self.inference_results[req.request_id] = result
		self.request_history.add(req.request_id)
		return result

	async def seed_model(self, req: SeedingRequest) -> SeedingResult:
		self.model_seeded = True
		fallback_depths = None
		if req.depths is None:
			width, height = req.resolution()
			fallback_depths = np.ones((len(req), height, width), dtype=np.float32)
		return SeedingResult.from_request(req, fallback_depths=fallback_depths)

	async def run_inference(self, req: InferenceRequest) -> InferenceResult:
		width, height = req.resolution()
		x = np.linspace(0.0, 1.0, width, dtype=np.float32)
		y = np.linspace(0.0, 1.0, height, dtype=np.float32)
		xx, yy = np.meshgrid(x, y)

		images = []
		depths = []
		for i in range(len(req)):
			frame_value = np.float32((i + 1) / max(len(req), 1))
			images.append(np.stack([xx, yy, np.full_like(xx, frame_value)], axis=-1))
			depths.append(np.full((height, width), frame_value, dtype=np.float32))

		return InferenceResult(
			request_id=req.request_id,
			result_ids=[f"{req.request_id}__debug_{i}" for i in range(len(req))],
			timestamps=req.timestamps.copy(),
			cameras_to_world=req.cameras_to_world.copy(),
			focal_lengths=req.focal_lengths.copy(),
			principal_points=req.principal_points.copy(),
			resolutions=req.resolutions.copy(),
			frame_count_without_padding=req.frame_count_without_padding,
			images=np.stack(images, axis=0),
			depths=np.stack(depths, axis=0),
			runtime_ms=0.0,
		)

	def metadata(self) -> dict:
		return {
			"model_name": "DebugInferenceModel",
			"model_version": (1, 0, 0),
			"aabb_min": self.aabb_min.tolist(),
			"aabb_max": self.aabb_max.tolist(),
			"min_frames_per_request": self.min_frames_per_request(),
			"max_frames_per_request": self.max_frames_per_request(),
			"inference_resolution": self.inference_resolution(),
			"inference_time_per_frame": self.inference_time_per_frame(),
			"default_framerate": self.default_framerate(),
			"requires_seeding": self.requires_seeding(),
		}

	def min_frames_per_request(self) -> int:
		return 1

	def max_frames_per_request(self) -> int:
		return 16

	def inference_time_per_frame(self) -> float:
		return 0.0

	def inference_resolution(self) -> list[tuple[int, int]]:
		return [(16, 8), (64, 32)]

	def default_framerate(self) -> float:
		return 24.0

	def requires_seeding(self) -> bool:
		return False
