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
import time
try:
    from typing import override
except ImportError:
	def override(f):
		return f

from loguru import logger as log
import numpy as np

from api_types import InferenceRequest, InferenceResult, CompressedInferenceResult, SeedingRequest, SeedingResult
from encoding import compress_images, CompressionFormat
from server_base import InferenceModel


class CosmosBaseModel(InferenceModel):
	"""
	Wraps a video generative model.
	"""

	def __init__(self, **kwargs):
		super().__init__(**kwargs)


	@override
	async def make_test_image(self) -> InferenceResult:
		raise NotImplementedError("Not implemented: make_test_image()")


	async def seed_model(self, req: SeedingRequest) -> None:
		import torch

		log.info(f"Seeding the model with request ID '{req.request_id}' ({len(req)} frames)")
		# TODO: option to seed without clearing the existing cache
		if self.pose_history_w2c:
			log.info("[i] Clearing existing 3D cache and history due to seeding.")
		self.model.clear_cache()
		self.pose_history_w2c.clear()
		self.intrinsics_history.clear()

		if hasattr(self.model, 'seed_model_from_values'):
			seeding_method = self.model.seed_model_from_values
		else:
			raise RuntimeError(f"Could not locate seeding method in model.")

		model_result = seeding_method(
			images_np=req.images,
			depths_np=req.depths,
			masks_np=req.masks,
			world_to_cameras_np=req.world_to_cameras(),
			focal_lengths_np=req.focal_lengths,
			principal_point_rel_np=req.principal_points,
			resolutions=req.resolutions,
		)
		self.model_seeded = True
		log.info("[+] Model seeded.")

		out_depths = None if (req.depths is not None) else self.model.get_cache_input_depths().cpu().numpy()
		if model_result is None:
			return SeedingResult.from_request(req, fallback_depths=out_depths)
		else:
			model_result = list(model_result)
			for i, r in enumerate(model_result):
				if isinstance(r, torch.Tensor):
					model_result[i] = r.cpu().numpy()

			(estimated_w2c_b44, estimated_focal_lengths_b2,
			 estimated_principal_point_abs_b2, working_resolutions_b2) = model_result

			# Principal point is expected to be relative to the resolution
			estimated_principal_point_rel_b2 = estimated_principal_point_abs_b2 / working_resolutions_b2

			return SeedingResult(
				request_id=req.request_id,
				cameras_to_world=estimated_w2c_b44[:, :3, :],
				focal_lengths=estimated_focal_lengths_b2,
				principal_points=estimated_principal_point_rel_b2,
				resolutions=working_resolutions_b2,
				depths=out_depths
			)

	@override
	async def run_inference(self, req: InferenceRequest) -> InferenceResult:
		import torch

		async with self.inference_lock:
			log.info(f"[+] Running inference for request \"{req.request_id}\"...")
			start_time = time.time()

			w2c = req.world_to_cameras()
			# Tricky: we receive intrinsics as in absolute units, assuming the
			# resolution requested by the user. But the V2V codebase expects
			# intrinsics in absolute units w.r.t. the *original seeding resolution*.
			original_res = req.resolutions.copy()
			original_res[:, 0] = self.model.W
			original_res[:, 1] = self.model.H
			intrinsics = req.intrinsics_matrix(for_resolutions=original_res)

			# We allow some overlaps on the cameras here during the inference.
			if len(self.pose_history_w2c) == 0:
				# First request: no frames to overlap
				overlap_frames = 0 # from which frame the model starts prediction
			else:
				# Subsequent requests: reuse `overlap_frames` poses from the most
				# recent completed request.
				overlap_frames = self.model.inference_overlap_frames
				assert overlap_frames < self.min_frames_per_request()

				w2c = np.concatenate([
					self.pose_history_w2c[-1][-overlap_frames:, ...],
					w2c[:-overlap_frames, ...]
				], axis=0)
				intrinsics = np.concatenate([
					self.intrinsics_history[-1][-overlap_frames:, ...],
					intrinsics[:-overlap_frames, ...]
				], axis=0)

			self.pose_history_w2c.append(w2c)
			self.intrinsics_history.append(intrinsics)

			# Run inference given the cameras
			inference_results = self.model.inference_on_cameras(
				w2c,
				intrinsics,
				fps=req.framerate,
				overlap_frames=overlap_frames,
				return_estimated_depths=req.return_depths,
				video_save_quality=req.video_encoding_quality,
				save_buffer=req.show_cache_renderings,
			)
			if isinstance(inference_results, dict):
				pred_no_overlap = inference_results['video_no_overlap']
				predicted_depth = inference_results['predicted_depth']
				video_save_path = inference_results.get('video_save_path')
			else:
				# Assume tuple or list
				_, _, _, pred_no_overlap, predicted_depth = inference_results
				video_save_path = None

			# Instead of synchronizing, which will block this thread and never yield to the
			# asyncio event loop, we record a CUDA event and yield until it is reached
			# by the GPU (= inference is complete).
			cuda_event = torch.cuda.Event()
			cuda_event.record()
			while not cuda_event.query():
				await asyncio.sleep(0.0005)

			if self.fake_delay_ms > 0:
				await asyncio.sleep(self.fake_delay_ms / 1000.0)

		# Note: we remove the overlap frame(s), if any, before returning the result.
		if isinstance(pred_no_overlap, torch.Tensor):
			pred_no_overlap = pred_no_overlap.cpu().numpy()
		if pred_no_overlap.ndim == 5:
			assert pred_no_overlap.shape[0] == 1, pred_no_overlap.shape
			pred_no_overlap = pred_no_overlap.squeeze()
		n_frames = pred_no_overlap.shape[0]
		# Reorder [n_frames, channels, height, width] to [n_frames, height, width, channels]
		images = pred_no_overlap.transpose(0, 2, 3, 1)

		if req.return_depths:
			if isinstance(predicted_depth, torch.Tensor):
				predicted_depth = predicted_depth.cpu().numpy()
			# Desired shape: n_frames, height, width
			if predicted_depth.ndim == 4:
				assert predicted_depth.shape[1] == 1, predicted_depth.shape
				predicted_depth = predicted_depth[:, 0, ...]
			depths = predicted_depth
		else:
			depths = None

		# TODO: for dynamic scenes, get actual timestamps for each frame?
		timestamps = np.zeros((n_frames,))

		upper = (-overlap_frames) if (overlap_frames > 0) else None  # For easier slicing
		kwargs = {
			'request_id': req.request_id,
			'result_ids': [f"{req.request_id}__frame_{k}" for k in range(n_frames)],
			'timestamps': timestamps,
			'cameras_to_world': req.cameras_to_world[:upper, ...],
			'focal_lengths': req.focal_lengths[:upper, ...],
			'principal_points': req.principal_points[:upper, ...],
			'frame_count_without_padding': req.frame_count_without_padding,
			'runtime_ms': 1000 * (time.time() - start_time),
		}
		if self.compress_inference_results and (video_save_path is not None):
			video_bytes = open(video_save_path, "rb").read()
			depths_compressed = compress_images(depths, CompressionFormat.NPZ, is_depth=True)

			result = CompressedInferenceResult(
				images=None,
				depths=None,
				resolutions=np.tile([[images.shape[2], images.shape[1]]], (images.shape[0], 1)),
				images_compressed=[video_bytes],
				images_format=CompressionFormat.MP4,
				depths_compressed=depths_compressed,  # May be None
				depths_format=CompressionFormat.NPZ,
				**kwargs
			)
		else:
			result = InferenceResult(
				images=images,
				depths=depths,
				**kwargs
			)

		return result


	@override
	def min_frames_per_request(self) -> int:
		# Note this might not be strictly respected due to overlap frames,
		# starting at the second inference batch.
		return self.model.frames_per_batch

	@override
	def max_frames_per_request(self) -> int:
		return self.model.frames_per_batch

	def inference_resolution(self) -> list[tuple[int, int]] | None:
		"""The supported inference resolutions (width, height),
		or None if any resolution is supported."""
		try:
			r = self.model.cfg.train_data.shared_params.crop_size
		except AttributeError:
			r = (self.model.H, self.model.W)
		return [(r[1], r[0]),]

	@override
	def inference_time_per_frame(self) -> int:
		# TODO: actual mean inference time
		return 4.0

	@override
	def requires_seeding(self) -> bool:
		return True

	@override
	def metadata(self) -> dict:
		return {
			"model_name": "CosmosBaseModel",
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
