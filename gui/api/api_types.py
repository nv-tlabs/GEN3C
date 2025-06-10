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
from dataclasses import dataclass, asdict
from enum import Enum
import os

# Enable OpenEXR support in OpenCV (disabled by default,
# see https://github.com/opencv/opencv/issues/21326).
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np

from encoding import CompressionFormat, IMAGE_COMPRESSION_FORMATS, compress_images, decompress_buffer, pad_or_trim_array, pad_or_trim_encoded_buffers


@dataclass(kw_only=True)
class RequestBase:
	request_id: str

	# Shape: [batch, 3, 4]
	cameras_to_world: np.ndarray
	# Absolute horizontal and vertical focal lengths in pixels.
	# They are expressed w.r.t. the `resolutions` field (pixel count and aspect ratio).
	# Shape: [batch, 2]
	focal_lengths: np.ndarray
	# Relative horizontal and vertical principal point (e.g. [0.5, 0.5])
	# Shape: [batch, 2]
	principal_points: np.ndarray
	# (width, height) in number of pixels.
	# Automatically set from the `self.images` field, if any.
	# Shape: [batch, 2]
	resolutions: np.ndarray = None

	# Number of frames in the original request, without padding.
	# Set automatically when calling `pad_to_frame_count()`.
	frame_count_without_padding: int | None = None


	def __post_init__(self):
		if hasattr(self, "images"):
			if self.resolutions is None:
				# Set resolution from the images
				self.resolutions = np.tile([[self.images.shape[2], self.images.shape[1]]],
										   (len(self), 1))
			else:
				# Check resolution against the images
				assert np.all(self.resolutions == (self.images.shape[2], self.images.shape[1]))
		elif self.resolutions is None:
			raise ValueError("Missing value `resolutions`")

		n = len(self)
		assert self.cameras_to_world.shape == (n, 3, 4)
		assert self.focal_lengths.shape == (n, 2)
		assert self.principal_points.shape == (n, 2)
		assert self.resolutions.shape == (n, 2)

	def world_to_cameras(self) -> np.ndarray:
		c2w_complete = np.zeros((self.cameras_to_world.shape[0], 4, 4), dtype=self.cameras_to_world.dtype)
		c2w_complete[:, :3, :] = self.cameras_to_world
		c2w_complete[:, 3, 3] = 1.0
		return np.linalg.inv(c2w_complete)

	def intrinsics_matrix(self, for_resolutions: np.ndarray | None) -> np.ndarray:
		"""Returns a batched intrinsics matrix [batch, 3, 3] following the
		format used by the Gen3C codebase."""
		result = np.zeros((len(self), 3, 3))
		# Focal length is already absolute
		result[:, 0, 0] = self.focal_lengths[:, 0]
		result[:, 1, 1] = self.focal_lengths[:, 1]
		# Note: convert from relative to absolute principal point
		result[:, 0, 2] = self.principal_points[:, 0] * self.resolutions[:, 0]
		result[:, 1, 2] = self.principal_points[:, 1] * self.resolutions[:, 1]
		result[:, 2, 2] += 1

		if for_resolutions is not None:
			# Resize intrinsics to match the new given resolutions
			assert for_resolutions.shape == self.resolutions.shape
			result[:, 0, :] *= (for_resolutions[:, 0, None] / self.resolutions[:, 0, None])
			result[:, 1, :] *= (for_resolutions[:, 1, None] / self.resolutions[:, 1, None])

		return result

	def resolution(self) -> tuple[int, int]:
		"""Resolution of the first image result in pixels as (width, height)."""
		return self.resolutions[0, 0], self.resolutions[0, 1]

	def __len__(self):
		return self.cameras_to_world.shape[0]


	def trim_to_original_frame_count(self, override_frame_count: int | None = None) -> None:
		"""
		Drop padding entries in order to match the original frame count.
		"""
		frame_count = override_frame_count or self.frame_count_without_padding
		print(f"Trimming {type(self).__name__} from {len(self)} back to original frame count {frame_count}.")
		if frame_count is None:
			return
		self._adjust_frame_count(frame_count)

	def pad_to_frame_count(self, n_frames: int) -> None:
		"""
		Add padding entries in order to match the desired frame count.
		Also records the current frame count as the original frame count.
		"""
		self.frame_count_without_padding = len(self)
		print(f"Padding {type(self).__name__} from {self.frame_count_without_padding} to {n_frames}.")
		self._adjust_frame_count(n_frames)


	def _adjust_frame_count(self, n_frames: int) -> None:
		"""
		Updates all fields to match the desired frame count.

		If it is higher than the current frame count, the last entry of each field is repeated.
		If it is lower, entries are dropped (from the end).
		"""
		self.cameras_to_world = pad_or_trim_array(self.cameras_to_world, n_frames)
		self.focal_lengths = pad_or_trim_array(self.focal_lengths, n_frames)
		self.principal_points = pad_or_trim_array(self.principal_points, n_frames)
		self.resolutions = pad_or_trim_array(self.resolutions, n_frames)


@dataclass(kw_only=True)
class SeedingRequest(RequestBase):
	"""
	Contains data required to seed the Gen3C model with initial data
	to bootstrap generation.

	Note that the intrinsics (defined in `RequestBase`) are provided
	as a suggestion and may be ignored by the model, if it is able to
	estimate them from the images instead.
	TODO: maybe provide a flag to indicate that the model should really
	      respect the provided values.
	"""
	# Values in [0, 1].
	# Shape: [batch, height, width, 3], float32
	images: np.ndarray
	# Per-pixel depth for each of the given images. If not provided, it
	# should be estimated automatically by the model.
	# Shape: [batch, height, width], float32
	depths: np.ndarray | None
	# Per-pixel mask for each of the given images. If not provided, it
	# should be estimated automatically by the model.
	# Shape: [batch, height, width], bool
	masks: np.ndarray | None = None

	def __post_init__(self):
		super().__post_init__()
		n = len(self)
		assert self.images.shape[0] == n and self.images.ndim == 4, self.images.shape
		if self.depths is not None:
			assert self.depths.shape[0] == n and self.depths.ndim == 3, self.depths.shape
		if self.masks is not None:
			assert self.masks.shape[0] == n and self.masks.ndim == 3, self.masks.shape


	def _adjust_frame_count(self, n_frames: int) -> None:
		raise RuntimeError("SeedingRequest: _adjust_frame_count() not supported")


	def compress(self,
				 format_rgb: CompressionFormat = CompressionFormat.JPG,
				 format_depth: CompressionFormat | None = None,
				 format_mask: CompressionFormat | None = None) -> "CompressedSeedingRequest":
		"""Compress the images and depths as images and return a `CompressedSeedingRequest`."""
		images_compressed = compress_images(self.images, format_rgb, is_depth=False)

		format_depth = format_depth or CompressionFormat.EXR
		depths_compressed = compress_images(self.depths, format_depth,
											is_depth=True)

		format_mask = format_mask or CompressionFormat.NPZ
		masks_compressed = compress_images(self.masks, format_mask,
										   is_bool=True)

		kwargs = asdict(self)
		# Will be replaced automatically with placeholders of the right shape.
		kwargs['images'] = None
		kwargs['depths'] = None
		kwargs['masks'] = None
		return CompressedSeedingRequest(
			images_compressed=images_compressed,
			images_format=format_rgb,
			depths_compressed=depths_compressed,
			depths_format=format_depth,
			masks_compressed=masks_compressed,
			masks_format=format_mask,
			**kwargs
		)


@dataclass(kw_only=True)
class CompressedSeedingRequest(SeedingRequest):
	"""
	Same as a `SeedingRequest`, but the image and depth buffers are
	sent as compressed JPG / PNG streams instead of raw bytes.
	They should be decompressed when received, before being used
	as standard `SeedingRequest`s.
	"""
	# List of compressed images (as raw bytes)
	images_compressed: list[bytes]
	images_format: CompressionFormat
	# List of compressed images (as raw bytes)
	depths_compressed: list[bytes] | None
	depths_format: CompressionFormat | None
	# List of compressed masks (as raw bytes)
	masks_compressed: list[bytes] | None
	masks_format: CompressionFormat | None

	def __post_init__(self):
		# Note: not calling parent checks because our image and depth
		# fields are not actually usable as-is.
		# super().__post_init__()

		# For convenience, auto-fill placeholder image and depth fields
		assert (self.resolutions is not None) or (self.images is not None), \
			   "CompressedSeedingRequest: at least one of resolutions or images must be provided"

		w, h = self.resolution()
		if self.images is None:
			self.images = np.empty((0, h, w, 3), dtype=np.float32)
		if (self.depths is None) and (self.depths_compressed is not None):
			self.depths = np.empty((0, h, w), dtype=np.float32)
		if (self.masks is None) and (self.masks_compressed is not None):
			self.masks = np.empty((0, h, w), dtype=np.bool)

		assert self.images.shape[0] == 0, \
			   "CompressedSeedingRequest should not have any raw image data"\
				" in `self.images` upon construction."

	def decompress(self) -> None:
		"""Decompress the images and fill them in-place."""
		self.images = decompress_buffer(self.images_compressed, self.images_format)
		self.depths = decompress_buffer(self.depths_compressed, self.depths_format, is_depth=True)
		self.masks = decompress_buffer(self.masks_compressed, self.masks_format, is_bool=True)


@dataclass(kw_only=True)
class SeedingResult(RequestBase):
	"""
	Contains the result of a seeding request,
	e.g. the depth maps for the seeding images that were estimated by the model
	if not provided in the original request.

	Note: since the `depths` field would need to remain relatively high-precision
	and lossless when compressed, we don't bother overring a compressed version
	of `SeedingResult` for now.
	"""
	# Per-pixel depth for each of the given images.
	# Shape: [batch, height, width]
	depths: np.ndarray | None = None

	def __post_init__(self):
		super().__post_init__()
		n = len(self)
		if self.depths is not None:
			if self.depths.ndim == 4 and self.depths.shape[1] == 1:
				# [batch, 1, height, width] -> [batch, height, width]
				self.depths = self.depths.squeeze(1)
			assert self.depths.shape[0] == n and self.depths.ndim == 3

	@staticmethod
	def from_request(req: SeedingRequest, fallback_depths: np.ndarray | None) -> "SeedingResult":
		resolutions = req.resolutions
		if fallback_depths is not None:
			resolutions[:, 0] = fallback_depths.shape[2]
			resolutions[:, 1] = fallback_depths.shape[1]

		return SeedingResult(
			request_id=req.request_id,
			cameras_to_world=req.cameras_to_world,
			focal_lengths=req.focal_lengths,
			principal_points=req.principal_points,
			resolutions=resolutions,
			depths=None if (req.depths is not None) else fallback_depths,
		)

	def _adjust_frame_count(self, n_frames: int) -> None:
		raise RuntimeError("SeedingRequest: _adjust_frame_count() not supported")


@dataclass(kw_only=True)
class InferenceRequest(RequestBase):
	# Time points for each frame to generate (useful when there's scene dynamics).
	# May be ignored by the model.
	# Shape: [batch,]
	timestamps: np.ndarray

	# Framerate of the generated video (frames per second). Only applicable
	# when requesting multiple frames at once.
	# May be ignored by the model, or rounded to the nearest integer.
	framerate: float = 30.0

	# Whether to estimate and return depth for each frame in the result.
	return_depths: bool = False

	# If inference results will be returned as a compressed video, use this
	# encoding quality (0..10).
	video_encoding_quality: int = 8

	# Whether to include the rendered cache in the generated video (for debugging / visualization)
	show_cache_renderings: bool = False

	def __post_init__(self):
		super().__post_init__()
		n = len(self)
		assert self.timestamps.shape[0] == n and self.timestamps.ndim == 1, \
			   f"Timestamps: expected shape ({n},), found: {self.timestamps.shape}"
		assert len(self.focal_lengths) == n
		assert len(self.principal_points) == n
		assert len(self.resolutions) == n

	def _adjust_frame_count(self, n_frames: int) -> None:
		super()._adjust_frame_count(n_frames)
		self.timestamps = pad_or_trim_array(self.timestamps, n_frames)


@dataclass(kw_only=True)
class InferenceResult(RequestBase):
	"""
	Note that fields that are already included in the request are repeated here,
	simply because the model may not have respected the request.
	"""
	# The model can use this field to indicate that multiple returned results
	# are identical, so that the client can decide to skip adding them.
	# It should be ignored if set to None.
	result_ids: list[str | None]

	# Shape: [batch,]
	timestamps: list[float]
	# Shape: [batch, height, width, 3]
	images: np.ndarray
	# Shape: [batch, height, width]
	depths: np.ndarray

	# Time it took to generate the whole batch of results, in milliseconds
	runtime_ms: float

	def __post_init__(self):
		super().__post_init__()
		n = len(self)
		assert self.timestamps.shape[0] == n and self.timestamps.ndim == 1, \
			   f"Timestamps: expected shape ({n},), found: {self.timestamps.shape}"
		assert self.images.ndim == 4 and self.images.shape[0] == n, self.images.shape
		assert self.depths.ndim == 3 and self.depths.shape[0] == n, self.depths.shape


	def _adjust_frame_count(self, n_frames: int) -> None:
		super()._adjust_frame_count(n_frames)
		self.timestamps = pad_or_trim_array(self.timestamps, n_frames)

		if self.images.shape[0] == 0:
			# Fields are just placeholders (compressed request), leave them be.
			return

		self.images = pad_or_trim_array(self.images, n_frames)
		self.depths = pad_or_trim_array(self.depths, n_frames)



@dataclass(kw_only=True)
class CompressedInferenceResult(InferenceResult):
	"""
	Same as a `InferenceResult`, but the image and depth buffers are
	sent as compressed MP4 / EXR streams instead of raw bytes.
	They should be decompressed when received, before being used
	as standard `InferenceResult`s.
	"""
	# List of compressed images (as raw bytes)
	images_compressed: list[bytes]
	images_format: CompressionFormat
	# List of compressed images (as raw bytes)
	depths_compressed: list[bytes] | None
	depths_format: CompressionFormat | None

	def __post_init__(self):
		# Note: not calling parent checks because our image and depth
		# fields are not actually usable as-is.
		# super().__post_init__()

		# For convenience, auto-fill placeholder image and depth fields
		assert (self.resolutions is not None) or (self.images is not None), \
			   "CompressedInferenceResult: at least one of resolutions or images must be provided"
		w, h = self.resolution()
		if self.images is None:
			self.images = np.empty((0, h, w, 3), dtype=np.float32)
		if (self.depths is None) and (self.depths_compressed is not None):
			self.depths = np.empty((0, h, w), dtype=np.float32)

		assert self.images.shape[0] == 0, \
			   "CompressedInferenceResult should not have any raw image data" \
				" in `self.images` upon construction."

		if self.images_format == CompressionFormat.MP4:
			assert len(self.images_compressed) == 1, \
				   "CompressedInferenceResult: with an MP4 compressed result," \
					" there should be only one buffer (the compressed video)."
		elif self.depths_compressed is not None:
			assert len(self.depths_compressed) == len(self.images_compressed)
			assert self.depths_format in IMAGE_COMPRESSION_FORMATS, \
				   f"CompressedInferenceResult: depths_format should be an image format, found {self.depths_format}"


	def _adjust_frame_count(self, n_frames: int) -> None:
		super()._adjust_frame_count(n_frames)
		self.images_compressed = pad_or_trim_encoded_buffers(self.images_compressed, self.images_format, n_frames)
		self.depths_compressed = pad_or_trim_encoded_buffers(self.depths_compressed, self.depths_format, n_frames)


	def decompress(self) -> None:
		"""Decompress the images and fill them in-place."""
		self.images = decompress_buffer(self.images_compressed, self.images_format, is_depth=False)
		self.depths = decompress_buffer(self.depths_compressed, self.depths_format, is_depth=True)


	def save_images(self, fname_or_directory: str) -> None:
		"""Save the compressed images to a file path or directory.

		If a full path is given, the path extension may be overriden based
		on the compression format.
		"""
		fname_or_directory = os.path.realpath(fname_or_directory)
		base, ext = os.path.splitext(fname_or_directory)
		if not ext:
			directory = fname_or_directory
			base = "inference_result"
		else:
			directory = os.path.dirname(fname_or_directory)
			base = os.path.splitext(os.path.basename(fname_or_directory))[0]

		os.makedirs(directory, exist_ok=True)
		single = len(self.images_compressed) == 1
		for i, buf in enumerate(self.images_compressed):
			image_path = os.path.join(
				directory,
				f"{base}.{self.images_format.value}"
				if single else f"base_{i:05d}.{self.images_format.value}"
			)
			with open(image_path, "wb") as f:
				f.write(buf)


class RequestState(Enum):
	"""
	Note: by "request" we mean an inference request, not an HTTP request.
	"""
	REQUEST_PENDING = "Request pending"
	REQUEST_SENT = "Request sent"
	RESULT_PENDING = "Result pending"
	COMPLETE = "Completed"
	FAILED = "Created"


@dataclass(kw_only=True)
class PendingRequest:
	request_id: str
	state: RequestState
	message: str = ""
	task: asyncio.Task | None = None
