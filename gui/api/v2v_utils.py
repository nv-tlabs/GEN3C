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

import glob
import io
import os
from os.path import join, isdir, isfile
import zipfile

import imageio.v3 as imageio
import json
import numpy as np
import pyexr
from tqdm import tqdm

from api_types import CompressedSeedingRequest, SeedingRequest
from encoding import CompressionFormat


def srgb_to_linear(img):
	limit = 0.04045
	mask = img > limit

	# Process the two cases in parallel using NumPy's vectorized operations
	result = np.empty_like(img)
	result[mask] = np.power((img[mask] + 0.055) / 1.055, 2.4)
	result[~mask] = img[~mask] / 12.92

	return result


def load_gen3c_seeding_data(data_directory: str, max_frames: int | None = None,
							frames_stride: int = 1) -> CompressedSeedingRequest:
	"""
	Example directory structure:
	├── camera.npz
	├── depth.npz
	├── mask.npz
	├── metadata.json
	└── rgb.mp4

	We will keep the data compressed as much as possible so that it can
	be uploaded faster to the inference server.
	"""
	bar = tqdm(range(6), desc="Seeding data loading")

	# [n_frames, height, width], float16
	depths = np.load(join(data_directory, "depth.npz"))['depth']
	assert depths.ndim == 3, depths.shape
	n_img = depths.shape[0]
	resolutions = np.tile([depths.shape[2], depths.shape[1]], reps=(n_img, 1))
	assert resolutions.shape == (n_img, 2)

	with io.BytesIO() as f:
		np.savez_compressed(f, depths)
		depths_compressed = f.getvalue()
	bar.update(1)

	# Intrinsics: [n_frames, 3, 3], float32
	# Organized as:
	#   [[fx,  0, cx],
	#    [ 0, fy, cy],
	#    [ 0,  0,  1]]
	camera_data = np.load(join(data_directory, "camera.npz"))
	intrinsics = camera_data['intrinsics']
	# Absolute focal lengths
	focal_lengths = np.stack([intrinsics[:, 0, 0], intrinsics[:, 1, 1]], axis=1)
	assert focal_lengths.shape == (n_img, 2)
	# Relative principal points
	principal_points = (intrinsics[:, :2, 2] / resolutions).astype(np.float32)
	assert principal_points.shape == (n_img, 2)
	bar.update(1)

	# [n_frames, height, width], bool
	masks = np.load(join(data_directory, "mask.npz"))['mask']
	with io.BytesIO() as f:
		np.savez_compressed(f, masks)
		masks_compressed = f.getvalue()
	bar.update(1)

	# TODO: set the frontend's FPS slider based on `metadata["fps"]`
	# metadata = json.load(open(join(data_directory, "metadata.json")))
	bar.update(1)

	images_compressed = open(join(data_directory, "rgb.mp4"), "rb").read()
	bar.update(1)

	# [n_frames, 4, 4], float32
	w2c = camera_data['w2c']
	cameras_to_world = np.linalg.inv(w2c)[:, :3, :]
	assert cameras_to_world.shape == (n_img, 3, 4)
	bar.update(1)


	return CompressedSeedingRequest(
		request_id="__seeding_from_files",
		images=None,  # Will be auto-filled with placeholders
		depths=None,  # Will be auto-filled with placeholders
		masks=None,  # Will be auto-filled with placeholders
		cameras_to_world=cameras_to_world,
		focal_lengths=focal_lengths,
		principal_points=principal_points,
		resolutions=resolutions,
		images_compressed=[images_compressed],
		images_format=CompressionFormat.MP4,
		depths_compressed=[depths_compressed],
		depths_format=CompressionFormat.NPZ,
		masks_compressed=[masks_compressed],
		masks_format=CompressionFormat.NPZ,
	)



def load_v2v_seeding_data(data_directory: str, max_frames: int | None = None,
						  frames_stride: int = 1) -> SeedingRequest:
	"""
	The seeding data would typically come from the client.
	For convenience during debugging, we allow loading it here.
	"""

	if isdir(data_directory):
		# --- Load seeding data from a directory.
		if isfile(join(data_directory, "rgb.mp4")) and isfile(join(data_directory, "metadata.json")):
			return load_gen3c_seeding_data(data_directory, max_frames=max_frames,
										   frames_stride=frames_stride)

		# Gen3C / INGP pre-processed format.
		# We assume depths, camera poses, etc are included.
		# Load the seeding frames
		n_img = len([img for img in sorted(os.listdir(join(data_directory, 'rgb')))
							if img.endswith('.jpg')])
		images = []
		depths = []
		for i_frame in range(n_img):
			# Load image data
			image = imageio.imread(join(data_directory, 'rgb', f'{i_frame:05d}.jpg'))
			image_np = image.astype(np.float32) / 255.0

			# Load depth data
			depth_np = np.load(join(data_directory, 'depth', f'{i_frame:05d}.npz'))['depth']
			images.append(image_np)
			depths.append(depth_np)
		del image_np, depth_np

		# Load camera trajectory
		with open(join(data_directory, 'cameras.json'), 'r') as f:
			cameras = json.load(f)
		cameras_to_world = np.asarray(cameras)[:n_img]

		if (max_frames is not None) and (max_frames < len(images)):
			images = images[::frames_stride][:max_frames]
			depths = depths[::frames_stride][:max_frames]
			cameras_to_world = cameras_to_world[::frames_stride][:max_frames]


	else:
		# --- Load a single image.
		# We will have to assume camera poses, etc and let depth be auto-estimated.
		n_img = 1
		image = imageio.imread(data_directory)
		images = [image.astype(np.float32) / 255.0]
		depths = None
		cameras_to_world = np.eye(4)[None, :3, :]

	# Shape: [batch, height, width, 3]
	images = np.stack(images, axis=0)
	if depths is not None:
		# Shape: [batch, height, width]
		depths = np.stack(depths, axis=0)

	# Note: assumed based on how this data was generated
	resolutions = np.tile([images.shape[2], images.shape[1]], reps=(n_img, 1))
	fov_y_rad = np.pi * (50.625 / 180.0)
	f = 0.5 / (np.tan(fov_y_rad / 2.0)) * resolutions[:, 1]
	focal_lengths = np.stack([f, f], axis=-1)
	principal_points = np.full((n_img, 2), 0.5)

	return SeedingRequest(
		request_id="__seeding_from_files",
		images=images,
		depths=depths,
		cameras_to_world=cameras_to_world,
		focal_lengths=focal_lengths,
		principal_points=principal_points,
		resolutions=resolutions,
	)


def ensure_alpha_channel(image: np.ndarray):
	# Allow alpha channel to be omitted for faster transfers
	assert image.shape[-1] in (3, 4)
	if image.shape[-1] == 3:
		image = np.concatenate([image, np.ones((*image.shape[:2], 1))],
								axis=-1)
	image = image.astype(np.float32)
	return image


def apply_to_pytree(pytree, cb):
	tp = type(pytree)
	if pytree is None:
		return None
	elif isinstance(pytree, (tuple, list)):
		return tp([apply_to_pytree(v, cb) for v in pytree])
	elif isinstance(pytree, dict):
		return { k: apply_to_pytree(v, cb) for k, v in pytree.items() }
	else:
		return cb(pytree)


def move_to_device(pytree, device):
	import torch

	def move(pytree):
		if torch.is_tensor(pytree):
			return pytree.to(device)
		elif isinstance(pytree, np.ndarray):
			return torch.from_numpy(pytree).to(device)
		else:
			# Let's assume it's a not something we need to move
			return pytree
			# raise NotImplementedError(f"move_to_device(): unsupported type {type(pytree)}")

	return apply_to_pytree(pytree, move)


def clone_tensors(pytree):
	import torch

	def clone(pytree):
		if torch.is_tensor(pytree):
			return pytree.clone()
		elif isinstance(pytree, np.ndarray):
			return pytree.copy()
		else:
			# Let's assume it's a not something we need to copy
			return pytree
			# raise NotImplementedError(f"clone_tensors(): unsupported type {type(pytree)}")

	return apply_to_pytree(pytree, clone)
