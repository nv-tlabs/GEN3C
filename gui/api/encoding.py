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

from enum import Enum
import io
import tempfile

import cv2
import numpy as np

class CompressionFormat(Enum):
	JPG = "jpg"
	PNG = "png"
	EXR = "exr"
	MP4 = "mp4"
	NPZ = "npz"

IMAGE_COMPRESSION_FORMATS = (CompressionFormat.JPG, CompressionFormat.PNG, CompressionFormat.EXR)


def compress_images(images: np.ndarray | None, format: CompressionFormat,
				    is_depth: bool = False, is_bool: bool = False) -> list[bytes] | None:
	"""
	Compress image(s) to the desired image format.
	Depth images should be encoded as EXR to preserve the data.
	"""
	if images is None:
		return None

	if is_depth or is_bool:
		assert images.ndim == 3, images.shape
	else:
		assert images.ndim == 4 and images.shape[-1] == 3, images.shape

	flags = []
	if format == CompressionFormat.JPG:
		flags = [int(cv2.IMWRITE_JPEG_QUALITY), 100]

	result = []
	if is_depth:
		# Note: leave as-is (floating point) to avoid quantization errors.
		assert format in (CompressionFormat.EXR, CompressionFormat.NPZ), "Depth images must be encoded as EXR or NPZ"
		images = images.astype(np.float32)
	elif is_bool:
		assert format == CompressionFormat.NPZ, "Bool images (e.g. masks) must be encoded as NPZ"
		images = images.astype(np.bool)
	else:
		images = (images * 255.0).astype(np.uint8)

	if format == CompressionFormat.NPZ:
		with io.BytesIO() as f:
			np.savez_compressed(f, images)
			result.append(f.getvalue())

	else:
		assert format in IMAGE_COMPRESSION_FORMATS, f"Unsupported image compression format: {format}"
		for i in range(images.shape[0]):
			_, encoded = cv2.imencode(f".{format.value}", images[i], flags)
			result.append(encoded.tobytes())

	return result


def decompress_buffer(buffers: list[bytes] | None, format: CompressionFormat,
					  is_depth: bool = False, is_bool: bool = False) -> np.ndarray | None:
	"""
	Returns the decoded image as 0..1 float values (or 0..inf for depth).
	"""
	if buffers is None:
		return None
	assert not (is_depth and is_bool), "Cannot be both a depth and a bool buffer."

	images = []
	for buf in buffers:

		if format == CompressionFormat.MP4:
			assert not is_bool and not is_depth, "Cannot decode a mask or depth from a video."

			# TODO: not sure why, but reading directly from the buffer leads to a segfault.
			# cap = cv2.VideoCapture(io.BytesIO(buf), apiPreference=cv2.CAP_FFMPEG, params=[])
			with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as f:
				f.write(buf)
				cap = cv2.VideoCapture(f.name)

				while True:
					ret, image = cap.read()
					if not ret:
						break
					image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
					# Note: the conversion from 0..1 to -1..1 will be done by the model.
					image = image.astype(np.float32) / 255.0
					images.append(image[None, ...])
				cap.release()

		else:
			if format == CompressionFormat.NPZ:
				image = np.load(io.BytesIO(buf), allow_pickle=False)
				if hasattr(image, "files"):
					assert len(image.files) == 1, image.files
					image = image[image.files[0]]
				# We assume it was saved with the right value range, shape and dtype.
				images.append(image)

			else:
				buf_np = np.frombuffer(buf, dtype=np.uint8)

				# OpenCV will automatically guess the image format.
				flags = cv2.IMREAD_ANYDEPTH if is_depth else cv2.IMREAD_ANYCOLOR
				image = np.array(cv2.imdecode(buf_np, flags))

				if is_bool:
					image = image.astype(np.bool)
				elif image.dtype == np.uint8:
					image = image.astype(np.float32) / 255.0

				images.append(image[None, ...])

	return np.concatenate(images, axis=0)



def pad_or_trim_array(arr: np.ndarray | None, target_size: int) -> np.ndarray | None:
	"""
	Pad or trim the array to the target size.
	"""
	if arr is None:
		return None

	n = arr.shape[0]
	if n == target_size:
		return arr
	elif n > target_size:
		return arr[:target_size]
	else:
		reps = (target_size - n, *([1] * (arr.ndim - 1)))
		return np.concatenate([
			arr,
			np.tile(arr[-1:], reps)
		], axis=0)



def pad_or_trim_encoded_buffers(buffers: list[bytes] | None, format: CompressionFormat,
							    target_size: int) -> list[bytes] | None:
	"""
	Pad or trim the encoded buffers to the target size.
	"""
	if buffers is None:
		return None

	if format in (CompressionFormat.JPG, CompressionFormat.PNG, CompressionFormat.EXR):
		# We just assume that there is one buffer per entry
		n = len(buffers)

		if n == target_size:
			return buffers
		elif n > target_size:
			return buffers[:target_size]
		else:
			return buffers + [buffers[-1]] * (target_size - n)

	elif format == CompressionFormat.NPZ:
		assert len(buffers) == 1, "NPZ buffers should be a single buffer"
		arr = np.load(io.BytesIO(buffers[0]), allow_pickle=False)
		if hasattr(arr, "files"):
			assert len(arr.files) == 1, arr.files
			arr = arr[arr.files[0]]

		arr = pad_or_trim_array(arr, target_size)
		with io.BytesIO() as f:
			np.savez_compressed(f, arr)
			return [f.getvalue()]


	elif format == CompressionFormat.MP4:
		# We assume there is one buffer per video
		assert len(buffers) == 1, "MP4 buffers should be a single buffer"
		buf = buffers[0]
		result = []

		# TODO: do all this with in-memory buffers instead of temporary files
		with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as f:
			f.write(buf)

			# Read back the video frame by frame
			cap = cv2.VideoCapture(f.name)
			fps = cap.get(cv2.CAP_PROP_FPS)
			width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			fourcc = cv2.VideoWriter_fourcc(*'mp4v')
			with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as f_out:
				out = cv2.VideoWriter(f_out.name, fourcc, fps, (width, height))

				n_written = 0
				last_frame = None
				for _ in range(target_size):
					ret, frame = cap.read()
					if not ret:
						break
					out.write(frame)
					last_frame = frame
					n_written += 1

				# If target size is longer than the original video, repeat the last valid frame
				for i in range(n_written, target_size):
					out.write(last_frame)

				out.release()

				f_out.seek(0)
				result.append(f_out.read())
			cap.release()

		return result

	else:
		raise ValueError(f"Unsupported compression format: {format}")
