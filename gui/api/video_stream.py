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


import cv2
import numpy as np

class RawVideoStream():
	"""
	A video stream from a raw mp4 file, using opencv.
	This does not support nested iterations.
	"""

	def __init__(
		self, path: str, seek_range: range | None = None
	) -> None:
		super().__init__()
		if seek_range is None:
			seek_range = range(-1)

		self.path = path

		# Read metadata
		vcap = cv2.VideoCapture(self.path)
		self._width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
		self._height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		_fps = vcap.get(cv2.CAP_PROP_FPS)
		_n_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
		vcap.release()

		self.start = seek_range.start
		self.end = seek_range.stop if seek_range.stop != -1 else _n_frames
		self.end = min(self.end, _n_frames)
		self.step = seek_range.step
		self._fps = _fps / self.step

	def frame_size(self) -> tuple[int, int]:
		"""Returns (height, width)."""
		return (self._height, self._width)

	def fps(self) -> float:
		return self._fps

	def __len__(self) -> int:
		return len(range(self.start, self.end, self.step))

	def __iter__(self):
		self.vcap = cv2.VideoCapture(self.path)
		self.current_frame_idx = -1
		return self

	def __next__(self) -> tuple[int, np.ndarray]:
		while True:
			ret, frame = self.vcap.read()
			self.current_frame_idx += 1

			if not ret:
				self.vcap.release()
				raise StopIteration

			if self.current_frame_idx >= self.end:
				self.vcap.release()
				raise StopIteration

			if self.current_frame_idx < self.start:
				continue

			if (self.current_frame_idx - self.start) % self.step == 0:
				break

		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		return self.current_frame_idx, frame
