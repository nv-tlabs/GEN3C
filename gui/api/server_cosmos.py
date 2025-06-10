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
from os.path import join, realpath
import sys
try:
    from typing import override
except ImportError:
	def override(f):
		return f

from loguru import logger as log
import numpy as np

from multi_gpu import MultiGPUInferenceAR
from server_base import ROOT_DIR
from server_cosmos_base import CosmosBaseModel

COSMOS_PREDICT1_ROOT = ROOT_DIR

TORCHRUN_DEFAULT_MASTER_ADDR = 'localhost'
TORCHRUN_DEFAULT_MASTER_PORT = 12355


def add_cosmos_venv_to_path():
	version_string = f"python{sys.version_info.major}.{sys.version_info.minor}"
	extras = [
		COSMOS_PREDICT1_ROOT,
		join(COSMOS_PREDICT1_ROOT, "cosmos_predict1"),
	]
	for e in extras:
		if e not in sys.path:
			sys.path.append(e)


class CosmosModel(CosmosBaseModel):
	"""
	Serves frames generated on-the-fly by the Cosmos generative model.
	Intended for use with the Cosmos-Predict-1 based Gen3C model.
	"""

	def __init__(self, gpu_count: int = 0, **kwargs):
		add_cosmos_venv_to_path()
		if not os.environ.get("HF_HOME"):
			os.environ["HF_HOME"] = join(COSMOS_PREDICT1_ROOT, "huggingface_home")

		super().__init__(**kwargs)

		assert os.path.isdir(join(COSMOS_PREDICT1_ROOT, "cosmos_predict1")), \
			   f"Could not find Cosmos (cosmos_predict1) directory at: {COSMOS_PREDICT1_ROOT}"


		from cosmos_predict1.diffusion.inference.gen3c_persistent import Gen3cPersistentModel, create_parser
		import torch

		if gpu_count == 0:
			# Use as many GPUs for inference as are available on this machine.
			gpu_count = torch.cuda.device_count()

		# Note: we use the argparse-based interface so that all defaults are preserved.
		parser = create_parser()
		common_args = [
			"--checkpoint_dir", self.checkpoint_path or join(COSMOS_PREDICT1_ROOT, "checkpoints"),
			"--video_save_name=", # Empty string
			"--video_save_folder", join(COSMOS_PREDICT1_ROOT, "outputs"),
			"--trajectory", "none",
			"--prompt=", # Empty string
			"--negative_prompt=", # Empty string
			"--offload_prompt_upsampler",
			"--disable_prompt_upsampler",
			"--disable_guardrail",
			"--num_gpus", str(gpu_count),
			"--guidance", "1.0",
			"--num_video_frames", "121",
			"--foreground_masking",
		]
		args = parser.parse_args(common_args)

		if gpu_count == 1:
			self.model = Gen3cPersistentModel(args)
		else:
			log.info(f"Loading Cosmos-Predict1 inference model on {gpu_count} GPUs.")
			self.model = MultiGPUInferenceAR(gpu_count, cosmos_variant="predict1", args=args)

		# Since the model may require overlap of inference batches,
		# we save previous inference poses so that we can provide any number of
		# previous camera poses when starting the next inference batch.
		# TODO: ensure some kind of ordering?
		self.pose_history_w2c: list[np.array] = []
		self.intrinsics_history: list[np.array] = []

		self.default_focal_length = (338.29, 338.29)
		self.default_principal_point = (0.5, 0.5)
		self.aabb_min = np.array([-16, -16, -16])
		self.aabb_max = np.array([16, 16, 16])


	def inference_resolution(self) -> list[tuple[int, int]] | None:
		"""The supported inference resolutions, or None if any resolution is supported."""
		return [(1280, 704),]


	@override
	def max_frames_per_request(self) -> int:
		# Not actually tested, but anyway we can expect autoregressive
		# generation to go wrong earlier than this.
		return self.model.frames_per_batch * 100

	@override
	def default_framerate(self) -> float:
		return 24.0

	def cleanup(self):
		if isinstance(self.model, MultiGPUInferenceAR):
			self.model.cleanup()

	@override
	def metadata(self) -> dict:
		result = super().metadata()
		result["model_name"] = "CosmosModel"
		return result


if __name__ == "__main__":
	model = CosmosModel()
