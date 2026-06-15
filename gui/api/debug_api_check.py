#!/usr/bin/env python3

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
import pickle
import time

os.environ["GEN3C_API_DEBUG"] = "1"

from fastapi.testclient import TestClient
import numpy as np

from api_serialization import API_MEDIA_TYPE, dumps_api_message, loads_api_message
from api_types import CompressedSeedingRequest, InferenceRequest, InferenceResult, SeedingRequest, SeedingResult
from server import app


HEADERS = {
	"Content-Type": API_MEDIA_TYPE,
	"Accept": API_MEDIA_TYPE,
}

_PICKLE_WAS_DESERIALIZED = False


def _mark_pickle_deserialized():
	global _PICKLE_WAS_DESERIALIZED
	_PICKLE_WAS_DESERIALIZED = True
	return "pickle payload executed"


class PicklePayload:
	def __reduce__(self):
		return (_mark_pickle_deserialized, ())


def make_inference_request(request_id: str, n_frames: int = 2) -> InferenceRequest:
	return InferenceRequest(
		request_id=request_id,
		timestamps=np.linspace(0.0, 1.0, n_frames, dtype=np.float32),
		cameras_to_world=np.zeros((n_frames, 3, 4), dtype=np.float32),
		focal_lengths=np.ones((n_frames, 2), dtype=np.float32),
		principal_points=np.full((n_frames, 2), 0.5, dtype=np.float32),
		resolutions=np.tile([[16, 8]], (n_frames, 1)).astype(np.int32),
		return_depths=True,
	)


def make_seeding_request() -> CompressedSeedingRequest:
	n_frames = 1
	width = 16
	height = 8
	image = np.zeros((n_frames, height, width, 3), dtype=np.float32)
	image[..., 0] = np.linspace(0.0, 1.0, width, dtype=np.float32)
	image[..., 1] = np.linspace(0.0, 1.0, height, dtype=np.float32)[None, :, None]

	req = SeedingRequest(
		request_id="debug-seed",
		cameras_to_world=np.zeros((n_frames, 3, 4), dtype=np.float32),
		focal_lengths=np.ones((n_frames, 2), dtype=np.float32),
		principal_points=np.full((n_frames, 2), 0.5, dtype=np.float32),
		resolutions=np.array([[width, height]], dtype=np.int32),
		images=image,
		depths=None,
		masks=None,
	)
	return req.compress()


def assert_pickle_payload_rejected(client: TestClient, endpoint: str, content_type: str) -> None:
	global _PICKLE_WAS_DESERIALIZED
	_PICKLE_WAS_DESERIALIZED = False

	response = client.post(
		endpoint,
		content=pickle.dumps(PicklePayload()),
		headers={"Content-Type": content_type},
	)

	assert response.status_code == 400, response.text
	assert not _PICKLE_WAS_DESERIALIZED, f"{endpoint} deserialized a pickle body"


def main() -> None:
	with TestClient(app) as client:
		metadata_response = client.get("/metadata")
		assert metadata_response.status_code == 200, metadata_response.text
		metadata = metadata_response.json()
		assert metadata["model_name"] == "DebugInferenceModel"

		for endpoint in ("/request-inference", "/seed-model"):
			assert_pickle_payload_rejected(client, endpoint, "application/octet-stream")
			assert_pickle_payload_rejected(client, endpoint, "application/json")

		sync_req = make_inference_request("debug-sync", n_frames=2)
		sync_response = client.post(
			"/request-inference?sync=1",
			content=dumps_api_message(sync_req),
			headers=HEADERS,
		)
		assert sync_response.status_code == 200, sync_response.text
		sync_result = loads_api_message(sync_response.content, allowed_types=(InferenceResult,))
		assert sync_result.request_id == sync_req.request_id
		assert sync_result.images.shape == (2, 8, 16, 3)
		assert sync_result.depths.shape == (2, 8, 16)

		async_req = make_inference_request("debug-async", n_frames=1)
		async_response = client.post(
			"/request-inference",
			content=dumps_api_message(async_req),
			headers=HEADERS,
		)
		assert async_response.status_code == 202, async_response.text

		for _ in range(20):
			result_response = client.get("/inference-result", params={"request_id": async_req.request_id})
			if result_response.status_code == 200:
				break
			assert result_response.status_code == 503, result_response.text
			time.sleep(0.05)
		assert result_response.status_code == 200, result_response.text
		async_result = loads_api_message(result_response.content, allowed_types=(InferenceResult,))
		assert async_result.request_id == async_req.request_id

		seed_req = make_seeding_request()
		seed_response = client.post(
			"/seed-model",
			content=dumps_api_message(seed_req),
			headers=HEADERS,
		)
		assert seed_response.status_code == 200, seed_response.text
		seed_result = loads_api_message(seed_response.content, allowed_types=(SeedingResult,))
		assert seed_result.request_id == seed_req.request_id
		assert seed_result.depths.shape == (1, 8, 16)

	print("GEN3C API debug check passed")


if __name__ == "__main__":
	main()
