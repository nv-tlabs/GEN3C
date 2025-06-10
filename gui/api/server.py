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

from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
import os
import pickle
import traceback

from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import Response
import imageio.v3 as imageio
import numpy as np

from api_types import CompressedSeedingRequest
from server_base import InferenceModel
from server_cosmos import CosmosModel


# ------------------------------

@dataclass
class ServerSettings():
	"""
	Note: we use a dataclass + env variables because we can't
	easily pass command line arguments through the `fastapi` launcher.
	"""
	model: str = os.environ.get("GEN3C_MODEL", "cosmos-predict1")
	checkpoint_path: str | None = os.environ.get("GEN3C_CKPT_PATH")
	data_path: str | None = os.environ.get("GEN3C_DATA_PATH")

	#: Additional latency to add to any inference request, in milliseconds.
	inference_latency: int = int(os.environ.get("GEN3C_INFERENCE_LATENCY", 0))

	#: Number of inference results to keep in cache.
	#: This may be useful when multiple requests are in flight and the user hasn't
	#: retrieved the results yet.
	inference_cache_size: int = int(os.environ.get("GEN3C_INFERENCE_CACHE_SIZE", 15))

	#: Number of GPUs to use for inference. Leave at 0 to automatically select
	#: based on available hardware.
	gpu_count: int = int(os.environ.get("GEN3C_GPU_COUNT", 0))


settings = ServerSettings()
model: InferenceModel | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
	global model

	model_name = settings.model.lower()
	if model_name in ("cosmos", "cosmos-predict1"):
		cls = CosmosModel
	else:
		raise ValueError(f"Unsupported model type: '{settings.model}'")

	model = cls(checkpoint_path=settings.checkpoint_path,
				data_path=settings.data_path,
				fake_delay_ms=settings.inference_latency,
				inference_cache_size=settings.inference_cache_size,
				gpu_count=settings.gpu_count)

	# --- Startup code
	# Pre-render at least one image to make sure everything is running
	if not model.requires_seeding():
		await model.make_test_image()

	yield

	# --- Shutdown code
	model.cleanup()
	del model

app = FastAPI(lifespan=lifespan)
logger = logging.getLogger('uvicorn.error')


# ------------------------------

def get_bool_query_param(request: Request, name: str, default: bool) -> bool:
	b_str = request.query_params.get(name, "1" if default else "0")
	return b_str.lower() in ("1", "true", "yes", "")


@app.post("/request-inference", response_class=Response, response_model=None)
async def request_inference(request: Request):
	"""
	Start a new asynchronous inference job.
	"""
	sync = get_bool_query_param(request, "sync", default=False)
	req: bytes = await request.body()
	req = pickle.loads(req)

	try:
		if sync:
			result = await model.request_inference_sync(req)
			return Response(content=pickle.dumps(result),
							media_type="application/octet-stream")
		else:
			model.request_inference(req)
	except Exception as e:
		logging.error("Inference request failed with exception:"
					  f"\n{e}\n{traceback.format_exc()}")
		return Response(str(e), status_code=400)

	return Response("Request accepted.", status_code=202)


@app.post("/seed-model", response_class=Response, response_model=None)
async def seed_model(request: Request):
	"""
	Start a new asynchronous inference job.
	"""
	sync = get_bool_query_param(request, "sync", default=False)
	req: bytes = await request.body()
	req = pickle.loads(req)

	if isinstance(req, CompressedSeedingRequest):
		req.decompress()

	try:
		# There isn't really anything async about the seeding request being done on the server
		# so far, so we just await. This could be changed in the future.
		result = await model.seed_model(req)
	except Exception as e:
		logging.error(f"Seeding request failed with exception:"
					  f"\n{e}\n{traceback.format_exc()}")
		return Response(str(e), status_code=400)

	# return Response("Seeding request accepted.", status_code=(200 if sync else 202))
	return Response(content=pickle.dumps(result),
					media_type="application/octet-stream")


@app.get("/inference-result", response_class=Response, response_model=None)
async def inference_results_or_none(request_id: str):
	try:
		result = model.inference_result_or_none(request_id)
	except Exception as e:
		# TODO: try to differentiate the status codes (doesn't exist, inference failed, etc)
		logging.error(f"Inference results request failed with exception:"
					  f"\n{e}\n{traceback.format_exc()}")
		return Response(str(e), status_code=500)

	if result is None:
		return Response(content="Result not ready",
						status_code=503)
	else:
		return Response(content=pickle.dumps(result),
						media_type="application/octet-stream")


@app.get("/image", response_class=Response)
def latest_rgb(format: str = "jpg"):
	# We return the data as pickled bytes to avoid the JSON serialization / deserialization overhead.
	image = model.get_latest_rgb()
	if image is None:
		return Response(content="No image available yet.", status_code=404)

	if format == "pickle":
		content = pickle.dumps(
			{
				"image": image,
			}
		)
		return Response(content=content, media_type="application/octet-stream")

	elif format in ("jpg", "png"):
		image = image.copy()
		# Allow alpha channel to be omitted for faster transfers
		if image.shape[-1] == 3:
			image = np.concatenate([
				image,
				np.ones((*image.shape[:2], 1))
			], axis=-1)

		if image.dtype != np.uint8:
			# TODO: proper handling of gamma compression, etc
			image[:, :, :3] = np.power(image[:, :, :3], 1 / 2.2) * 255
			image[:, :, 3] = image[:, :, 3] * 255
		if format != "png":
			image = image[:, :, :3]

		content = imageio.imwrite(uri="<bytes>", image=image.astype(np.uint8), extension="." + format)
		return Response(content=content, media_type=f"image/{format}")

	else:
		return Response(f"Unsupported image format: {format}", status_code=400)


@app.get("/metadata")
def metadata():
	return model.metadata()
