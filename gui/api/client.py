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

import argparse
import asyncio
import code
from copy import deepcopy
from datetime import datetime
import glob
import os
from os.path import realpath, dirname, join
import pickle
import subprocess
import sys
import time

import cv2
import httpx
import numpy as np
import pyexr
import tqdm

ROOT_DIR = realpath(dirname(dirname(__file__)))
DATA_DIR = join(ROOT_DIR, "data")

sys.path.append(join(ROOT_DIR, "scripts"))

# Search for pyngp in the build folder.
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(os.path.join(ROOT_DIR, "build*", "**/*.pyd"), recursive=True)]
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(os.path.join(ROOT_DIR, "build*", "**/*.so"), recursive=True)]

import pyngp as ngp
from pyngp import tlog

from api_types import SeedingRequest, CompressedSeedingRequest, SeedingResult, \
					  InferenceRequest, InferenceResult, CompressedInferenceResult, \
					  RequestState, PendingRequest
from httpx_utils import httpx_request
from v2v_utils import load_v2v_seeding_data, ensure_alpha_channel, srgb_to_linear



def repl(testbed):
	print("-------------------\npress Ctrl-Z to return to gui\n---------------------------")
	code.InteractiveConsole(locals=locals()).interact()
	print("------- returning to gui...")


def open_file_with_default_app(video_path: str) -> None:
	"""Open the saved video file with the default video player application."""
	try:
		if sys.platform == "win32":
			# Windows
			os.startfile(video_path)
		else:
			# Avoid venv, etc interfering with the application that will open.
			env = os.environ.copy()
			for k in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR", "LD_LIBRARY_PATH"):
				if k in env:
					del env[k]
			if sys.platform == "darwin":
				# macOS
				subprocess.run(["open", video_path], check=True, env=env)
			else:
				# Linux, etc.
				subprocess.run(["xdg-open", video_path], check=True, env=env)
	except Exception as e:
		tlog.error(f"Failed to open video file: {e}")


class Gen3cClient():
	def __init__(
		self,
		files: list[str],
		host: str,
		port: int,
		width: int = 1920,
		height: int = 1080,
		vr: bool = False,
		request_latency_ms: int = 100,
		inference_resolution: tuple[int, int] = (1920, 1080),
		# max_pending_requests: int = 2,
		max_pending_requests: int = 1,
		request_timeous_seconds: float = 1000,
		seed_max_frames: int | None = None,
		seed_stride: int = 1,
		output_dir: str | None = None,
	):
		self.url = f"http://{host}:{port}"
		self.client_id = f"gen3c{os.getpid()}"
		self.request_latency_ms = request_latency_ms
		self.inference_resolution = inference_resolution
		self.max_pending_requests = max_pending_requests
		self.req_timeout_s = request_timeous_seconds
		self.seed_max_frames = seed_max_frames
		self.seed_stride = seed_stride

		testbed = ngp.Testbed(ngp.TestbedMode.Gen3c)
		testbed.root_dir = ROOT_DIR
		testbed.set_gen3c_cb(self.gui_callback)
		testbed.file_drop_callback = self.file_drop_callback

		if output_dir is not None:
			os.makedirs(output_dir, exist_ok=True)
		else:
			output_dir = join(ROOT_DIR, "outputs")
		testbed.gen3c_output_dir = output_dir
		testbed.video_path = join(output_dir, "gen3c_%Y-%m-%d_%H-%M-%S.mp4")

		# --- Check metadata from server to ensure compatibility
		testbed.reproject_visualize_src_views = False
		testbed.render_aabb.min = np.array([-16.0, -16.0, -16.0])
		testbed.render_aabb.max = np.array([16.0, 16.0, 16.0])
		try:
			tlog.info(f"Requesting metadata from server {host}:{port}")
			metadata = self.request_metadata_sync()
			testbed.render_aabb.min = np.array(metadata["aabb_min"]).astype(np.float32)
			testbed.render_aabb.max = np.array(metadata["aabb_max"]).astype(np.float32)
			testbed.aabb = ngp.BoundingBox(testbed.render_aabb.min, testbed.render_aabb.max)
			testbed.gen3c_info = f"Connected to server {host}:{port}, model name: {metadata.get('model_name')}"

			model_inference_res: list[tuple[int, int]] | None = metadata.get("inference_resolution")
			if model_inference_res is not None:
				for supported_res in model_inference_res:
					if tuple(supported_res) == self.inference_resolution:
						break
				else:
					r = tuple(model_inference_res[0])
					tlog.warning(f"Client inference resolution {self.inference_resolution} is not"
						         f" supported by the inference server, adopting resolution {r} instead.")
					self.inference_resolution = r
			testbed.camera_path.render_settings.resolution = self.inference_resolution

			testbed.gen3c_inference_is_connected = True
			testbed.gen3c_render_with_gen3c = True

		except httpx.ConnectError as e:
			# The metadata-based setup happens only once at startup. Since we failed to
			# get the metadata from the server, it's easier to just raise and exit here.
			raise RuntimeError(
				f"Connection error! Make sure the server was started at: {host}:{port}\n{e}"
			) from e

		testbed.camera_path.render_settings.fps = metadata.get("default_framerate") or 24.0
		self.min_frames_per_request: int = metadata.get("min_frames_per_request", 1)
		self.max_frames_per_request: int = metadata.get("max_frames_per_request", 1)
		if self.min_frames_per_request > 1:
			# Set default render settings such that the model can generate it
			# in a single batch exactly.
			testbed.camera_path.default_duration_seconds = \
				self.min_frames_per_request / testbed.camera_path.render_settings.fps
			testbed.camera_path.duration_seconds = testbed.camera_path.default_duration_seconds

		# Expected time that the model will take to generate each frame, in seconds
		self.inference_time_per_frame: float = metadata.get("inference_time_per_frame", 0.0)
		# Don't automatically request new frames all the time if inference is slow
		testbed.gen3c_auto_inference &= (self.inference_time_per_frame < 1.0)

		self.seeding_pending: bool = False
		self.model_requires_seeding: bool = metadata.get("requires_seeding", True)
		if self.model_requires_seeding:
			testbed.gen3c_info += "\nThis model requires seeding data."

		# Pick a sensible GUI resolution depending on arguments.
		sw = width
		sh = height
		while sw * sh > 1920 * 1080 * 4:
			sw = int(sw / 2)
			sh = int(sh / 2)

		testbed.init_window(sw, sh)
		if vr:
			testbed.init_vr()
		self.testbed: ngp.Testbed = testbed

		self.lens = ngp.Lens()
		self.lens.mode = ngp.LensMode.Perspective

		self.client = httpx.AsyncClient()

		self.last_request_id: int = 0
		self.start_t: float = None
		self.last_request_t: float = None
		self.pending_requests: dict[str, PendingRequest] = {}

		# Handle files given as command-line arguments.
		if files:
			self.file_drop_callback(files)



	async def run(self):
		testbed = self.testbed

		self.start_t = time.monotonic()
		self.last_request_t = time.monotonic()
		# TODO: any way to make the rendering itself async? (pybind11 support?)
		while testbed.frame():
			# --- At each frame
			if testbed.want_repl():
				repl(testbed)

			if self.model_requires_seeding and self.seeding_pending and self.testbed.gen3c_seed_path:
				tlog.info(f"Loading seeding data with path: {self.testbed.gen3c_seed_path}")
				# Load the seeding data.
				seed_req = self.load_seeding_data(self.testbed.gen3c_seed_path)
				if seed_req is not None:
					self.adapt_view_to_cameras(seed_req.cameras_to_world)
					# Send the seeding request over to the server (could be a slow upload).
					self.send_seeding_request(seed_req)
				self.seeding_pending = False

			# Give coroutines a chance to run (especially if there are pending HTTP requests).
			# This is essentially a "yield".
			# TODO: how can we sleep only for the minimum needed time?
			#       Probably we would need to request the `testbed`'s frame in an
			#       async way as well? Something like:
			#    await testbed.frame()
			await asyncio.sleep(0.003 if self._transfer_in_progress() else 0.0001)

			# Check pending inference requests
			self.get_request_results()

			# New inference request
			# TODO: if there are too many pending requests, cancel the oldest one
			#       instead of continuing to wait.
			now = time.monotonic()
			if ((1000 * (now - self.last_request_t) > self.request_latency_ms)
				and testbed.gen3c_auto_inference
				and testbed.is_rendering
				and len(self.pending_requests) < self.max_pending_requests):

				self.request_frames()

	def get_request_results(self):
		to_remove = set()
		for req_id, state in self.pending_requests.items():

			if state.state in (RequestState.FAILED, RequestState.COMPLETE):
				# Cleanup requests that are done one way or another
				to_remove.add(req_id)

			elif state.state == RequestState.REQUEST_PENDING:
				# Before checking the results, we wait for the inference request to have
				# been received by the server at least.
				self.testbed.gen3c_inference_info = f"Waiting for inference request {req_id} to be received by the server..."
				continue

			elif state.state == RequestState.REQUEST_SENT:
				# Server has received the inference request, we should now start checking results
				def on_result_received(result: InferenceResult | None,
									   response: httpx.Response, failed: bool = False):
					if failed:
						tlog.error(f"Results request for inference {req_id} failed!\n"
								   f"{response.content}")
						self.testbed.gen3c_inference_info = f"Error: {response.content}"
						state.state = RequestState.FAILED
						state.task = None
						return

					if result is None:
						# Result not ready yet, check again soon
						state.state = RequestState.REQUEST_SENT
						state.task = None
						return

					# Actual result received!
					assert isinstance(result, InferenceResult)
					state.state = RequestState.COMPLETE
					self.testbed.gen3c_inference_info = ""

					tlog.success(f"Received results {req_id}: took {result.runtime_ms:.1f} ms to generate.")

					need_frames = self.testbed.gen3c_display_frames or self.testbed.gen3c_save_frames
					result.trim_to_original_frame_count()
					if isinstance(result, CompressedInferenceResult):
						# Save the compressed video straight to disk
						video_path = datetime.now().strftime(self.testbed.video_path)
						result.save_images(video_path)
						tlog.success(f"[+] Wrote generated video to: {video_path}")

						tlog.info(f"Opening file with default application: {video_path}")
						open_file_with_default_app(video_path)
						if need_frames:
							result.decompress()

					# Add all received frames to the viewer.
					if self.testbed.gen3c_save_frames:
						os.makedirs(self.testbed.gen3c_output_dir, exist_ok=True)

					view_ids = set(self.testbed.src_view_ids())
					for res_i in range(len(result)):
						if not need_frames:
							continue

						# Only display the result if we don't already have it shown
						res_id = result.result_ids[res_i]

						if (res_id is not None) and (res_id in view_ids):
							tlog.debug(f"Skipping result since id {res_id} is already displayed")
							continue

						# Allow alpha channel to be omitted for faster transfers
						image = ensure_alpha_channel(result.images[res_i, ...])

						if self.testbed.gen3c_save_frames:
							safe_res_id = (res_id or f"{res_i:04d}").replace(":", "_")
							fname = join(self.testbed.gen3c_output_dir,
										 f"rgb_{safe_res_id}.exr")
							pyexr.write(fname, image)

							fname = join(self.testbed.gen3c_output_dir,
										 f"depth_{safe_res_id}.exr")
							pyexr.write(fname, result.depths[res_i, ...].astype(np.float32))
							tlog.success(f"[+] Wrote inference result to: {fname}")

						if self.testbed.gen3c_display_frames:
							has_valid_depth = np.any(np.isfinite(result.depths[res_i, ...]))
							if has_valid_depth:
								self.testbed.add_src_view(
									result.cameras_to_world[res_i, ...],
									result.focal_lengths[res_i][0],
									result.focal_lengths[res_i][1],
									result.principal_points[res_i][0],
									result.principal_points[res_i][1],
									self.lens,
									image,
									result.depths[res_i, ...],
									result.timestamps[res_i],
									is_srgb=True,
								)
								self.testbed.reset_accumulation(reset_pip=True)
								tlog.info(f"Added {res_id}[{res_i}] to viewer")
							else:
								tlog.debug(f"Not adding {res_id}[{res_i}] to viewer because it has no valid depth."
										   " Only keyframes (last frame of each batch) typically have valid depth.")

					# Don't display more than 8 views at once by default to avoid
					# slowing down the rendering too much.
					self.set_max_number_of_displayed_views(8)

				tlog.debug(f"Checking results of request {req_id}...")
				state.state = RequestState.RESULT_PENDING
				state.task = self._get_inference_results(req_id, on_result_received)

			elif state.state == RequestState.RESULT_PENDING:
				# We already sent a request to check on the results, let's wait until
				# a response comes back (through the `on_result_received` cb).
				if self.testbed.gen3c_inference_progress < 0:
					# Only show the spinner if downloading the results hasn't started yet.
					spinner = "|/-\\"[int(4 * time.time()) % 4]
					self.testbed.gen3c_inference_info = f"[{spinner}] Waiting for server to complete inference..."
				pass


		for k in to_remove:
			del self.pending_requests[k]
		self.testbed.camera_path.rendering = len(self.pending_requests) > 0

	# ----------

	def request_metadata_sync(self) -> InferenceResult:
		# Synchronous request (no need to `await`)
		return httpx_request("get", self.url + "/metadata", timeout=self.req_timeout_s).json()


	def request_frames(self, sync: bool = False) -> asyncio.Task | InferenceResult:
		# The user wants a certain number of frames, but the model can only generate
		# `self.min_frames_per_request` per request. Pad to get there.
		n_desired_frames = int(np.ceil(self.testbed.camera_path.duration_seconds
							           * self.testbed.camera_path.render_settings.fps))
		n_frames_padded = max(
			int(np.ceil(n_desired_frames / self.min_frames_per_request) * self.min_frames_per_request),
			self.min_frames_per_request
		)
		self.testbed.gen3c_inference_info = (
			f"Requesting {n_desired_frames} frames ({n_frames_padded} total with padding, "
			f"model has min batch size {self.min_frames_per_request})."
		)
		tlog.info(self.testbed.gen3c_inference_info)
		# TODO: enforce `max_frames_per_request` from the server, too (with a clear error message)
		now = time.monotonic()

		cameras_to_world = np.repeat(self.testbed.camera_matrix[None, ...],
									 repeats=n_desired_frames, axis=0)

		# By default, use the preview camera focal length.
		# We assume square pixels, so horizontal and vertical focal lengths are equal.
		default_focal_length = self.testbed.relative_focal_length * self.inference_resolution[self.testbed.fov_axis]
		focal_lengths = np.array([default_focal_length] * n_desired_frames)

		match self.testbed.gen3c_camera_source:
			case ngp.Gen3cCameraSource.Fake:
				# --- Camera movement: fake based on fixed translation and rotation speeds
				counter = np.arange(n_desired_frames)[..., None]

				if np.any(self.testbed.gen3c_rotation_speed != 0):
					angles = counter * self.testbed.gen3c_rotation_speed[None, ...]
					alphas = angles[:, 0]
					betas = angles[:, 1]
					gammas = angles[:, 2]

					# TODO: nicer way to build the rotation matrix
					fake_rotation = np.tile(np.eye(3, 3)[None, ...], (n_desired_frames, 1, 1))
					fake_rotation[:, 0, 0] = np.cos(betas) * np.cos(gammas)
					fake_rotation[:, 0, 1] = (
						np.sin(alphas) * np.sin(betas) * np.cos(gammas)
						- np.cos(alphas) * np.sin(gammas)
					)
					fake_rotation[:, 0, 2] = (
						np.cos(alphas) * np.sin(betas) * np.cos(gammas)
						+ np.sin(alphas) * np.sin(gammas)
					)

					fake_rotation[:, 1, 0] = np.cos(betas) * np.sin(gammas)
					fake_rotation[:, 1, 1] = (
						np.sin(alphas) * np.sin(betas) * np.sin(gammas)
						+ np.cos(alphas) * np.cos(gammas)
					)
					fake_rotation[:, 1, 2] = (
						np.cos(alphas) * np.sin(betas) * np.sin(gammas)
						- np.sin(alphas) * np.cos(gammas)
					)

					fake_rotation[:, 2, 0] = -np.sin(betas)
					fake_rotation[:, 2, 1] = np.sin(alphas) * np.cos(betas)
					fake_rotation[:, 2, 2] = np.cos(alphas) * np.cos(betas)

					cameras_to_world[:, :3, :3] @= fake_rotation

				if np.any(self.testbed.gen3c_translation_speed != 0):
					fake_translation = counter * self.testbed.gen3c_translation_speed[None, ...]
					cameras_to_world[:, :, 3] += fake_translation

			case ngp.Gen3cCameraSource.Viewpoint:
				# --- Camera movement: based on the current viewpoint + predicted movement
				tlog.error("Not implemented: Gen3C camera movement source: Viewpoint")
				return

			case ngp.Gen3cCameraSource.Authored:
				# --- Camera movement: based on the current authored camera path
				keyframes = [
					self.testbed.camera_path.eval_camera_path(t)
					for t in np.linspace(0, 1, n_desired_frames, endpoint=True)
				]
				cameras_to_world = [
					keyframe.m()[None, ...]
					for keyframe in keyframes
				]
				cameras_to_world = np.concatenate(cameras_to_world, axis=0)

				focal_lengths = np.stack([
					[
						ngp.fov_to_focal_length(self.inference_resolution[self.testbed.fov_axis], keyframe.fov)
					] * 2
					for keyframe in keyframes
				], axis=0)

			case _:
				raise ValueError("Unsupported Gen3C camera movement source:",
								 self.testbed.gen3c_camera_source)
		t0 = now - self.start_t
		timestamps = [t0 + i * self.inference_time_per_frame
					  for i in range(n_desired_frames)]

		request_id = f"{self.client_id}:{self.last_request_id + 1}"

		tlog.debug(f"Creating new request {request_id}")
		req = InferenceRequest(
			request_id=request_id,
			timestamps=np.array(timestamps),
			cameras_to_world=cameras_to_world,
			focal_lengths=focal_lengths,
			principal_points=np.array([self.testbed.screen_center] * n_desired_frames),
			resolutions=np.array([self.inference_resolution] * n_desired_frames),
			framerate=self.testbed.camera_path.render_settings.fps,
			# If we don't need to display the generated frames, we can save time
			# by not estimating & downloading depth maps.
			return_depths=self.testbed.gen3c_display_frames,
			video_encoding_quality=self.testbed.camera_path.render_settings.quality,
			show_cache_renderings=self.testbed.gen3c_show_cache_renderings,
		)
		# Add any necessary padding to the request to match the server's batch size.
		req.pad_to_frame_count(n_frames_padded)

		# Send an inference request to the server and add it to the
		# list of pending requests.
		self.request_frame(req, sync=sync)

		tlog.info("Waiting for inference results (this may take a while)...")
		self.last_request_t = now
		self.last_request_id += 1


	def request_frame(self, req: InferenceRequest, sync: bool = False) -> asyncio.Task | InferenceResult:
		qp = "?sync=1" if sync else ""
		url = self.url + "/request-inference" + qp
		data = pickle.dumps(req)

		def req_done_cb(task_or_res: asyncio.Task | httpx.Response) -> None:
			if sync:
				res: httpx.Response = task_or_res
			else:
				try:
					res: httpx.Response = task_or_res.result()
				except RuntimeError as e:
					tlog.error(f"Inference request task failed!\n{e}")

			if res.status_code != 202:
				tlog.error(f"Inference request failed!\n{res.content}")

			if sync:
				return pickle.loads(res.content)
			else:
				if req.request_id not in self.pending_requests:
					tlog.error(f"Inference request {req.request_id} was created on the server,"
							   f" but it is not part of our pending requests"
							   f" (pending: {list(self.pending_requests.keys())})")

				state = self.pending_requests[req.request_id]
				state.state = RequestState.REQUEST_SENT
				state.task = None

		task_or_res = httpx_request(
			"post", url, data=data, timeout=self.req_timeout_s,
			async_client=(None if sync else self.client),
			callback=req_done_cb
		)
		if not sync:
			self.pending_requests[req.request_id] = PendingRequest(
				request_id=req.request_id,
				state=RequestState.REQUEST_PENDING,
				task=task_or_res,
			)
		return task_or_res


	def _get_inference_results(self, request_id: str, on_result_received: callable) -> asyncio.Task:
		def task_cb(task):
			# Hide the progress bar (regardless of success or failure)
			self.testbed.gen3c_inference_progress = -1.0

			try:
				res: httpx.Response = task.result()
			except RuntimeError as e:
				tlog.error(f"Results request task for inference {request_id} failed!\n{e}")

			if res.status_code == 503:
				# Result not ready yet, wait some more
				on_result_received(result=None, response=res)
				return
			elif res.status_code != 200:
				# Result failed, we shouldn't retry further
				on_result_received(result=None, response=res, failed=True)
				return

			# Result ready
			on_result_received(pickle.loads(res.content), response=res)
			return

		def progress_cb(progress: float, bar: tqdm, **kwargs):
			total_mb = bar.total / (1024 * 1024)
			self.testbed.gen3c_inference_info = f"Downloading inference results ({total_mb:.1f} MB)"
			self.testbed.gen3c_inference_progress = progress

		return httpx_request(
			"get",
			self.url + f"/inference-result?request_id={request_id}",
			# Waiting for the model to finish inference can be very long,
			# especially for single-GPU inference.
			timeout=10 * self.req_timeout_s,
			progress=True,
			desc=f"Inference results for {request_id}",
			async_client=self.client,
			callback=task_cb,
			progress_callback=progress_cb
		)

	# ----------

	def load_seeding_data(self, seeding_data_path: str, display: bool = True,
						  normalize_cameras: bool = False) -> SeedingRequest:

		if not os.path.exists(seeding_data_path):
			tlog.error(f"Cannot seed with invalid path: \"{seeding_data_path}\"")
			return None
		tlog.info(f"Seeding model from \"{seeding_data_path}\"")

		req = load_v2v_seeding_data(seeding_data_path, max_frames=self.seed_max_frames,
								    frames_stride=self.seed_stride)

		if normalize_cameras:
			if isinstance(req, CompressedSeedingRequest):
				raise NotImplementedError("Normalizing cameras not implemented for compressed seeding data")

			# Post-process the cameras so that they are centered at (0.5, 0.5, 0.5)
			# and so that they fit within a reasonable scale.
			current_origins = req.cameras_to_world[:, :3, 3]
			current_center = np.mean(current_origins, axis=0)
			current_scale = np.mean(np.linalg.norm(current_origins, axis=1))
			# TODO: robust scale estimation using the median depth as well
			if req.depths is not None:
				median_depth = np.nanmedian(req.depths)
				current_scale = max(current_scale, median_depth)

			# tlog.debug(f"Current scale: {current_scale}")

			if current_scale != 0.0:
				normalized_origins = (current_origins - current_center) / current_scale

				new_center = np.array([0.5, 0.5, 0.5], dtype=np.float32)
				# aabb_scale = np.linalg.norm(self.testbed.render_aabb.max - self.testbed.render_aabb.min)
				# new_scale = aabb_scale / 4
				new_scale = 1.0
				req.cameras_to_world[:, :3, 3] = (normalized_origins * new_scale) + new_center

				# Rescale the depth values by the same
				req.depths *= new_scale / current_scale
			# TODO: retain this information so that we can undo the transform when
			# communicating with the server or saving stuff out.

		if display and (req.depths is not None):
			# If there's not depth data available at this point, we'll download it from the server
			# when seeding is done, and display the frames then.
			self.display_seeding_data(req, save_frames=self.testbed.gen3c_save_frames)

		return req


	def display_seeding_data(self, req: SeedingRequest, res: SeedingResult | None = None,
							 save_frames: bool = False) -> None:
		self.testbed.clear_src_views()

		if isinstance(req, CompressedSeedingRequest):
			# Since the de-compression is done inline, we make sure not to
			# populate uncompressed data in the request before sending it over.
			req = deepcopy(req)
			req.decompress()

		images = req.images
		depths = req.depths
		if res is not None:
			# Adopt extrinsics and intrinsics from the server, the model might
			# have estimated them better than our hardcoded guess.
			focal_lengths = res.focal_lengths.copy()
			cameras_to_world = res.cameras_to_world
			principal_points = res.principal_points

			if res.depths is not None:
				# TODO: the depth estimated by the server may have a completely different scale.
				depths = res.depths
				if res.depths.shape[1:] != images.shape[1:3]:
					# Depth prediction took place on the server at a different resolution,
					# let's resize the RGB images to match.
					tlog.debug(f"Resizing seeding images for display to match depth resolution {depths.shape[1:3]}")
					resized = []
					for i in range(len(req)):
						resized.append(
							cv2.resize(images[i, ...], (depths.shape[2], depths.shape[1]),
									   interpolation=cv2.INTER_CUBIC)
						)
						# Let's assume that the inference server already adjusted the intrinsics
						# to match the requested inference resolution.
						# focal_lengths[i, 0] *= depths.shape[2] / images.shape[2]
						# focal_lengths[i, 1] *= depths.shape[1] / images.shape[1]
					images = np.stack(resized, axis=0)
		else:
			focal_lengths = req.focal_lengths.copy()
			cameras_to_world = req.cameras_to_world
			principal_points = req.principal_points


		if save_frames:
			os.makedirs(self.testbed.gen3c_output_dir, exist_ok=True)
		for seed_i in range(len(req)):
			res_id = f"seeding_{seed_i:04d}"
			image = ensure_alpha_channel(images[seed_i, ...])

			if save_frames:
				safe_res_id = res_id
				fname = join(self.testbed.gen3c_output_dir, f"rgb_{safe_res_id}.exr")
				pyexr.write(fname, image)

				if depths is not None:
					fname = join(self.testbed.gen3c_output_dir, f"depth_{safe_res_id}.exr")
					pyexr.write(fname, depths[seed_i, ...].astype(np.float32))
					tlog.success(f"[+] Wrote seeding frame to: {fname}")

			if depths is None:
				# Still no depth values available, cannot display
				continue

			self.testbed.add_src_view(
				cameras_to_world[seed_i, ...],
				focal_lengths[seed_i][0],
				focal_lengths[seed_i][1],
				principal_points[seed_i][0],
				principal_points[seed_i][1],
				self.lens,
				image,
				depths[seed_i, ...],
				# TODO: seeding request could also have timestamps
				seed_i * 1 / 30,
				is_srgb=True,
			)
			tlog.success(f"[+] Displaying seeding view: {res_id}")

		tlog.info(f"Setting camera path from seeding view.")
		# First, initialize the camera path from all seeding view.
		self.set_max_number_of_displayed_views(len(req))
		self.testbed.init_camera_path_from_reproject_src_cameras()
		# Then, limit the number of displayed views so that rendering doesn't slow down too much.
		self.set_max_number_of_displayed_views(8)
		self.testbed.reset_accumulation(reset_pip=True)


	def send_seeding_request(self, req: SeedingRequest, sync: bool = False) -> asyncio.Task | None:
		"""
		Note: we do seeding requests synchronously by default so that we don't have to implement
		eager checking, etc.
		"""

		qp = "?sync=1" if sync else ""
		url = self.url + "/seed-model" + qp
		depth_was_missing = (req.depths is None)

		def req_done_cb(task_or_res: asyncio.Task | httpx.Response) -> None:
			# Hide the progress bar (regardless of success or failure)
			self.testbed.gen3c_seeding_progress = -1.0
			if sync:
				res: httpx.Response = task_or_res
			else:
				try:
					res: httpx.Response = task_or_res.result()
				except RuntimeError as e:
					tlog.error(f"Seeding request task failed!\n{e}")
					return

			if res.status_code >= 300:
				tlog.error(f"Seeding request failed!\n{res.content}")
				return None

			if depth_was_missing:
				response: SeedingResult = pickle.loads(res.content)
				self.display_seeding_data(req, res=response, save_frames=self.testbed.gen3c_save_frames)

			message = "Model seeded."
			self.testbed.gen3c_info = "\n".join([
				self.testbed.gen3c_info.split("\n")[0],
				message
			])
			tlog.success(message)

		def progress_cb(progress: float, **kwargs):
			self.testbed.gen3c_seeding_progress = progress

		if not isinstance(req, CompressedSeedingRequest):
			req = req.compress()

		data = pickle.dumps(req)
		try:
			progress_direction = "both" if depth_was_missing else "auto"
			return httpx_request("post", url, data=data, timeout=self.req_timeout_s,
								 progress=True, progress_direction=progress_direction,
								 desc="Seeding",
								 async_client=(None if sync else self.client),
								 callback=req_done_cb,
								 progress_callback=progress_cb)
		except (httpx.TimeoutException, httpx.ConnectError) as e:
			tlog.error(f"Seeding request failed (timeout or connection error)!\n{e}")
			return None

	# ----------

	def set_max_number_of_displayed_views(self, n_views: int) -> None:
		tlog.info(f"Setting max number of displayed views to {n_views}")
		# Jump to the last view.
		self.testbed.reproject_max_src_view_index = min(self.testbed.reproject_src_views_count(), n_views)

	def _transfer_in_progress(self) -> bool:
		return (self.testbed.gen3c_inference_progress >= 0.0) or (self.testbed.gen3c_seeding_progress >= 0.0)

	# ----------

	def adapt_view_to_cameras(self, cameras_to_world: np.ndarray,
							  go_to_default_camera: bool = True) -> None:
		"""
		Analyzes the given set of cameras, and tries to adapt the current
		up vector, default camera pose, etc to match.

		Note: this hasn't been tested very thoroughly yet and could easily
		do the wrong thing depending on the inputs.
		"""
		assert cameras_to_world.shape[1:] == (3, 4)

		# --- Up vector
		# Average of the cameras' individual up vectors, snapped to an axis.
		mean_up = np.mean(cameras_to_world[:, :3, 1], axis=0)
		up_axis = np.argmax(np.abs(mean_up))
		up = np.zeros((3,), dtype=np.float32)
		up[up_axis] = -np.sign(mean_up[up_axis])
		self.testbed.up_dir = up

		# --- Default camera pose
		default_c2w = cameras_to_world[0, :3, :]

		# Note: `default_camera` is technically a 4x3 camera, but the bindings
		# expose it as a 3x4 matrix, so we can set it as normal here.
		self.testbed.default_camera = default_c2w
		tlog.debug(f"Based on the seeding data, setting up dir to {self.testbed.up_dir}"
				   f" and default camera to:\n{self.testbed.default_camera}")

		if go_to_default_camera:
			self.testbed.reset_camera()



	def gui_callback(self, event: str) -> bool:
		match event:
			case "seed_model":
				seed_req = self.load_seeding_data(self.testbed.gen3c_seed_path)
				if seed_req is not None:
					self.adapt_view_to_cameras(seed_req.cameras_to_world)
					self.send_seeding_request(seed_req)
				# "True" means we handled the event, not that seeding was successful.
				return True

			case "request_inference":
				self.request_frames(sync=False)
				return True

			case "abort_inference":
				tlog.info("Aborting inference request...")
				tlog.error("Not implemented yet: aborting an ongoing inference request. Ignoring.")
				return True

		return False

	def file_drop_callback(self, paths: list[str]) -> bool:
		tlog.info(f"Received {len(paths)} file{'s' if len(paths) > 1 else ''} via drag & drop: {paths}")
		for path in paths:
			ext = os.path.splitext(path)[1].lower()
			if os.path.isdir(path) or ext in (".jpg", ".png", ".exr"):
				self.testbed.gen3c_seed_path = path
				self.seeding_pending = True
			elif ext == ".json":
				try:
					self.testbed.load_camera_path(path)
				except RuntimeError as e:
					tlog.error(f"Error loading camera path, perhaps the formata is incorrect?\n\t{e}")
			else:
				tlog.error(f"Don't know how to handle given file: {path}")
		return True



if __name__ == "__main__":
	parser = argparse.ArgumentParser("client.py")
	parser.add_argument("files", nargs="*",
						help="Files to be loaded. Can be a camera path, scene name,"
							 " seed image, or pre-processed video directory.")
	parser.add_argument("--host", default="127.0.0.1")
	parser.add_argument("--port", default=8000)
	parser.add_argument("--request-latency-ms", "--latency", default=250)
	parser.add_argument("--inference-resolution", nargs=2, default=(576, 320))
	parser.add_argument("--vr", action="store_true")
	parser.add_argument("--seed-max-frames", type=int, default=None,
						help="If seeding from a video, maximum number of frames to use.")
	parser.add_argument("--seed-stride", type=int, default=1,
						help="If seeding from a video, number of frames to skip when reading (stride).")
	parser.add_argument("--output-dir", "-o", type=str, default=None,
						help="Directory in which to save the inference results.")
	args = parser.parse_args()

	client = Gen3cClient(**vars(args))
	asyncio.run(client.run())
