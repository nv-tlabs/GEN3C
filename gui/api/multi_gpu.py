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

from __future__ import annotations

import argparse
import os
import signal

from loguru import logger as log

from v2v_utils import move_to_device, clone_tensors


TORCHRUN_DEFAULT_MASTER_ADDR = 'localhost'
TORCHRUN_DEFAULT_MASTER_PORT = 12355


def _get_inference_class(cosmos_variant: str):
	if cosmos_variant == 'predict1':
		from cosmos_predict1.diffusion.inference.gen3c_persistent import Gen3cPersistentModel
		from cosmos_predict1.utils.distributed import is_rank0
		return Gen3cPersistentModel, is_rank0
	else:
		raise ValueError(f"Unsupported cosmos variant: {cosmos_variant}")


def _inference_worker(rank: int, args: argparse.Namespace,
					  gpu_count: int,
					  cosmos_variant: str,
					  input_queues: 'list[torch.multiprocessing.Queue]',
					  result_queue: 'torch.multiprocessing.Queue',
					  attrs_queue: 'torch.multiprocessing.Queue'):
	"""
	One such function will run, in a separate process, for each GPU.
	Each process loads the model and keeps it in memory.
	"""
	log.debug(f'inference_worker for rank {rank} starting, doing imports now')
	import torch
	import torch.distributed as dist

	InferenceAR, is_tp_cp_pp_rank0 = _get_inference_class(cosmos_variant)
	log.debug(f'inference_worker for rank {rank} done with imports.')

	# The FQDN of the host that is running worker with rank 0; used to initialize the Torch Distributed backend.
	os.environ.setdefault("MASTER_ADDR", TORCHRUN_DEFAULT_MASTER_ADDR)
	# The port on the MASTER_ADDR that can be used to host the C10d TCP store.
	os.environ.setdefault("MASTER_PORT", str(TORCHRUN_DEFAULT_MASTER_PORT))
	# The local rank.
	os.environ["LOCAL_RANK"] = str(rank)
	# The global rank.
	os.environ["RANK"] = str(rank)
	# The rank of the worker group. A number between 0 and max_nnodes. When running a single worker group per node, this is the rank of the node.
	os.environ["GROUP_RANK"] = str(rank)
	# The rank of the worker across all the workers that have the same role. The role of the worker is specified in the WorkerSpec.
	os.environ["ROLE_RANK"] = str(rank)
	# The local world size (e.g. number of workers running locally); equals to --nproc-per-node specified on torchrun.
	os.environ["LOCAL_WORLD_SIZE"] = str(gpu_count)
	# The world size (total number of workers in the job).
	os.environ["WORLD_SIZE"] = str(gpu_count)
	# The total number of workers that was launched with the same role specified in WorkerSpec.
	os.environ["ROLE_WORLD_SIZE"] = str(gpu_count)
	# # The number of worker group restarts so far.
	# os.environ["TORCHELASTIC_RESTART_COUNT"] = TODO
	# # The configured maximum number of restarts.
	# os.environ["TORCHELASTIC_MAX_RESTARTS"] = TODO
	# # Equal to the rendezvous run_id (e.g. unique job id).
	# os.environ["TORCHELASTIC_RUN_ID"] = TODO
	# # System executable override. If provided, the python user script will use the value of PYTHON_EXEC as executable. The sys.executable is used by default.
	# os.environ["PYTHON_EXEC"] = TODO

	# We're already parallelizing over the context, so we can't also parallelize inside the tokenizers (?)
	os.environ["TOKENIZERS_PARALLELISM"] = "false"

	device = f"cuda:{rank}"
	torch.cuda.set_device(rank)

	input_queue = input_queues[rank]
	del input_queues

	# Load model once
	log.debug(f'inference_worker for rank {rank} creating the model object now')
	local_model = InferenceAR(args)
	del args

	log.debug(f'inference_worker for rank {rank} ready, pushing a "ready" message to the queue')
	result_queue.put((rank, "ready"))

	# Install interrupt signal handler so that we can shut down gracefully.
	should_quit = False
	def signal_handler(signum, frame):
		nonlocal should_quit
		log.info(f"[RANK{rank}] Received signal {signum}, shutting down")
		should_quit = True
		try:
			input_queue.put(None)
		except ValueError:
			pass

	signal.signal(signal.SIGINT, signal_handler)

	while not should_quit:
		try:
			inputs_task = input_queue.get()
		except ValueError:
			# Queue was closed, we can exit.
			log.debug(f"[RANK{rank}] Input queue was closed, exiting.")
			break
		if inputs_task is None:
			# Special sentinel value to indicate that we are done and can exit.
			log.debug(f"[RANK{rank}] Got input {inputs_task}, exiting.")
			break

		# Note: we don't need to chunk the inputs for this rank / process, this is done
		#       automatically in the model.
		# Note: we don't need to move the inputs to a specific device either since the
		#       Gen3C API expects NumPy arrays.
		if False:
			log.debug(f"[RANK{rank}] Moving task to {device=}")
			inputs_task = move_to_device(inputs_task, device)

		# Run the requested task
		with torch.no_grad():
			task_type, args, kwargs = inputs_task
			log.debug(f"[RANK{rank}] Got task: {task_type=}")

			if task_type == 'inference':
				log.debug(f"[RANK{rank}] Running `inference_on_cameras()`...")
				output = local_model.inference_on_cameras(*args, **kwargs)
				log.debug(f"[RANK{rank}] Done `inference_on_cameras()`!")

				if is_tp_cp_pp_rank0():
					log.debug(f"[RANK{rank}] Moving outputs of `inference_on_cameras()` to the CPU")
					output = move_to_device(output, device='cpu')
					log.debug(f"[RANK{rank}] Pushing outputs of `inference_on_cameras()` to the results queue")
					result_queue.put(output)

			elif task_type == 'seeding':
				log.debug(f"[RANK{rank}] Calling `seed_model_from_values()...`")
				if cosmos_variant == 'predict1':
					output = local_model.seed_model_from_values(*args, **kwargs)
				else:
					raise NotImplementedError(f"Unsupported cosmos variant: {cosmos_variant}")
				output = move_to_device(output, device='cpu')
				result_queue.put((rank, "seed_model_from_values_done", output))
				log.debug(f"[RANK{rank}] Done with `seed_model_from_values()`")

			elif task_type == 'clear_cache':
				log.debug(f"[RANK{rank}] Calling `clear_cache()...`")
				local_model.clear_cache()
				result_queue.put((rank, "clear_cache_done"))
				log.debug(f"[RANK{rank}] Done with `clear_cache()`")

			elif task_type == 'get_cache_input_depths':
				log.debug(f"[RANK{rank}] Calling `get_cache_input_depths()...`")
				input_depths = local_model.get_cache_input_depths()
				attrs_queue.put(('cache_input_depths', input_depths.cpu(), True))
				log.debug(f"[RANK{rank}] Done with `get_cache_input_depths()`")

			elif task_type == 'getattr':
				assert kwargs is None
				assert len(args) == 1
				attr_name = args[0]
				assert isinstance(attr_name, str)
				has_attr = hasattr(local_model, attr_name)
				attr_value_or_none = getattr(local_model, attr_name)

				if has_attr and (attr_value_or_none is not None) and torch.is_tensor(attr_value_or_none):
					log.debug(f"[RANK{rank}] Attribute {attr_name=} is a torch tensor on "
						  f"device {attr_value_or_none.device}, cloning it before sending it through the queue")
					attr_value_or_none = attr_value_or_none.clone()

				log.debug(f"[RANK{rank}] Pushing attribute value for {attr_name=}")
				attrs_queue.put((attr_name, attr_value_or_none, has_attr))

			else:
				raise NotImplementedError(f"Unsupported task type for Cosmos inference worker: {task_type}")

	# Cleanup before exiting
	local_model.cleanup()
	del local_model


def inference_worker(*args, **kwargs):
	try:
		_inference_worker(*args, **kwargs)
	except Exception as e:
		import traceback
		rank = os.environ.get("LOCAL_RANK", "(unknown)")
		log.error(f"[RANK{rank}] encountered exception: {e}. Will re-raise after cleanup."
				  f" Stack trace:\n{traceback.format_exc()}")

		try:
			import torch.distributed as dist
			dist.destroy_process_group()
			log.info(f"[RANK{rank}] Destroyed model parallel group after catching exception."
					 " Will re-raise now.")
		except Exception as _:
			pass

		raise e


class MultiGPUInferenceAR():
	"""
	Adapter class to run multi-GPU Cosmos inference in the context of the FastAPI inference server.
	This class implements the same interface as `InferenceAR`, but spawns one process per GPU and
	forwards inference requests to the multiple processes via a work queue.

	The worker processes wait for work from the queue, perform inference, and gather all results
	on the rank 0 process. That process then pushes results to the result queue.
	"""
	def __init__(self, gpu_count: int, cosmos_variant: str, args: argparse.Namespace):
		import torch
		import torch.multiprocessing as mp

		self.gpu_count = gpu_count
		assert self.gpu_count <= torch.cuda.device_count(), \
			   f"Requested {self.gpu_count} GPUs, but only {torch.cuda.device_count()} are available."

		ctx = mp.get_context('spawn')
		manager = ctx.Manager()
		self.input_queues: list[mp.Queue] = [ctx.Queue() for _ in range(self.gpu_count)]
		self.result_queue = manager.Queue()
		self.attrs_queue = manager.Queue()

		log.info(f"Spawning {self.gpu_count} processes (one per GPU)")
		self.ctx = mp.spawn(
			inference_worker,
			args=(args, self.gpu_count, cosmos_variant,
				  self.input_queues, self.result_queue, self.attrs_queue),
			nprocs=self.gpu_count,
			join=False
		)

		log.info(f"Waiting for {self.gpu_count} processes to load the model...")
		for _ in range(self.gpu_count):
			v = self.result_queue.get()
			if not isinstance(v, tuple) or len(v) != 2 or v[1] != "ready":
				raise ValueError(f"Expected a 'ready' message from each process, but received: {v}")
			log.info(f"Process {v[0]} is ready.")


	def inference_on_cameras(self, *args, **kwargs):
		log.debug(f"inference_on_cameras(): submitting request to {len(self.input_queues)} inference processes.")
		for iq in self.input_queues:
			# Send the same input to each process
			task = ('inference', args, kwargs)
			iq.put(task)

		# Wait on the result queue to produce the result (this could take a while).
		log.debug(f"inference_on_cameras(): waiting for result...")
		outputs = self.result_queue.get()
		log.debug(f"inference_on_cameras(): got inference results! Cloning and returning.")
		return clone_tensors(outputs)


	def seed_model_from_values(self, *args, **kwargs):
		log.debug(f"seed_model_from_values(): submitting request to {len(self.input_queues)} inference processes.")
		for iq in self.input_queues:
			task = ('seeding', args, kwargs)
			iq.put(task)

		# TODO: refactor this, and maybe use some events or another primitive
		log.info(f"Waiting for {self.gpu_count} processes to be done with seeding...")
		for i in range(self.gpu_count):
			v = self.result_queue.get()
			if not isinstance(v, tuple) or len(v) != 3 or v[1] != "seed_model_from_values_done":
				raise ValueError(f"Expected a 'seed_model_from_values_done' message from each process, but received: {v}")
			log.info(f"Process {v[0]} is done with `seed_model_from_values()`.")

			# Arbitrarily pick the output from the first process
			if i == 0:
				outputs = v[2]

		return clone_tensors(outputs)


	def clear_cache(self):
		for iq in self.input_queues:
			task = ('clear_cache', None, None)
			iq.put(task)

		# TODO: refactor this, and maybe use some events or another primitive
		log.info(f"Waiting for {self.gpu_count} processes to be done with clear_cache...")
		for _ in range(self.gpu_count):
			v = self.result_queue.get()
			if not isinstance(v, tuple) or len(v) != 2 or v[1] != "clear_cache_done":
				raise ValueError(f"Expected a 'clear_cache_done' message from each process, but received: {v}")
			log.info(f"Process {v[0]} is done with `clear_cache()`.")


	def get_cache_input_depths(self):
		name = 'cache_input_depths'
		task = ('get_cache_input_depths', None, None)
		self.input_queues[0].put(task)

		# TODO: refactor this, and maybe use some events or another primitive
		looked_up_name, value, exists = self.attrs_queue.get()
		if looked_up_name != name:
			# TODO: this could be handled better (retry or enforce some ordering maybe).
			raise ValueError(f"Queried model for attribute '{name}' but got attribute '{looked_up_name}',"
							 " there was likely a race condition.")
		log.debug(f"Got a valid response, returning value for `get_cache_input_depths()`")
		return value


	def __getattr__(self, name: str):
		log.debug(f"__getattr__({name=}) called")
		# Note: this will not be called for methods we implement here, or attributes
		# that actually exist in this object.
		# Query the attribute from rank 0 (arbitrarily)
		task = ('getattr', (name,), None)
		self.input_queues[0].put(task)

		# Get result (blocking)
		log.debug(f"Waiting for response on `attrs_queue`...")
		looked_up_name, value, exists = self.attrs_queue.get()
		if looked_up_name != name:
			# TODO: this could be handled better (retry or enforce some ordering maybe).
			raise ValueError(f"Queried model for attribute '{name}' but got attribute '{looked_up_name}',"
							 " there was likely a race condition.")
		if not exists:
			raise AttributeError(f"Model has no attribute named '{name}'")
		log.debug(f"Got a valid response, returning {name} == {value}")
		return value


	def cleanup(self):
		"""
		Clean up resources before shutting down.
		"""
		log.info(f"MultiGPUInferenceAR winding down, asking {len(self.input_queues)} processes to clean up.")

		# "Close" all queues (there's no actual `close` method in PyTorch MP queues)
		for iq in self.input_queues:
			iq.put(None)

		# Wait for all processes to finish
		log.info(f"Waiting for {len(self.input_queues)} processes to finish (join).")
		self.ctx.join()
		log.info(f"{len(self.input_queues)} processes have finished.")
