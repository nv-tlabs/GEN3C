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

"""
Safe serialization for the inference API request wire format.

The API server reads untrusted HTTP request bodies, so it must never decode them
with ``pickle``: ``pickle.loads`` executes arbitrary code carried in the stream,
which on an unauthenticated endpoint is remote code execution.

This module serializes the request dataclasses with ``safetensors`` for the numpy
arrays and a small JSON header (carried in the safetensors metadata) for the
scalar / enum / compressed-buffer fields. A request body produced this way can
only ever decode to one of the allow-listed request types, and can never carry an
executable gadget.
"""

from __future__ import annotations

import base64
import dataclasses
import json
from enum import Enum

import numpy as np
from safetensors.numpy import load as _st_load
from safetensors.numpy import save as _st_save

import api_types

#: Only these classes can be reconstructed by :func:`loads`. Producing an
#: arbitrary class from the wire is exactly the capability being removed, so the
#: set is explicit rather than derived from the payload.
_CLASSES = {
	cls.__name__: cls
	for cls in (
		api_types.InferenceRequest,
		api_types.SeedingRequest,
		api_types.CompressedSeedingRequest,
	)
}

_ENUMS = {
	api_types.CompressionFormat.__name__: api_types.CompressionFormat,
}

#: safetensors stores a ``str -> str`` metadata map in its header; the whole field
#: descriptor lives under this single key.
_HEADER_KEY = "gen3c_api"


def dumps(obj) -> bytes:
	"""Serialize a request dataclass to a safetensors blob."""
	cls = type(obj)
	if cls.__name__ not in _CLASSES:
		raise TypeError(f"safe_serialization: cannot serialize {cls.__name__!r}")

	tensors: dict[str, np.ndarray] = {}
	fields: dict[str, dict] = {}
	for field in dataclasses.fields(cls):
		fields[field.name] = _encode(field.name, getattr(obj, field.name), tensors)

	header = json.dumps({"class": cls.__name__, "fields": fields})
	return _st_save(tensors, metadata={_HEADER_KEY: header})


def loads(data: bytes):
	"""Deserialize a safetensors blob produced by :func:`dumps`."""
	header = _read_header(data)
	cls = _CLASSES.get(header.get("class"))
	if cls is None:
		raise ValueError(f"safe_serialization: unknown request type {header.get('class')!r}")

	tensors = _st_load(bytes(data))
	kwargs = {name: _decode(spec, tensors) for name, spec in header["fields"].items()}
	return cls(**kwargs)


# -- internals ----------------------------------------------------------------

def _encode(name: str, value, tensors: dict) -> dict:
	if isinstance(value, np.generic):
		value = value.item()

	if value is None:
		return {"k": "none"}
	if isinstance(value, np.ndarray):
		if value.size == 0:
			return {"k": "empty", "shape": list(value.shape), "dtype": value.dtype.str}
		key = f"arr::{name}"
		tensors[key] = np.ascontiguousarray(value)
		return {"k": "arr", "key": key}
	if isinstance(value, bool):
		return {"k": "bool", "v": value}
	if isinstance(value, int):
		return {"k": "int", "v": value}
	if isinstance(value, float):
		return {"k": "float", "v": value}
	if isinstance(value, str):
		return {"k": "str", "v": value}
	if isinstance(value, Enum):
		return {"k": "enum", "enum": type(value).__name__, "v": value.value}
	if isinstance(value, (list, tuple)):
		if value and all(isinstance(x, (bytes, bytearray)) for x in value):
			return {"k": "bytes", "v": [base64.b64encode(bytes(x)).decode("ascii") for x in value]}
		return {"k": "json", "v": list(value)}

	raise TypeError(f"safe_serialization: field {name!r} has unsupported type {type(value).__name__}")


def _decode(spec: dict, tensors: dict):
	kind = spec["k"]
	if kind == "none":
		return None
	if kind == "arr":
		return tensors[spec["key"]]
	if kind == "empty":
		return np.empty(tuple(spec["shape"]), dtype=np.dtype(spec["dtype"]))
	if kind in ("bool", "int", "float", "str", "json"):
		return spec["v"]
	if kind == "enum":
		return _ENUMS[spec["enum"]](spec["v"])
	if kind == "bytes":
		return [base64.b64decode(s) for s in spec["v"]]

	raise ValueError(f"safe_serialization: unknown field kind {kind!r}")


def _read_header(data: bytes) -> dict:
	# safetensors layout: a little-endian u64 header length, then the JSON header,
	# whose "__metadata__" map carries our descriptor.
	if len(data) < 8:
		raise ValueError("safe_serialization: truncated blob")
	n = int.from_bytes(bytes(data[:8]), "little")
	meta = json.loads(bytes(data[8 : 8 + n])).get("__metadata__", {})
	raw = meta.get(_HEADER_KEY)
	if raw is None:
		raise ValueError("safe_serialization: missing header (not a gen3c request blob)")
	return json.loads(raw)
