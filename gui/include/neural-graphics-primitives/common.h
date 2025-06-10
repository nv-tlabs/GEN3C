/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/** @file   common.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Shared functionality among multiple neural-graphics-primitives components.
 */

#pragma once

#ifdef _WIN32
#	define NOMINMAX
#endif

#include <tiny-cuda-nn/common.h>
using namespace tcnn;

#if defined(__CUDA_ARCH__)
#	define NGP_PRAGMA_UNROLL _Pragma("unroll")
#	define NGP_PRAGMA_NO_UNROLL _Pragma("unroll 1")
#else
#	define NGP_PRAGMA_UNROLL
#	define NGP_PRAGMA_NO_UNROLL
#endif

#if defined(__CUDACC__) || (defined(__clang__) && defined(__CUDA__))
#	define NGP_HOST_DEVICE __host__ __device__
#else
#	define NGP_HOST_DEVICE
#endif

namespace ngp {

enum class EMeshRenderMode : int {
	Off,
	VertexColors,
	VertexNormals,
	FaceIDs,
};

enum class EGroundTruthRenderMode : int {
	Shade,
	Depth,
	NumRenderModes,
};
static constexpr const char* GroundTruthRenderModeStr = "Shade\0Depth\0\0";

enum class ERenderMode : int {
	AO,
	Shade,
	Normals,
	Positions,
	Depth,
	Distortion,
	Cost,
	Slice,
	NumRenderModes,
	EncodingVis, // EncodingVis exists outside of the standard render modes
};
static constexpr const char* RenderModeStr = "AO\0Shade\0Normals\0Positions\0Depth\0Distortion\0Cost\0Slice\0\0";

enum class EPmVizMode : int {
	Shade,
	Depth,
	Offset,
	Holes,
};
static constexpr const char* PmVizModeStr = "Shade\0Depth\0Offset\0Holes\0\0";

enum class ERandomMode : int {
	Random,
	Halton,
	Sobol,
	Stratified,
	NumImageRandomModes,
};
static constexpr const char* RandomModeStr = "Random\0Halton\0Sobol\0Stratified\0\0";

enum class ELossType : int {
	L2,
	L1,
	Mape,
	Smape,
	Huber,
	LogL1,
	RelativeL2,
};
static constexpr const char* LossTypeStr = "L2\0L1\0MAPE\0SMAPE\0Huber\0LogL1\0RelativeL2\0\0";

enum class EMeshSdfMode : int {
	Watertight,
	Raystab,
	PathEscape,
};
static constexpr const char* MeshSdfModeStr = "Watertight\0Raystab\0PathEscape\0\0";

enum class EColorSpace : int {
	Linear,
	SRGB,
	VisPosNeg,
};
static constexpr const char* ColorSpaceStr = "Linear\0SRGB\0\0";

enum class ETonemapCurve : int { Identity, ACES, Hable, Reinhard };
static constexpr const char* TonemapCurveStr = "Identity\0ACES\0Hable\0Reinhard\0\0";

enum class EDlssQuality : int {
	UltraPerformance,
	MaxPerformance,
	Balanced,
	MaxQuality,
	UltraQuality,
	NumDlssQualitySettings,
	None,
};
static constexpr const char* DlssQualityStr = "UltraPerformance\0MaxPerformance\0Balanced\0MaxQuality\0UltraQuality\0Invalid\0None\0\0";
static constexpr const char* DlssQualityStrArray[] = {
	"UltraPerformance", "MaxPerformance", "Balanced", "MaxQuality", "UltraQuality", "Invalid", "None"
};

enum class ETestbedMode : int {
	Gen3c,
	None,
};

enum class ESDFGroundTruthMode : int {
	RaytracedMesh,
	SpheretracedMesh,
	SDFBricks,
};

enum EPmPixelState : uint8_t {
	Hole = 0,
	FillableHole,
	FilledHole,
	Reprojected,
};


struct TrainingXForm {
	NGP_HOST_DEVICE bool operator==(const TrainingXForm& other) const { return start == other.start && end == other.end; }

	mat4x3 start;
	mat4x3 end;
};

enum class ELensMode : int {
	Perspective,
	OpenCV,
	FTheta,
	LatLong,
	OpenCVFisheye,
	Equirectangular,
	Orthographic,
};
static constexpr const char* LensModeStr = "Perspective\0OpenCV\0F-Theta\0LatLong\0OpenCV Fisheye\0Equirectangular\0Orthographic\0\0";

struct Lens {
	ELensMode mode = ELensMode::Perspective;
	float params[7] = {};

	NGP_HOST_DEVICE bool is_360() const { return mode == ELensMode::Equirectangular || mode == ELensMode::LatLong; }

	NGP_HOST_DEVICE bool supports_dlss() {
		return mode == ELensMode::LatLong || mode == ELensMode::Equirectangular || mode == ELensMode::Perspective ||
			mode == ELensMode::Orthographic || mode == ELensMode::OpenCV || mode == ELensMode::OpenCVFisheye;
	}
};

enum class EGen3cCameraSource : int {
	// Fake camera trajectory based on fixed translation and rotation speeds.
	Fake = 0,
	// Camera trajectory from the current viewpoint + predicted movement,
	// including when using a VR headset.
	Viewpoint,
	// Camera trajectory from a path authored with the camera tools.
	Authored
};
static constexpr const char* Gen3cCameraSourceStr = "Fake\0Viewpoint\0Authored\0\0";

inline NGP_HOST_DEVICE uint32_t binary_search(float val, const float* data, uint32_t length) {
	if (length == 0) {
		return 0;
	}

	uint32_t it;
	uint32_t count, step;
	count = length;

	uint32_t first = 0;
	while (count > 0) {
		it = first;
		step = count / 2;
		it += step;
		if (data[it] < val) {
			first = ++it;
			count -= step + 1;
		} else {
			count = step;
		}
	}

	return min(first, length - 1);
}

template <typename T> struct Buffer2DView {
	T* data = nullptr;
	ivec2 resolution = 0;

	// Lookup via integer pixel position (no bounds checking)
	NGP_HOST_DEVICE T at(const ivec2& px) const { return data[px.x + px.y * resolution.x]; }

	// Lookup via UV coordinates in [0,1]^2
	NGP_HOST_DEVICE T at(const vec2& uv) const {
		ivec2 px = clamp(ivec2(vec2(resolution) * uv), 0, resolution - 1);
		return at(px);
	}

	// Lookup via UV coordinates in [0,1]^2 and LERP the nearest texels
	NGP_HOST_DEVICE T at_lerp(const vec2& uv) const {
		const vec2 px_float = vec2(resolution) * uv;
		const ivec2 px = ivec2(px_float);

		const vec2 weight = px_float - vec2(px);

		auto read_val = [&](ivec2 pos) { return at(clamp(pos, 0, resolution - 1)); };

		return (
			(1 - weight.x) * (1 - weight.y) * read_val({px.x, px.y}) + (weight.x) * (1 - weight.y) * read_val({px.x + 1, px.y}) +
			(1 - weight.x) * (weight.y) * read_val({px.x, px.y + 1}) + (weight.x) * (weight.y) * read_val({px.x + 1, px.y + 1})
		);
	}

	NGP_HOST_DEVICE operator bool() const { return data; }
};

} // namespace ngp
