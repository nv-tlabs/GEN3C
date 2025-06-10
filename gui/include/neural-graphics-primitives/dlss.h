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

/** @file   dlss.h
 *  @author Thomas Müller, NVIDIA
 */

#pragma once

#include <neural-graphics-primitives/common.h>

#include <memory>

namespace ngp {

class IDlss {
public:
	virtual ~IDlss() {}

	virtual void update_feature(
		const ivec2& in_resolution,
		bool is_hdr,
		bool sharpen
	) = 0;
	virtual void run(
		const ivec2& in_resolution,
		bool is_hdr,
		float sharpening,
		const vec2& jitter_offset,
		bool shall_reset
	) = 0;

	virtual cudaSurfaceObject_t frame() = 0;
	virtual cudaSurfaceObject_t depth() = 0;
	virtual cudaSurfaceObject_t mvec() = 0;
	virtual cudaSurfaceObject_t exposure() = 0;
	virtual cudaSurfaceObject_t output() = 0;

	virtual ivec2 clamp_resolution(const ivec2& resolution) const = 0;
	virtual ivec2 out_resolution() const = 0;
	virtual ivec2 max_out_resolution() const = 0;

	virtual bool is_hdr() const = 0;
	virtual bool sharpen() const = 0;
	virtual EDlssQuality quality() const = 0;
};

class IDlssProvider {
public:
	virtual ~IDlssProvider() {}

	virtual size_t allocated_bytes() const = 0;
	virtual std::unique_ptr<IDlss> init_dlss(const ivec2& out_resolution) = 0;
};

#ifdef NGP_VULKAN
std::shared_ptr<IDlssProvider> init_vulkan_and_ngx();
#endif

}
