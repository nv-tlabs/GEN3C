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

/** @file   discrete_distribution.h
 *  @author Thomas Müller, NVIDIA
 */

#pragma once

#include <vector>

namespace ngp {

struct DiscreteDistribution {
	void build(std::vector<float> weights) {
		float total_weight = 0;
		for (float w : weights) {
			total_weight += w;
		}
		float inv_total_weight = 1 / total_weight;

		float cdf_accum = 0;
		cdf.clear();
		for (float w : weights) {
			float norm = w * inv_total_weight;
			cdf_accum += norm;
			pmf.emplace_back(norm);
			cdf.emplace_back(cdf_accum);
		}
		cdf.back() = 1.0f; // Prevent precision problems from causing overruns in the end
	}

	uint32_t sample(float val) {
		return std::min(binary_search(val, cdf.data(), (uint32_t)cdf.size()), (uint32_t)cdf.size()-1);
	}

	std::vector<float> pmf;
	std::vector<float> cdf;
};

}
