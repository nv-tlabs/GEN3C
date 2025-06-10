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

/** @file   gpu_memory_json.h
 *  @author Nikolaus Binder and Thomas Müller, NVIDIA
 *  @brief  binding between GPUMemory and JSON librariy
 */

#pragma once

#include <json/json.hpp>

namespace tcnn {

inline nlohmann::json::binary_t gpu_memory_to_json_binary(const void* gpu_data, size_t n_bytes) {
	nlohmann::json::binary_t data_cpu;
	data_cpu.resize(n_bytes);
	CUDA_CHECK_THROW(cudaMemcpy(data_cpu.data(), gpu_data, n_bytes, cudaMemcpyDeviceToHost));
	return data_cpu;
}

inline void json_binary_to_gpu_memory(const nlohmann::json::binary_t& cpu_data, void* gpu_data, size_t n_bytes) {
	CUDA_CHECK_THROW(cudaMemcpy(gpu_data, cpu_data.data(), n_bytes, cudaMemcpyHostToDevice));
}

template <typename T>
inline void to_json(nlohmann::json& j, const GPUMemory<T>& gpu_data) {
	j = gpu_memory_to_json_binary(gpu_data.data(), gpu_data.get_bytes());
}

template <typename T>
inline void from_json(const nlohmann::json& j, GPUMemory<T>& gpu_data) {
	if (j.is_binary()) {
		const nlohmann::json::binary_t& cpu_data = j.get_binary();
		gpu_data.resize(cpu_data.size()/sizeof(T));
		json_binary_to_gpu_memory(cpu_data, gpu_data.data(), gpu_data.get_bytes());
	} else if (j.is_object()) {
		// https://json.nlohmann.me/features/binary_values/#json
		json::array_t arr = j["bytes"];
		nlohmann::json::binary_t cpu_data;
		cpu_data.resize(arr.size());
		for(size_t i = 0; i < arr.size(); ++i) {
			cpu_data[i] = (uint8_t)arr[i];
		}
		gpu_data.resize(cpu_data.size()/sizeof(T));
		json_binary_to_gpu_memory(cpu_data, gpu_data.data(), gpu_data.get_bytes());
	} else {
		throw std::runtime_error("Invalid json type: must be either binary or object");
	}
}

}
