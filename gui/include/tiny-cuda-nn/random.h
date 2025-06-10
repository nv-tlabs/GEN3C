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

/** @file   random.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Collection of CUDA kernels related to random numbers
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>

#include <pcg32/pcg32.h>

namespace tcnn {

template <typename T, typename RNG, size_t N_TO_GENERATE, typename F>
__global__ void generate_random_kernel(const size_t n_elements, RNG rng, T* __restrict__ out, const F transform) {
	const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	const size_t n_threads = blockDim.x * gridDim.x;

	rng.advance(i*N_TO_GENERATE);

	TCNN_PRAGMA_UNROLL
	for (size_t j = 0; j < N_TO_GENERATE; ++j) {
		const size_t idx = i + n_threads * j;
		if (idx >= n_elements) {
			return;
		}

		out[idx] = transform((T)rng.next_float());
	}
}

template <typename T, typename RNG, typename F>
void generate_random(cudaStream_t stream, RNG& rng, size_t n_elements, T* out, F&& transform) {
	static constexpr size_t N_TO_GENERATE = 4;

	size_t n_threads = div_round_up(n_elements, N_TO_GENERATE);
	generate_random_kernel<T, RNG, N_TO_GENERATE><<<n_blocks_linear(n_threads), N_THREADS_LINEAR, 0, stream>>>(n_elements, rng, out, transform);

	rng.advance(n_elements);
}

template <typename T, typename RNG>
void generate_random_uniform(cudaStream_t stream, RNG& rng, size_t n_elements, T* out, const T lower = (T)0.0, const T upper = (T)1.0) {
	generate_random(stream, rng, n_elements, out, [upper, lower] __device__ (T val) { return val * (upper - lower) + lower; });
}

template <typename T, typename RNG>
void generate_random_uniform(RNG& rng, size_t n_elements, T* out, const T lower = (T)0.0, const T upper = (T)1.0) {
	generate_random_uniform(nullptr, rng, n_elements, out, lower, upper);
}

template <typename T, typename RNG>
void generate_random_logistic(cudaStream_t stream, RNG& rng, size_t n_elements, T* out, const T mean = (T)0.0, const T stddev = (T)1.0) {
	generate_random(stream, rng, n_elements, out, [mean, stddev] __device__ (T val) { return (T)logit(val) * stddev * 0.551328895f + mean; });
}

template <typename T, typename RNG>
void generate_random_logistic(RNG& rng, size_t n_elements, T* out, const T mean = (T)0.0, const T stddev = (T)1.0) {
	generate_random_logistic(nullptr, rng, n_elements, out, mean, stddev);
}

}
