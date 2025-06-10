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

/** @file   vec_pybind11.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  pybind11 bindings for NGP's vector and matrix types. Adapted from
 *          Patrik Huber's glm binding code per the BSD license of pybind11.
 */

#pragma once

#include <tiny-cuda-nn/vec.h>

#include <cstddef>

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#endif

namespace pybind11 {
namespace detail {

template <typename T, uint32_t N>
struct type_caster<tcnn::tvec<T, N>> {
	using vector_type = tcnn::tvec<T, N>;
	using Scalar = T;
	static constexpr std::size_t num_elements = N;

	bool load(handle src, bool) {
		auto buf = array_t<Scalar>::ensure(src);
		if (!buf) {
			return false;
		}

		if (buf.ndim() != 1) {
			return false; // not a rank-1 tensor (i.e. vector)
		}

		if (buf.shape(0) != num_elements) {
			return false; // not a 2-elements vector
		}

		for (size_t i = 0; i < num_elements; ++i) {
			value[i] = *buf.data(i);
		}

		return true;
	}

	static handle cast(const vector_type& src, return_value_policy, handle) {
		return array(
			num_elements,
			src.data()
		).release();
	}

	// Specifies the doc-string for the type in Python:
	PYBIND11_TYPE_CASTER(vector_type, _("vec"));
};

template <typename T, uint32_t N, uint32_t M>
struct type_caster<tcnn::tmat<T, N, M>> {
	using matrix_type = tcnn::tmat<T, N, M>;
	using Scalar = T;
	static constexpr std::size_t num_rows = M;
	static constexpr std::size_t num_cols = N;

	bool load(handle src, bool) {
		auto buf = array_t<Scalar>::ensure(src);
		if (!buf) {
			return false;
		}

		if (buf.ndim() != 2) {
			return false; // not a rank-2 tensor (i.e. matrix)
		}

		if (buf.shape(0) != num_rows || buf.shape(1) != num_cols) {
			return false; // not a 4x4 matrix
		}

		for (size_t i = 0; i < num_cols; ++i) {
			for (size_t j = 0; j < num_rows; ++j) {
				value[i][j] = *buf.data(j, i);
			}
		}

		return true;
	}

	static handle cast(const matrix_type& src, return_value_policy, handle) {
		return array(
			{ num_rows, num_cols },
			{ sizeof(Scalar), sizeof(Scalar) * num_rows }, // strides - flip the row/col layout!
			src.data()
		).release();
	}

	// Specifies the doc-string for the type in Python:
	PYBIND11_TYPE_CASTER(matrix_type, _("mat"));
};

}
}

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
