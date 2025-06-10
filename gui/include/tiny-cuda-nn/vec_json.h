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

/** @file   vec_json.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Conversion between tcnn's vector / matrix / quaternion types
 *          and nlohmann::json.
 */

#pragma once

#include <tiny-cuda-nn/common.h>

#include <json/json.hpp>

namespace tcnn {

template <typename T, uint32_t N, uint32_t M>
void to_json(nlohmann::json& j, const tmat<T, N, M>& mat) {
	for (int row = 0; row < M; ++row) {
		nlohmann::json column = nlohmann::json::array();
		for (int col = 0; col < N; ++col) {
			column.push_back(mat[col][row]);
		}
		j.push_back(column);
	}
}

template <typename T, uint32_t N, uint32_t M>
void from_json(const nlohmann::json& j, tmat<T, N, M>& mat) {
	for (std::size_t row = 0; row < M; ++row) {
		const auto& jrow = j.at(row);
		for (std::size_t col = 0; col < N; ++col) {
			const auto& value = jrow.at(col);
			mat[col][row] = value.get<T>();
		}
	}
}

template <typename T, uint32_t N>
void to_json(nlohmann::json& j, const tvec<T, N>& v) {
	for (uint32_t i = 0; i < N; ++i) {
		j.push_back(v[i]);
	}
}

template <typename T, uint32_t N>
void from_json(const nlohmann::json& j, tvec<T, N>& v) {
	for (uint32_t i = 0; i < N; ++i) {
		v[i] = j.at(i).get<T>();
	}
}

template <typename T>
void to_json(nlohmann::json& j, const tquat<T>& q) {
	j.push_back(q.x);
	j.push_back(q.y);
	j.push_back(q.z);
	j.push_back(q.w);
}

template <typename T>
void from_json(const nlohmann::json& j, tquat<T>& q) {
	q.x = j.at(0).get<T>();
	q.y = j.at(1).get<T>();
	q.z = j.at(2).get<T>();
	q.w = j.at(3).get<T>();
}

}
