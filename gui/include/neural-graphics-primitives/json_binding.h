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

/** @file   json_binding.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Conversion between some ngp types and nlohmann::json.
 */

#pragma once

#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/common.h>

#include <tiny-cuda-nn/vec_json.h>

#include <json/json.hpp>

namespace ngp {

inline void to_json(nlohmann::json& j, const BoundingBox& box) {
	j["min"] = box.min;
	j["max"] = box.max;
}

inline void from_json(const nlohmann::json& j, BoundingBox& box) {
	box.min = j.at("min");
	box.max = j.at("max");
}

inline void to_json(nlohmann::json& j, const Lens& lens) {
	if (lens.mode == ELensMode::OpenCV) {
		j["is_fisheye"] = false;
		j["k1"] = lens.params[0];
		j["k2"] = lens.params[1];
		j["p1"] = lens.params[2];
		j["p2"] = lens.params[3];
	} else if (lens.mode == ELensMode::OpenCVFisheye) {
		j["is_fisheye"] = true;
		j["k1"] = lens.params[0];
		j["k2"] = lens.params[1];
		j["k3"] = lens.params[2];
		j["k4"] = lens.params[3];
	} else if (lens.mode == ELensMode::FTheta) {
		j["ftheta_p0"] = lens.params[0];
		j["ftheta_p1"] = lens.params[1];
		j["ftheta_p2"] = lens.params[2];
		j["ftheta_p3"] = lens.params[3];
		j["ftheta_p4"] = lens.params[4];
		j["w"] = lens.params[5];
		j["h"] = lens.params[6];
	} else if (lens.mode == ELensMode::LatLong) {
		j["latlong"] = true;
	} else if (lens.mode == ELensMode::Equirectangular) {
		j["equirectangular"] = true;
	} else if (lens.mode == ELensMode::Orthographic) {
		j["orthographic"] = true;
	}
}

inline void from_json(const nlohmann::json& j, Lens& lens) {
	if (j.contains("k1")) {
		if (j.value("is_fisheye", false)) {
			lens.mode = ELensMode::OpenCVFisheye;
			lens.params[0] = j.at("k1");
			lens.params[1] = j.at("k2");
			lens.params[2] = j.at("k3");
			lens.params[3] = j.at("k4");
		} else {
			lens.mode = ELensMode::OpenCV;
			lens.params[0] = j.at("k1");
			lens.params[1] = j.at("k2");
			lens.params[2] = j.at("p1");
			lens.params[3] = j.at("p2");
		}
	} else if (j.contains("ftheta_p0")) {
		lens.mode = ELensMode::FTheta;
		lens.params[0] = j.at("ftheta_p0");
		lens.params[1] = j.at("ftheta_p1");
		lens.params[2] = j.at("ftheta_p2");
		lens.params[3] = j.at("ftheta_p3");
		lens.params[4] = j.at("ftheta_p4");
		lens.params[5] = j.at("w");
		lens.params[6] = j.at("h");
	} else if (j.contains("latlong")) {
		lens.mode = ELensMode::LatLong;
	} else if (j.contains("equirectangular")) {
		lens.mode = ELensMode::Equirectangular;
	} else if (j.contains("orthographic")) {
		lens.mode = ELensMode::Orthographic;
	} else {
		lens.mode = ELensMode::Perspective;
	}
}

inline void from_json(const nlohmann::json& j, TrainingXForm& x) {
	x.start = j.at("start");
	x.end = j.at("end");
}

inline void to_json(nlohmann::json& j, const TrainingXForm& x) {
	j["start"] = x.start;
	j["end"] = x.end;
}

}
