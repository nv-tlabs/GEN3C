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

/** @file   python_api.cpp
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/testbed.h>
#include <neural-graphics-primitives/thread_pool.h>

#include <json/json.hpp>

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11_json/pybind11_json.hpp>
#include <tiny-cuda-nn/vec_pybind11.h>
#include <tinylogger/tinylogger.h>

#include <filesystem/path.h>

#ifdef NGP_GUI
#	include <imgui/imgui.h>
#	ifdef _WIN32
#		include <GL/gl3w.h>
#	else
#		include <GL/glew.h>
#	endif
#	include <GLFW/glfw3.h>
#endif

using namespace nlohmann;
namespace py = pybind11;

namespace ngp {

// Returns RGBA and depth buffers
std::pair<py::array_t<float>, py::array_t<float>>
	Testbed::render_to_cpu(int width, int height, int spp, bool linear, float start_time, float end_time, float fps, float shutter_fraction) {
	m_windowless_render_surface.resize({width, height});
	m_windowless_render_surface.reset_accumulation();

	if (end_time < 0.f) {
		end_time = start_time;
	}

	bool path_animation_enabled = start_time >= 0.f;
	if (!path_animation_enabled) { // the old code disabled camera smoothing for non-path renders; so we preserve that behaviour
		m_smoothed_camera = m_camera;
	}

	// this rendering code assumes that the intra-frame camera motion starts from m_smoothed_camera (ie where we left off) to allow for EMA
	// camera smoothing. in the case of a camera path animation, at the very start of the animation, we have yet to initialize
	// smoothed_camera to something sensible
	// - it will just be the default boot position. oops!
	// that led to the first frame having a crazy streak from the default camera position to the start of the path.
	// so we detect that case and explicitly force the current matrix to the start of the path
	if (start_time == 0.f) {
		set_camera_from_time(start_time);
		m_smoothed_camera = m_camera;
	}

	auto start_cam_matrix = m_smoothed_camera;

	// now set up the end-of-frame camera matrix if we are moving along a path
	if (path_animation_enabled) {
		set_camera_from_time(end_time);
		apply_camera_smoothing(1000.f / fps);
	}

	auto end_cam_matrix = m_smoothed_camera;
	auto prev_camera_matrix = m_smoothed_camera;

	for (int i = 0; i < spp; ++i) {
		float start_alpha = ((float)i) / (float)spp * shutter_fraction;
		float end_alpha = ((float)i + 1.0f) / (float)spp * shutter_fraction;

		auto sample_start_cam_matrix = start_cam_matrix;
		auto sample_end_cam_matrix = camera_log_lerp(start_cam_matrix, end_cam_matrix, shutter_fraction);
		if (i == 0) {
			prev_camera_matrix = sample_start_cam_matrix;
		}

		if (path_animation_enabled) {
			set_camera_from_time(start_time + (end_time - start_time) * (start_alpha + end_alpha) / 2.0f);
			m_smoothed_camera = m_camera;
		}

		if (m_autofocus) {
			autofocus();
		}

		render_frame(
			m_stream.get(),
			sample_start_cam_matrix,
			sample_end_cam_matrix,
			prev_camera_matrix,
			m_screen_center,
			m_relative_focal_length,
			{}, // foveation
			{}, // prev foveation
			{}, // lens
			m_visualized_dimension,
			m_windowless_render_surface,
			!linear
		);
		prev_camera_matrix = sample_start_cam_matrix;
	}

	// For cam smoothing when rendering the next frame.
	m_smoothed_camera = end_cam_matrix;

	py::array_t<float> result_rgba({height, width, 4});
	py::buffer_info buf_rgba = result_rgba.request();

	py::array_t<float> result_depth({height, width});
	py::buffer_info buf_depth = result_depth.request();

	CUDA_CHECK_THROW(cudaMemcpy2DFromArray(
		buf_rgba.ptr, width * sizeof(float) * 4, m_windowless_render_surface.surface_provider().array(), 0, 0, width * sizeof(float) * 4, height, cudaMemcpyDeviceToHost
	));

	CUDA_CHECK_THROW(
		cudaMemcpy(buf_depth.ptr, m_windowless_render_surface.depth_buffer(), height * width * sizeof(float), cudaMemcpyDeviceToHost)
	);

	return {result_rgba, result_depth};
}

py::array_t<float> Testbed::render_to_cpu_rgba(
	int width, int height, int spp, bool linear, float start_time, float end_time, float fps, float shutter_fraction
) {
	return render_to_cpu(width, height, spp, linear, start_time, end_time, fps, shutter_fraction).first;
}

py::array_t<float> Testbed::view(bool linear, size_t view_idx) const {
	if (m_views.size() <= view_idx) {
		throw std::runtime_error{fmt::format("View #{} does not exist.", view_idx)};
	}

	auto& view = m_views.at(view_idx);
	auto& render_buffer = *view.render_buffer;

	auto res = render_buffer.out_resolution();

	py::array_t<float> result({res.y, res.x, 4});
	py::buffer_info buf = result.request();
	float* data = (float*)buf.ptr;

	CUDA_CHECK_THROW(cudaMemcpy2DFromArray(
		data, res.x * sizeof(float) * 4, render_buffer.surface_provider().array(), 0, 0, res.x * sizeof(float) * 4, res.y, cudaMemcpyDeviceToHost
	));

	if (linear) {
		ThreadPool{}.parallel_for<size_t>(0, res.y, [&](size_t y) {
			size_t base = y * res.x;
			for (uint32_t x = 0; x < res.x; ++x) {
				size_t px = base + x;
				data[px * 4 + 0] = srgb_to_linear(data[px * 4 + 0]);
				data[px * 4 + 1] = srgb_to_linear(data[px * 4 + 1]);
				data[px * 4 + 2] = srgb_to_linear(data[px * 4 + 2]);
			}
		});
	}

	return result;
}

std::pair<py::array_t<float>, py::array_t<uint32_t>>
	Testbed::reproject(const mat4x3& src, const py::array_t<float>& src_img, const py::array_t<float>& src_depth, const mat4x3& dst) {

	py::buffer_info src_img_buf = src_img.request();
	py::buffer_info src_depth_buf = src_depth.request();

	if (src_img_buf.ndim != 3) {
		throw std::runtime_error{"src image should be (H,W,C) where C=4"};
	}

	if (src_img_buf.shape[2] != 4) {
		throw std::runtime_error{"src image should be (H,W,C) where C=4"};
	}

	if (src_depth_buf.ndim != 2) {
		throw std::runtime_error{"src depth should be (H,W)"};
	}

	if (src_img_buf.shape[0] != src_depth_buf.shape[0] || src_img_buf.shape[1] != src_depth_buf.shape[1]) {
		throw std::runtime_error{"image and depth dimensions don't match"};
	}

	const ivec2 src_res = {(int)src_img_buf.shape[1], (int)src_img_buf.shape[0]};
	const ivec2 dst_res = src_res; // For now

	auto src_render_buffer = std::make_shared<CudaRenderBuffer>(std::make_shared<CudaSurface2D>());
	src_render_buffer->resize(src_res);

	auto dst_render_buffer = std::make_shared<CudaRenderBuffer>(std::make_shared<CudaSurface2D>());
	dst_render_buffer->resize(dst_res);

	View src_view, dst_view;

	src_view.camera0 = src_view.camera1 = src_view.prev_camera = src;
	src_view.device = &primary_device();
	src_view.foveation = src_view.prev_foveation = {};
	src_view.screen_center = vec2(0.5f);
	src_view.full_resolution = src_res;
	src_view.visualized_dimension = -1;
	src_view.relative_focal_length = m_relative_focal_length;
	src_view.render_buffer = src_render_buffer;

	dst_view.camera0 = dst_view.camera1 = dst_view.prev_camera = dst;
	dst_view.device = &primary_device();
	dst_view.foveation = dst_view.prev_foveation = {};
	dst_view.screen_center = vec2(0.5f);
	dst_view.full_resolution = dst_res;
	dst_view.visualized_dimension = -1;
	dst_view.relative_focal_length = m_relative_focal_length;
	dst_view.render_buffer = dst_render_buffer;

	CUDA_CHECK_THROW(cudaMemcpyAsync(
		src_render_buffer->frame_buffer(), src_img_buf.ptr, product(src_res) * sizeof(float) * 4, cudaMemcpyHostToDevice, m_stream.get()
	));
	CUDA_CHECK_THROW(cudaMemcpyAsync(
		src_render_buffer->depth_buffer(), src_depth_buf.ptr, product(src_res) * sizeof(float), cudaMemcpyHostToDevice, m_stream.get()
	));

	std::vector<const View*> src_views = {&src_view};
	reproject_views(src_views, dst_view);

	py::array_t<float> result_rgba({dst_res.y, dst_res.x, 4});
	py::buffer_info buf_rgba = result_rgba.request();

	py::array_t<uint32_t> result_idx({dst_res.y, dst_res.x});
	py::buffer_info buf_idx = result_idx.request();

	CUDA_CHECK_THROW(cudaMemcpyAsync(
		buf_rgba.ptr, dst_render_buffer->frame_buffer(), product(dst_res) * sizeof(float) * 4, cudaMemcpyDeviceToHost, m_stream.get()
	));

	auto idx_buffer = GPUImage<uint32_t>(dst_res, m_stream.get());

	parallel_for_gpu(
		m_stream.get(),
		idx_buffer.n_elements(),
		[out = idx_buffer.view(), in = dst_view.index_field.view(), src_width = src_res.x, dst_width = dst_res.x] __device__(size_t i) {
			ivec2 idx = ivec2(i % dst_width, i / dst_width);
			ivec2 src_idx = in(idx.y, idx.x).px;
			out(idx.y, idx.x) = src_idx.x + src_idx.y * src_width;
		}
	);

	CUDA_CHECK_THROW(
		cudaMemcpyAsync(buf_idx.ptr, idx_buffer.data(), product(dst_res) * sizeof(uint32_t), cudaMemcpyDeviceToHost, m_stream.get())
	);

	return {result_rgba, result_idx};
}

uint32_t Testbed::add_src_view(
	mat4x3 camera_to_world, float fx, float fy, float cx, float cy, Lens lens, pybind11::array_t<float> img, pybind11::array_t<float> depth, float timestamp, bool is_srgb
) {
	py::buffer_info src_img_buf = img.request();
	py::buffer_info src_depth_buf = depth.request();

	if (src_img_buf.ndim != 3) {
		throw std::runtime_error{"src image should be (H,W,C) where C=4"};
	}

	if (src_img_buf.shape[2] != 4) {
		throw std::runtime_error{"src image should be (H,W,C) where C=4"};
	}

	if (src_depth_buf.ndim != 2) {
		throw std::runtime_error{"src depth should be (H,W)"};
	}

	if (src_img_buf.shape[0] != src_depth_buf.shape[0] || src_img_buf.shape[1] != src_depth_buf.shape[1]) {
		throw std::runtime_error{"image and depth dimensions don't match"};
	}

	const ivec2 src_res = {(int)src_img_buf.shape[1], (int)src_img_buf.shape[0]};

	static uint32_t id = 0;

	m_reproject_src_views.emplace_back();
	if (m_reproject_max_src_view_count > 0 && m_reproject_src_views.size() > (size_t)m_reproject_max_src_view_count) {
		m_reproject_src_views.pop_front();
	}

	auto& src_view = m_reproject_src_views.back();
	src_view.uid = id++;
	src_view.camera0 = src_view.camera1 = src_view.prev_camera = camera_to_world;
	src_view.device = &primary_device();
	src_view.foveation = src_view.prev_foveation = {};
	src_view.screen_center = vec2(cx, cy);
	src_view.full_resolution = src_res;
	src_view.visualized_dimension = -1;
	src_view.relative_focal_length = vec2(fx, fy) / (float)src_res[m_fov_axis];
	src_view.render_buffer = std::make_shared<CudaRenderBuffer>(std::make_shared<CudaSurface2D>());
	src_view.render_buffer->resize(src_res);
	src_view.lens = lens;

	CUDA_CHECK_THROW(cudaMemcpyAsync(
		src_view.render_buffer->frame_buffer(), src_img_buf.ptr, product(src_res) * sizeof(float) * 4, cudaMemcpyHostToDevice, m_stream.get()
	));
	CUDA_CHECK_THROW(cudaMemcpyAsync(
		src_view.render_buffer->depth_buffer(), src_depth_buf.ptr, product(src_res) * sizeof(float), cudaMemcpyHostToDevice, m_stream.get()
	));

	if (is_srgb) {
		// Convert from sRGB to linear on the GPU directly
		parallel_for_gpu(
			m_stream.get(),
			product(src_res) * 4,
			[values = (float *) src_view.render_buffer->frame_buffer()] __device__(size_t i) {
				if ((i % 4) == 3) {
					// Don't linearize the alpha channel
					return;
				}
				values[i] = srgb_to_linear(values[i]);
			}
		);
	}

	return src_view.uid;
}


pybind11::array_t<uint32_t> Testbed::src_view_ids() const {
	py::array_t<uint32_t> result({(int)m_reproject_src_views.size()});
	py::buffer_info buf = result.request();
	uint32_t* data = (uint32_t*)buf.ptr;
	for (size_t i = 0; i < m_reproject_src_views.size(); ++i) {
		data[i] = m_reproject_src_views[i].uid;
	}
	return result;
}

#ifdef NGP_GUI
py::array_t<float> Testbed::screenshot(bool linear, bool front_buffer) const {
	std::vector<float> tmp(product(m_window_res) * 4);
	glReadBuffer(front_buffer ? GL_FRONT : GL_BACK);
	glReadPixels(0, 0, m_window_res.x, m_window_res.y, GL_RGBA, GL_FLOAT, tmp.data());

	py::array_t<float> result({m_window_res.y, m_window_res.x, 4});
	py::buffer_info buf = result.request();
	float* data = (float*)buf.ptr;

	// Linear, alpha premultiplied, Y flipped
	ThreadPool{}.parallel_for<size_t>(0, m_window_res.y, [&](size_t y) {
		size_t base = y * m_window_res.x;
		size_t base_reverse = (m_window_res.y - y - 1) * m_window_res.x;
		for (uint32_t x = 0; x < m_window_res.x; ++x) {
			size_t px = base + x;
			size_t px_reverse = base_reverse + x;
			data[px_reverse * 4 + 0] = linear ? srgb_to_linear(tmp[px * 4 + 0]) : tmp[px * 4 + 0];
			data[px_reverse * 4 + 1] = linear ? srgb_to_linear(tmp[px * 4 + 1]) : tmp[px * 4 + 1];
			data[px_reverse * 4 + 2] = linear ? srgb_to_linear(tmp[px * 4 + 2]) : tmp[px * 4 + 2];
			data[px_reverse * 4 + 3] = tmp[px * 4 + 3];
		}
	});

	return result;
}
#endif

PYBIND11_MODULE(pyngp, m) {
	m.doc() = "Gen3C GUI";

	m.def("free_temporary_memory", &free_all_gpu_memory_arenas);

	py::enum_<ETestbedMode>(m, "TestbedMode")
		.value("Gen3c", ETestbedMode::Gen3c)
		.value("None", ETestbedMode::None)
		.export_values();

	m.def("mode_from_scene", &mode_from_scene);
	m.def("mode_from_string", &mode_from_string);

	py::enum_<EGroundTruthRenderMode>(m, "GroundTruthRenderMode")
		.value("Shade", EGroundTruthRenderMode::Shade)
		.value("Depth", EGroundTruthRenderMode::Depth)
		.export_values();

	py::enum_<ERenderMode>(m, "RenderMode")
		.value("AO", ERenderMode::AO)
		.value("Shade", ERenderMode::Shade)
		.value("Normals", ERenderMode::Normals)
		.value("Positions", ERenderMode::Positions)
		.value("Depth", ERenderMode::Depth)
		.value("Distortion", ERenderMode::Distortion)
		.value("Cost", ERenderMode::Cost)
		.value("Slice", ERenderMode::Slice)
		.export_values();

	py::enum_<ERandomMode>(m, "RandomMode")
		.value("Random", ERandomMode::Random)
		.value("Halton", ERandomMode::Halton)
		.value("Sobol", ERandomMode::Sobol)
		.value("Stratified", ERandomMode::Stratified)
		.export_values();

	py::enum_<ELossType>(m, "LossType")
		.value("L2", ELossType::L2)
		.value("L1", ELossType::L1)
		.value("Mape", ELossType::Mape)
		.value("Smape", ELossType::Smape)
		.value("Huber", ELossType::Huber)
		// Legacy: we used to refer to the Huber loss
		// (L2 near zero, L1 further away) as "SmoothL1".
		.value("SmoothL1", ELossType::Huber)
		.value("LogL1", ELossType::LogL1)
		.value("RelativeL2", ELossType::RelativeL2)
		.export_values();

	py::enum_<ESDFGroundTruthMode>(m, "SDFGroundTruthMode")
		.value("RaytracedMesh", ESDFGroundTruthMode::RaytracedMesh)
		.value("SpheretracedMesh", ESDFGroundTruthMode::SpheretracedMesh)
		.value("SDFBricks", ESDFGroundTruthMode::SDFBricks)
		.export_values();

	py::enum_<EMeshSdfMode>(m, "MeshSdfMode")
		.value("Watertight", EMeshSdfMode::Watertight)
		.value("Raystab", EMeshSdfMode::Raystab)
		.value("PathEscape", EMeshSdfMode::PathEscape)
		.export_values();

	py::enum_<EColorSpace>(m, "ColorSpace").value("Linear", EColorSpace::Linear).value("SRGB", EColorSpace::SRGB).export_values();

	py::enum_<ETonemapCurve>(m, "TonemapCurve")
		.value("Identity", ETonemapCurve::Identity)
		.value("ACES", ETonemapCurve::ACES)
		.value("Hable", ETonemapCurve::Hable)
		.value("Reinhard", ETonemapCurve::Reinhard)
		.export_values();

	py::enum_<ELensMode>(m, "LensMode")
		.value("Perspective", ELensMode::Perspective)
		.value("OpenCV", ELensMode::OpenCV)
		.value("FTheta", ELensMode::FTheta)
		.value("LatLong", ELensMode::LatLong)
		.value("OpenCVFisheye", ELensMode::OpenCVFisheye)
		.value("Equirectangular", ELensMode::Equirectangular)
		.value("Orthographic", ELensMode::Orthographic)
		.export_values();


	py::class_<BoundingBox>(m, "BoundingBox")
		.def(py::init<>())
		.def(py::init<const vec3&, const vec3&>())
		.def("center", &BoundingBox::center)
		.def("contains", &BoundingBox::contains)
		.def("diag", &BoundingBox::diag)
		.def("distance", &BoundingBox::distance)
		.def("distance_sq", &BoundingBox::distance_sq)
		.def("enlarge", py::overload_cast<const vec3&>(&BoundingBox::enlarge))
		.def("enlarge", py::overload_cast<const BoundingBox&>(&BoundingBox::enlarge))
		.def("get_vertices", &BoundingBox::get_vertices)
		.def("inflate", &BoundingBox::inflate)
		.def("intersection", &BoundingBox::intersection)
		.def("intersects", py::overload_cast<const BoundingBox&>(&BoundingBox::intersects, py::const_))
		.def("ray_intersect", &BoundingBox::ray_intersect)
		.def("relative_pos", &BoundingBox::relative_pos)
		.def("signed_distance", &BoundingBox::signed_distance)
		.def_readwrite("min", &BoundingBox::min)
		.def_readwrite("max", &BoundingBox::max);

	py::class_<Lens> lens(m, "Lens");
	lens.def(py::init<>()).def_readwrite("mode", &Lens::mode).def_property_readonly("params", [](py::object& obj) {
		Lens& o = obj.cast<Lens&>();
		return py::array{sizeof(o.params) / sizeof(o.params[0]), o.params, obj};
	});

	m.def("fov_to_focal_length", py::overload_cast<int, float>(&ngp::fov_to_focal_length),
		  py::arg("resolution"), py::arg("degrees"))
	 .def("fov_to_focal_length", py::overload_cast<const ivec2&, const vec2&>(&fov_to_focal_length),
		  py::arg("resolution"), py::arg("degrees"))
	 .def("focal_length_to_fov", py::overload_cast<int, float>(&ngp::focal_length_to_fov),
		  py::arg("resolution"), py::arg("focal_length"))
	 .def("focal_length_to_fov", py::overload_cast<const ivec2&, const vec2&>(&ngp::focal_length_to_fov),
		  py::arg("resolution"), py::arg("focal_length"))
	 .def("relative_focal_length_to_fov", &ngp::relative_focal_length_to_fov,
		  py::arg("rel_focal_length"));

	py::class_<fs::path>(m, "path").def(py::init<>()).def(py::init<const std::string&>());

	py::implicitly_convertible<std::string, fs::path>();

	py::class_<Testbed> testbed(m, "Testbed");
	testbed.def(py::init<ETestbedMode>(), py::arg("mode") = ETestbedMode::None)
		.def_readonly("mode", &Testbed::m_testbed_mode)
		// General control
		.def(
			"init_window",
			&Testbed::init_window,
			"Init a GLFW window that shows real-time progress and a GUI. 'second_window' creates a second copy of the output in its own window.",
			py::arg("width"),
			py::arg("height"),
			py::arg("hidden") = false,
			py::arg("second_window") = false
		)
		.def("destroy_window", &Testbed::destroy_window, "Destroy the window again.")
		.def(
			"init_vr",
			&Testbed::init_vr,
			"Init rendering to a connected and active VR headset. Requires a window to have been previously created via `init_window`."
		)
		.def(
			"view",
			&Testbed::view,
			"Outputs the currently displayed image by a given view (0 by default).",
			py::arg("linear") = true,
			py::arg("view") = 0
		)
		.def("view_camera", &Testbed::view_camera, "Outputs the current camera matrix of a given view (0 by default).", py::arg("view") = 0)
		.def(
			"add_src_view",
			&Testbed::add_src_view,
			"Adds a source view to the pool of views for reprojection.",
			py::arg("camera_to_world"),
			py::arg("fx"),
			py::arg("fy"),
			py::arg("cx"),
			py::arg("cy"),
			py::arg("img"),
			py::arg("depth"),
			py::arg("lens"),
			py::arg("timestamp"),
			py::arg("is_srgb") = false
		)
		.def("src_view_ids", &Testbed::src_view_ids, "Returns the IDs of all source views currently registered.")
		.def("clear_src_views", &Testbed::clear_src_views, "Remove all views from the pool of views for reprojection.")
#ifdef NGP_GUI
		.def_readwrite("keyboard_event_callback", &Testbed::m_keyboard_event_callback)
		.def_readwrite("file_drop_callback", &Testbed::m_file_drop_callback)
		.def("is_key_pressed", [](py::object& obj, int key) { return ImGui::IsKeyPressed(key); })
		.def("is_key_down", [](py::object& obj, int key) { return ImGui::IsKeyDown(key); })
		.def("is_alt_down", [](py::object& obj) { return ImGui::GetIO().KeyMods & ImGuiKeyModFlags_Alt; })
		.def("is_ctrl_down", [](py::object& obj) { return ImGui::GetIO().KeyMods & ImGuiKeyModFlags_Ctrl; })
		.def("is_shift_down", [](py::object& obj) { return ImGui::GetIO().KeyMods & ImGuiKeyModFlags_Shift; })
		.def("is_super_down", [](py::object& obj) { return ImGui::GetIO().KeyMods & ImGuiKeyModFlags_Super; })
		.def(
			"screenshot",
			&Testbed::screenshot,
			"Takes a screenshot of the current window contents.",
			py::arg("linear") = true,
			py::arg("front_buffer") = true
		)
		.def_readwrite("vr_use_hidden_area_mask", &Testbed::m_vr_use_hidden_area_mask)
		.def_readwrite("vr_use_depth_reproject", &Testbed::m_vr_use_depth_reproject)
#endif
		.def("want_repl", &Testbed::want_repl, "returns true if the user clicked the 'I want a repl' button")
		.def(
			"frame", &Testbed::frame, py::call_guard<py::gil_scoped_release>(), "Process a single frame. Renders if a window was previously created."
		)
		.def(
			"render",
			&Testbed::render_to_cpu_rgba,
			"Renders an image at the requested resolution. Does not require a window.",
			py::arg("width") = 1920,
			py::arg("height") = 1080,
			py::arg("spp") = 1,
			py::arg("linear") = true,
			py::arg("start_t") = -1.f,
			py::arg("end_t") = -1.f,
			py::arg("fps") = 30.f,
			py::arg("shutter_fraction") = 1.0f
		)
		.def(
			"render_with_depth",
			&Testbed::render_to_cpu,
			"Renders an image at the requested resolution. Does not require a window.",
			py::arg("width") = 1920,
			py::arg("height") = 1080,
			py::arg("spp") = 1,
			py::arg("linear") = true,
			py::arg("start_t") = -1.f,
			py::arg("end_t") = -1.f,
			py::arg("fps") = 30.f,
			py::arg("shutter_fraction") = 1.0f
		)
		.def("reproject", &Testbed::reproject, "Reprojects an RGBA + depth image from a known camera view to another camera view.")
		.def("reset_camera", &Testbed::reset_camera, "Reset camera to default state.")
		.def(
			"reset_accumulation",
			&Testbed::reset_accumulation,
			"Reset rendering accumulation.",
			py::arg("due_to_camera_movement") = false,
			py::arg("immediate_redraw") = true,
			py::arg("reset_pip") = false
		)
		.def("load_camera_path", &Testbed::load_camera_path, py::arg("path"), "Load a camera path")
		.def(
			"load_file",
			&Testbed::load_file,
			py::arg("path"),
			"Load a file and automatically determine how to handle it. Can be a snapshot, dataset, network config, or camera path."
		)
		.def_property("loop_animation", &Testbed::loop_animation, &Testbed::set_loop_animation)
		// Interesting members.
		.def_readwrite("reproject_min_t", &Testbed::m_reproject_min_t)
		.def_readwrite("reproject_step_factor", &Testbed::m_reproject_step_factor)
		.def_readwrite("reproject_parallax", &Testbed::m_reproject_parallax)
		.def_readwrite("reproject_second_view", &Testbed::m_reproject_enable)
		.def_readwrite("reproject_enable", &Testbed::m_reproject_enable)
		.def_readwrite("reproject_visualize_src_views", &Testbed::m_reproject_visualize_src_views)
		.def_readwrite("reproject_min_src_view_index", &Testbed::m_reproject_min_src_view_index)
		.def_readwrite("reproject_max_src_view_index", &Testbed::m_reproject_max_src_view_index)
		.def_readwrite("reproject_max_src_view_count", &Testbed::m_reproject_max_src_view_count)
		.def("reproject_src_views_count", [](const Testbed& testbed) { return testbed.m_reproject_src_views.size(); })
		.def_readwrite("reproject_reuse_last_frame", &Testbed::m_reproject_reuse_last_frame)
		.def("init_camera_path_from_reproject_src_cameras", &Testbed::init_camera_path_from_reproject_src_cameras)
		.def_readwrite("pm_enable", &Testbed::m_pm_enable)
		.def_readwrite("dynamic_res", &Testbed::m_dynamic_res)
		.def_readwrite("dynamic_res_target_fps", &Testbed::m_dynamic_res_target_fps)
		.def_readwrite("fixed_res_factor", &Testbed::m_fixed_res_factor)
		.def_readwrite("background_color", &Testbed::m_background_color)
		.def_readwrite("render_transparency_as_checkerboard", &Testbed::m_render_transparency_as_checkerboard)
		.def_readwrite("render_groundtruth", &Testbed::m_render_ground_truth)
		.def_readwrite("render_ground_truth", &Testbed::m_render_ground_truth)
		.def_readwrite("groundtruth_render_mode", &Testbed::m_ground_truth_render_mode)
		.def_readwrite("render_mode", &Testbed::m_render_mode)
		.def_readwrite("render_near_distance", &Testbed::m_render_near_distance)
		.def_readwrite("slice_plane_z", &Testbed::m_slice_plane_z)
		.def_readwrite("dof", &Testbed::m_aperture_size)
		.def_readwrite("aperture_size", &Testbed::m_aperture_size)
		.def_readwrite("autofocus", &Testbed::m_autofocus)
		.def_readwrite("autofocus_target", &Testbed::m_autofocus_target)
		.def_readwrite("camera_path", &Testbed::m_camera_path)
		.def_readwrite("record_camera_path", &Testbed::m_record_camera_path)
		.def_readwrite("floor_enable", &Testbed::m_floor_enable)
		.def_readwrite("exposure", &Testbed::m_exposure)
		.def_property("scale", &Testbed::scale, &Testbed::set_scale)
		.def_readonly("bounding_radius", &Testbed::m_bounding_radius)
		.def_readwrite("render_aabb", &Testbed::m_render_aabb)
		.def_readwrite("render_aabb_to_local", &Testbed::m_render_aabb_to_local)
		.def_readwrite("is_rendering", &Testbed::m_render)
		.def_readwrite("aabb", &Testbed::m_aabb)
		.def_readwrite("raw_aabb", &Testbed::m_raw_aabb)
		.def_property("fov", &Testbed::fov, &Testbed::set_fov)
		.def_property("fov_xy", &Testbed::fov_xy, &Testbed::set_fov_xy)
		.def_readwrite("fov_axis", &Testbed::m_fov_axis)
		.def_readwrite("relative_focal_length", &Testbed::m_relative_focal_length)
		.def_readwrite("zoom", &Testbed::m_zoom)
		.def_readwrite("screen_center", &Testbed::m_screen_center)
		.def_readwrite("camera_matrix", &Testbed::m_camera)
		.def_readwrite("up_dir", &Testbed::m_up_dir)
		.def_readwrite("sun_dir", &Testbed::m_sun_dir)
		.def_readwrite("default_camera", &Testbed::m_default_camera)
		.def_property("look_at", &Testbed::look_at, &Testbed::set_look_at)
		.def_property("view_dir", &Testbed::view_dir, &Testbed::set_view_dir)
		.def_readwrite("camera_smoothing", &Testbed::m_camera_smoothing)
		.def_readwrite("render_with_lens_distortion", &Testbed::m_render_with_lens_distortion)
		.def_readwrite("render_lens", &Testbed::m_render_lens)
		.def_property(
			"display_gui",
			[](py::object& obj) { return obj.cast<Testbed&>().m_imgui.mode == Testbed::ImGuiMode::Enabled; },
			[](const py::object& obj, bool value) {
				obj.cast<Testbed&>().m_imgui.mode = value ? Testbed::ImGuiMode::Enabled : Testbed::ImGuiMode::Disabled;
			}
		)
		.def_property(
			"video_path",
			[](Testbed& obj) { return obj.m_imgui.video_path; },
			[](Testbed& obj, const std::string& value) {
				if (value.size() > Testbed::ImGuiVars::MAX_PATH_LEN)
					throw std::runtime_error{"Video path is too long."};
				strcpy(obj.m_imgui.video_path, value.c_str());
			}
		)
		.def_readwrite("visualize_unit_cube", &Testbed::m_visualize_unit_cube)
		.def_readwrite("snap_to_pixel_centers", &Testbed::m_snap_to_pixel_centers)
		.def_readwrite("parallax_shift", &Testbed::m_parallax_shift)
		.def_readwrite("color_space", &Testbed::m_color_space)
		.def_readwrite("tonemap_curve", &Testbed::m_tonemap_curve)
		.def_property(
			"dlss",
			[](py::object& obj) { return obj.cast<Testbed&>().m_dlss; },
			[](const py::object& obj, bool value) {
				if (value && !obj.cast<Testbed&>().m_dlss_provider) {
					if (obj.cast<Testbed&>().m_render_window) {
						throw std::runtime_error{"DLSS not supported."};
					} else {
						throw std::runtime_error{"DLSS requires a Window to be initialized via `init_window`."};
					}
				}

				obj.cast<Testbed&>().m_dlss = value;
			}
		)
		.def_readwrite("dlss_sharpening", &Testbed::m_dlss_sharpening)
		.def_property(
			"root_dir",
			[](py::object& obj) { return obj.cast<Testbed&>().root_dir().str(); },
			[](const py::object& obj, const std::string& value) { obj.cast<Testbed&>().set_root_dir(value); }
		);

	py::enum_<EGen3cCameraSource>(m, "Gen3cCameraSource")
		.value("Fake", EGen3cCameraSource::Fake)
		.value("Viewpoint", EGen3cCameraSource::Viewpoint)
		.value("Authored", EGen3cCameraSource::Authored);

	testbed
		.def(
			"set_gen3c_cb",
			[](Testbed& testbed, const Testbed::gen3c_cb_t& cb) {
				// testbed.m_gen3c_cb.reset(cb);
				testbed.m_gen3c_cb = cb;
			}
		)
		.def_readwrite("gen3c_info", &Testbed::m_gen3c_info)
		.def_readwrite("gen3c_seed_path", &Testbed::m_gen3c_seed_path)
		.def_readwrite("gen3c_auto_inference", &Testbed::m_gen3c_auto_inference)
		.def_readwrite("gen3c_camera_source", &Testbed::m_gen3c_camera_source)
		.def_readwrite("gen3c_translation_speed", &Testbed::m_gen3c_translation_speed)
		.def_readwrite("gen3c_rotation_speed", &Testbed::m_gen3c_rotation_speed)
		.def_readwrite("gen3c_inference_info", &Testbed::m_gen3c_inference_info)
		.def_readwrite("gen3c_seeding_progress", &Testbed::m_gen3c_seeding_progress)
		.def_readwrite("gen3c_inference_progress", &Testbed::m_gen3c_inference_progress)
		.def_readwrite("gen3c_inference_is_connected", &Testbed::m_gen3c_inference_is_connected)
		.def_readwrite("gen3c_render_with_gen3c", &Testbed::m_gen3c_render_with_gen3c)
		// Output
		.def_readwrite("gen3c_save_frames", &Testbed::m_gen3c_save_frames)
		.def_readwrite("gen3c_display_frames", &Testbed::m_gen3c_display_frames)
		.def_readwrite("gen3c_output_dir", &Testbed::m_gen3c_output_dir)
		.def_readwrite("gen3c_show_cache_renderings", &Testbed::m_gen3c_show_cache_renderings);

	py::class_<CameraKeyframe>(m, "CameraKeyframe")
		.def(py::init<>())
		.def(
			py::init<const quat&, const vec3&, float, float>(),
			py::arg("r"),
			py::arg("t"),
			py::arg("fov"),
			py::arg("timestamp")
		)
		.def(
			py::init<const mat4x3&, float, float>(),
			py::arg("m"),
			py::arg("fov"),
			py::arg("timestamp")
		)
		.def_readwrite("R", &CameraKeyframe::R)
		.def_readwrite("T", &CameraKeyframe::T)
		.def_readwrite("fov", &CameraKeyframe::fov)
		.def_readwrite("timestamp", &CameraKeyframe::timestamp)
		.def("m", &CameraKeyframe::m)
		.def("from_m", &CameraKeyframe::from_m, py::arg("rv"))
		.def("same_pos_as", &CameraKeyframe::same_pos_as, py::arg("rhs"));

	py::enum_<EEditingKernel>(m, "EditingKernel")
		.value("None", EEditingKernel::None)
		.value("Gaussian", EEditingKernel::Gaussian)
		.value("Quartic", EEditingKernel::Quartic)
		.value("Hat", EEditingKernel::Hat)
		.value("Box", EEditingKernel::Box);

	py::class_<CameraPath::RenderSettings>(m, "CameraPathRenderSettings")
		.def_readwrite("resolution", &CameraPath::RenderSettings::resolution)
		.def_readwrite("spp", &CameraPath::RenderSettings::spp)
		.def_readwrite("fps", &CameraPath::RenderSettings::fps)
		.def_readwrite("shutter_fraction", &CameraPath::RenderSettings::shutter_fraction)
		.def_readwrite("quality", &CameraPath::RenderSettings::quality);

	py::class_<CameraPath::Pos>(m, "CameraPathPos").def_readwrite("kfidx", &CameraPath::Pos::kfidx).def_readwrite("t", &CameraPath::Pos::t);

	py::class_<CameraPath>(m, "CameraPath")
		.def_readwrite("keyframes", &CameraPath::keyframes)
		.def_readwrite("update_cam_from_path", &CameraPath::update_cam_from_path)
		.def_readwrite("play_time", &CameraPath::play_time)
		.def_readwrite("auto_play_speed", &CameraPath::auto_play_speed)
		.def_readwrite("default_duration_seconds", &CameraPath::default_duration_seconds)
		.def_readwrite("loop", &CameraPath::loop)
		.def_readwrite("keyframe_subsampling", &CameraPath::keyframe_subsampling)
		.def_property("duration_seconds", &CameraPath::duration_seconds, &CameraPath::set_duration_seconds)
		.def_readwrite("editing_kernel_type", &CameraPath::editing_kernel_type)
		.def_readwrite("editing_kernel_radius", &CameraPath::editing_kernel_radius)
		.def_readwrite("spline_order", &CameraPath::spline_order)
		.def_readwrite("render_settings", &CameraPath::render_settings)
		.def_readwrite("rendering", &CameraPath::rendering)
		.def_readwrite("render_frame_idx", &CameraPath::render_frame_idx)
		.def_readwrite("render_start_time", &CameraPath::render_start_time)
		.def_readwrite("render_frame_end_camera", &CameraPath::render_frame_end_camera)
		.def("clear", &CameraPath::clear)
		.def("has_valid_timestamps", &CameraPath::has_valid_timestamps)
		.def("make_keyframe_timestamps_equidistant", &CameraPath::make_keyframe_timestamps_equidistant)
		.def("sanitize_keyframes", &CameraPath::sanitize_keyframes)
		.def("get_pos", &CameraPath::get_pos, py::arg("playtime"))
		.def("get_playtime", &CameraPath::get_playtime, py::arg("i"))
		.def("get_keyframe", &CameraPath::get_keyframe, py::arg("i"))
		.def("eval_camera_path", &CameraPath::eval_camera_path, py::arg("t"))
		.def("save", &CameraPath::save, py::arg("path"))
		.def("load", &CameraPath::load, py::arg("path"), py::arg("first_xform"))
		.def(
			"add_camera",
			&CameraPath::add_camera,
			py::arg("camera"),
			py::arg("fov"),
			py::arg("timestamp")
		);

	// Minimal logging framework (tlog)
	// https://github.com/Tom94/tinylogger/
	py::module_ tlog = m.def_submodule("tlog", "Tiny logging framework");
	tlog.def("none", [](const std::string &s) { tlog::none() << s; })
		.def("info", [](const std::string &s) { tlog::info() << s; })
		.def("debug", [](const std::string &s) { tlog::debug() << s; })
		.def("warning", [](const std::string &s) { tlog::warning() << s; })
		.def("error", [](const std::string &s) { tlog::error() << s; })
		.def("success", [](const std::string &s) { tlog::success() << s; });
}

} // namespace ngp
