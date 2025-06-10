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

/** @file   testbed.cu
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/json_binding.h>
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/testbed.h>

#include <tiny-cuda-nn/common_host.h>

#include <json/json.hpp>

#include <filesystem/directory.h>
#include <filesystem/path.h>

#include <playne-equivalence/playne_equivalence.cuh>

#include <fstream>
#include <unordered_set>

#ifdef NGP_GUI
#	include <imgui/backends/imgui_impl_glfw.h>
#	include <imgui/backends/imgui_impl_opengl3.h>
#	include <imgui/misc/cpp/imgui_stdlib.h>
#	include <imgui/imgui.h>
#	include <imguizmo/ImGuizmo.h>
#	ifdef _WIN32
#		include <GL/gl3w.h>
#	else
#		include <GL/glew.h>
#	endif
#	include <GLFW/glfw3.h>
#	include <GLFW/glfw3native.h>
#	include <cuda_gl_interop.h>
#endif

// Windows.h is evil
#undef min
#undef max
#undef near
#undef far

using namespace std::literals::chrono_literals;
using nlohmann::json;

namespace ngp {

int do_system(const std::string& cmd) {
#ifdef _WIN32
	tlog::info() << "> " << cmd;
	return _wsystem(utf8_to_utf16(cmd).c_str());
#else
	tlog::info() << "$ " << cmd;
	return system(cmd.c_str());
#endif
}

std::atomic<size_t> g_total_n_bytes_allocated{0};

void Testbed::update_imgui_paths() {
	snprintf(m_imgui.cam_path_path, sizeof(m_imgui.cam_path_path), "%s", (root_dir() / "cam.json").str().c_str());
	snprintf(m_imgui.video_path, sizeof(m_imgui.video_path), "%s", (root_dir() / "video.json").str().c_str());
	snprintf(m_imgui.cam_export_path, sizeof(m_imgui.cam_export_path), "%s", (root_dir() / "cam_export.json").str().c_str());
}

void Testbed::set_mode(ETestbedMode mode) {
	if (mode == m_testbed_mode) {
		return;
	}

	// Clear device-owned data that might be mode-specific
	for (auto&& device : m_devices) {
		device.clear();
	}

	m_testbed_mode = mode;

	// Set various defaults depending on mode
	m_use_aux_devices = false;

	if (m_testbed_mode == ETestbedMode::Gen3c) {
		if (m_dlss_provider && m_aperture_size == 0.0f) {
			m_dlss = true;
		}
	} else {
		m_dlss = false;
	}

	m_reproject_enable = m_testbed_mode == ETestbedMode::Gen3c;

	reset_camera();

#ifdef NGP_GUI
	update_vr_performance_settings();
#endif
}

void Testbed::load_file(const fs::path& path) {
	if (!path.exists()) {
		tlog::error() << "File '" << path.str() << "' does not exist.";
		return;
	}

	// If we get a json file, we need to parse it to determine its purpose.
	if (equals_case_insensitive(path.extension(), "json")) {
		json file;
		{
			std::ifstream f{native_string(path)};
			file = json::parse(f, nullptr, true, true);
		}

		// Camera path
		if (file.contains("path")) {
			load_camera_path(path);
			return;
		}
	}

	tlog::error() << "File '" << path.str() << "' is not a valid file to load.";
}

void Testbed::reset_accumulation(bool due_to_camera_movement, bool immediate_redraw, bool reset_pip) {
	if (immediate_redraw) {
		redraw_next_frame();
	}

	if (!due_to_camera_movement || !reprojection_available()) {
		m_windowless_render_surface.reset_accumulation();
		for (auto& view : m_views) {
			view.render_buffer->reset_accumulation();
		}
	}

	if (reset_pip) {
		m_pip_render_buffer->reset_accumulation();
	}
}

void Testbed::translate_camera(const vec3& rel, const mat3& rot, bool allow_up_down) {
	vec3 movement = rot * rel;
	if (!allow_up_down) {
		movement -= dot(movement, m_up_dir) * m_up_dir;
	}

	m_camera[3] += movement;
	reset_accumulation(true);
}

vec3 Testbed::look_at() const { return view_pos() + view_dir() * m_scale; }

void Testbed::set_look_at(const vec3& pos) { m_camera[3] += pos - look_at(); }

void Testbed::set_scale(float scale) {
	auto prev_look_at = look_at();
	m_camera[3] = (view_pos() - prev_look_at) * (scale / m_scale) + prev_look_at;
	m_scale = scale;
}

void Testbed::set_view_dir(const vec3& dir) {
	auto old_look_at = look_at();
	m_camera[0] = normalize(cross(dir, m_up_dir));
	m_camera[1] = normalize(cross(dir, m_camera[0]));
	m_camera[2] = normalize(dir);
	set_look_at(old_look_at);
}

void Testbed::reset_camera() {
	m_fov_axis = 1;
	m_zoom = 1.0f;
	m_screen_center = vec2(0.5f);

	set_fov(50.625f);
	m_scale = 1.5f;

	m_camera = m_default_camera;
	m_camera[3] -= m_scale * view_dir();

	m_smoothed_camera = m_camera;
	m_sun_dir = normalize(vec3(1.0f));

	reset_accumulation();
}

fs::path Testbed::root_dir() {
	if (m_root_dir.empty()) {
		set_root_dir(discover_root_dir());
	}

	return m_root_dir;
}

void Testbed::set_root_dir(const fs::path& dir) { m_root_dir = dir; }

inline float linear_to_db(float x) { return -10.f * logf(x) / logf(10.f); }


#ifdef NGP_GUI
bool imgui_colored_button(const char* name, float hue) {
	ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(hue, 0.6f, 0.6f));
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(hue, 0.7f, 0.7f));
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(hue, 0.8f, 0.8f));
	bool rv = ImGui::Button(name);
	ImGui::PopStyleColor(3);
	return rv;
}

void Testbed::overlay_fps() {
	ImGui::PushFont((ImFont*)m_imgui.overlay_font);
	ImGui::SetNextWindowPos({10.0f, 10.0f}, ImGuiCond_Always, {0.0f, 0.0f});
	ImGui::SetNextWindowBgAlpha(0.35f);
	if (ImGui::Begin(
			"Overlay",
			nullptr,
			ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
				ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove
		)) {
		ImGui::Text("%.1f FPS", 1000.0f / m_render_ms.ema_val());
	}
	ImGui::PopFont();
}

void Testbed::imgui() {
	// If a GUI interaction causes an error, write that error to the following string and call
	//   ImGui::OpenPopup("Error");
	static std::string imgui_error_string = "";

	m_picture_in_picture_res = 0;

	// Good default position and size for the camera path editing window
	ImGui::SetNextWindowPos({10.0f, 10.0f}, ImGuiCond_FirstUseEver);
	int window_width, window_height;
	glfwGetWindowSize(m_glfw_window, &window_width, &window_height);
	ImGui::SetNextWindowSize({420.0f, window_height - 20.0f}, ImGuiCond_FirstUseEver);

	if (ImGui::Begin("Camera path & video generation", 0, ImGuiWindowFlags_NoScrollbar)) {
		if (ImGui::CollapsingHeader("Path manipulation", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Checkbox("Record camera path", &m_record_camera_path);
			ImGui::SameLine();
			if (ImGui::Button("Clear")) {
				m_camera_path.clear();
			}

			if (m_reproject_enable) {
				ImGui::SameLine();
				if (ImGui::Button("Init from views")) {
					init_camera_path_from_reproject_src_cameras();
				}
			}

			if (int read = m_camera_path.imgui(m_imgui.cam_path_path, m_frame_ms.val(), m_camera, fov(), mat4x3::identity())) {
				if (!m_camera_path.rendering || m_gen3c_render_with_gen3c) {
					reset_accumulation(true);

					if (m_camera_path.update_cam_from_path) {
						set_camera_from_time(m_camera_path.play_time);

						// A value of larger than 1 indicates that the camera path wants
						// to override camera smoothing.
						if (read > 1) {
							m_smoothed_camera = m_camera;
						}
					} else {
						m_pip_render_buffer->reset_accumulation();
					}
				}
			}

			if (!m_camera_path.keyframes.empty()) {
				float w = ImGui::GetContentRegionAvail().x;
				if (m_camera_path.update_cam_from_path) {
					m_picture_in_picture_res = 0;
					ImGui::Image((ImTextureID)(size_t)m_rgba_render_textures.front()->texture(), ImVec2(w, w * 9.0f / 16.0f));
				} else {
					m_picture_in_picture_res = (float)std::min((int(w) + 31) & (~31), 1920 / 4);
					ImGui::Image((ImTextureID)(size_t)m_pip_render_texture->texture(), ImVec2(w, w * 9.0f / 16.0f));
				}
			}
		}

		if (!m_camera_path.keyframes.empty() && ImGui::CollapsingHeader("Video generation", ImGuiTreeNodeFlags_DefaultOpen)) {
			// Render a video
			// TODO: simplify this (only allow rendering with Gen3C).
			ImGui::BeginDisabled(m_camera_path.rendering);
			if (imgui_colored_button(m_camera_path.rendering ? "Waiting for model..." : "Generate video", 0.4)) {
				bool was_rendering = m_camera_path.rendering;
				m_camera_path.rendering = !m_camera_path.rendering;

				if (m_gen3c_render_with_gen3c) {
					if (m_gen3c_cb) {
						m_gen3c_cb(was_rendering ? "abort_inference" : "request_inference");
					}
				} else {
					if (!clear_tmp_dir()) {
						imgui_error_string = "Failed to clear temporary directory 'tmp' to hold rendered images.";
						ImGui::OpenPopup("Error");

						m_camera_path.rendering = false;
					}

					if (m_camera_path.rendering) {
						m_camera_path.render_start_time = std::chrono::steady_clock::now();
						m_camera_path.update_cam_from_path = true;
						m_camera_path.play_time = 0.0f;
						m_camera_path.auto_play_speed = 1.0f;
						m_camera_path.render_frame_idx = 0;

						m_dlss = false;

						reset_accumulation(true);
						set_camera_from_time(m_camera_path.play_time);
						m_smoothed_camera = m_camera;
					} else {
						m_camera_path.update_cam_from_path = false;
						m_camera_path.play_time = 0.0f;
						m_camera_path.auto_play_speed = 0.0f;
					}
				}
			}
			ImGui::EndDisabled();

			ImGui::SameLine();
			ImGui::BeginDisabled(!m_gen3c_inference_is_connected || !m_gen3c_cb);
			ImGui::Checkbox("Gen3C inference", &m_gen3c_render_with_gen3c);
			ImGui::EndDisabled();

			if (m_camera_path.rendering) {
				const auto elapsed = std::chrono::steady_clock::now() - m_camera_path.render_start_time;

				const float duration = m_camera_path.duration_seconds();
				const uint32_t progress = m_camera_path.render_frame_idx * m_camera_path.render_settings.spp + m_views.front().render_buffer->spp();
				const uint32_t goal = m_camera_path.render_settings.n_frames(duration) * m_camera_path.render_settings.spp;
				const auto est_remaining = elapsed * (float)(goal - progress) / std::max(progress, 1u);

				if (m_gen3c_render_with_gen3c) {
					if (!m_gen3c_inference_info.empty()) {
						ImGui::TextWrapped("%s", m_gen3c_inference_info.c_str());
					}

					if (m_gen3c_inference_progress > 0) {
						ImGui::ProgressBar(m_gen3c_inference_progress);
					}
				} else {
					ImGui::Text(
						"%s",
						fmt::format(
							"Frame {}/{}, Elapsed: {}, Remaining: {}",
							m_camera_path.render_frame_idx + 1,
							m_camera_path.render_settings.n_frames(duration),
							tlog::durationToString(std::chrono::steady_clock::now() - m_camera_path.render_start_time),
							tlog::durationToString(est_remaining)
						)
							.c_str()
					);

					ImGui::ProgressBar((float)progress / goal);
				}
			}

			ImGui::BeginDisabled(m_camera_path.rendering);

			ImGui::Checkbox("Show rendered Gen3C cache in video", &m_gen3c_show_cache_renderings);
			// Note: 3D cache visualization is incompatible with adding Gen3C frames to the viewport.
			if (m_gen3c_show_cache_renderings)
				m_gen3c_display_frames = false;
			ImGui::BeginDisabled(m_gen3c_show_cache_renderings);
			ImGui::Checkbox("Add Gen3C keyframes to viewport after inference", &m_gen3c_display_frames);
			ImGui::EndDisabled();  // m_gen3c_show_cache_renderings

			ImGui::InputText("Video file##Video file path", m_imgui.video_path, sizeof(m_imgui.video_path));
			m_camera_path.render_settings.filename = m_imgui.video_path;
			ImGui::SliderInt("MP4 quality", &m_camera_path.render_settings.quality, 0, 10);

			float duration_seconds = m_camera_path.duration_seconds();
			if (ImGui::InputFloat("Duration (seconds)", &duration_seconds) && duration_seconds > 0.0f) {
				m_camera_path.set_duration_seconds(duration_seconds);
			}

			ImGui::InputFloat("FPS (frames/second)", &m_camera_path.render_settings.fps);

			ImGui::BeginDisabled(m_gen3c_render_with_gen3c);
			ImGui::InputInt2("Resolution", &m_camera_path.render_settings.resolution.x);
			// ImGui::InputInt("SPP (samples/pixel)", &m_camera_path.render_settings.spp);
			if (m_gen3c_render_with_gen3c) {
				m_camera_path.render_settings.spp = 1;
			}
			// ImGui::SliderFloat("Shutter fraction", &m_camera_path.render_settings.shutter_fraction, 0.0f, 1.0f);
			ImGui::EndDisabled(); // end m_gen3c_render_with_gen3c

			ImGui::EndDisabled(); // end m_camera_path.rendering

			ImGui::Spacing();
			bool export_cameras = imgui_colored_button("Export cameras", 0.7);

			ImGui::SameLine();

			static bool w2c = false;
			ImGui::Checkbox("W2C", &w2c);

			ImGui::InputText("Cameras file##Camera export path", m_imgui.cam_export_path, sizeof(m_imgui.cam_export_path));
			m_camera_path.render_settings.filename = m_imgui.video_path;

			if (export_cameras) {
				std::vector<json> cameras;
				const float duration = m_camera_path.duration_seconds();
				for (uint32_t i = 0; i < m_camera_path.render_settings.n_frames(duration); ++i) {
					mat4x3 start_cam = m_camera_path.eval_camera_path((float)i / (m_camera_path.render_settings.n_frames(duration))).m();
					mat4x3 end_cam = m_camera_path
										 .eval_camera_path(
											 ((float)i + m_camera_path.render_settings.shutter_fraction) /
											 (m_camera_path.render_settings.n_frames(duration))
										 )
										 .m();
					if (w2c) {
						start_cam = inverse(mat4x4(start_cam));
						end_cam = inverse(mat4x4(end_cam));
					}

					cameras.push_back({
						{"start", start_cam},
						{"end",   end_cam  },
					});
				}

				json j;
				j["cameras"] = cameras;
				j["resolution"] = m_camera_path.render_settings.resolution;
				j["duration_seconds"] = m_camera_path.duration_seconds();
				j["fps"] = m_camera_path.render_settings.fps;
				j["spp"] = m_camera_path.render_settings.spp;
				j["quality"] = m_camera_path.render_settings.quality;
				j["shutter_fraction"] = m_camera_path.render_settings.shutter_fraction;

				std::ofstream f(native_string(m_imgui.cam_export_path));
				f << j;
			}
		}
	}
	ImGui::End();

	// Good default position and size for the right-hand side window
	int pane_width = 350;
	ImGui::SetNextWindowPos({window_width - pane_width - 10.0f, 10.0f}, ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize({(float)pane_width, window_height - 20.0f}, ImGuiCond_FirstUseEver);

	ImGui::Begin("Gen3C v" NGP_VERSION);

	size_t n_bytes = tcnn::total_n_bytes_allocated() + g_total_n_bytes_allocated;
	if (m_dlss_provider) {
		n_bytes += m_dlss_provider->allocated_bytes();
	}

	ImGui::Text("Frame: %.2f ms (%.1f FPS); Mem: %s", m_frame_ms.ema_val(), 1000.0f / m_frame_ms.ema_val(), bytes_to_string(n_bytes).c_str());
	bool accum_reset = false;

	if (m_testbed_mode == ETestbedMode::Gen3c && ImGui::CollapsingHeader("Video generation server", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::TextWrapped("%s", m_gen3c_info.c_str());
		ImGui::Spacing();

		// Create a child box with a title and borders
		if (ImGui::TreeNodeEx("Seeding", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::TextWrapped("Enter the path to an image or a pre-processed video directory.");
			ImGui::InputText("Path", &m_gen3c_seed_path);

			ImGui::BeginDisabled(m_gen3c_seed_path.empty());
			if (ImGui::Button("Seed") && m_gen3c_cb) {
				m_gen3c_cb("seed_model");
			}
			if (m_gen3c_seeding_progress > 0) {
				ImGui::ProgressBar(m_gen3c_seeding_progress);
			}
			ImGui::EndDisabled();

			ImGui::Spacing();
			ImGui::TreePop();
		}

		// ImGui::Separator();

		// We need this to be executed even if the panel below is collapsed.
		switch (m_gen3c_camera_source) {
			case EGen3cCameraSource::Fake: {
				m_gen3c_auto_inference = false;
				break;
			}
			case EGen3cCameraSource::Viewpoint: {
				break;
			}
			case EGen3cCameraSource::Authored: {
				m_gen3c_auto_inference = false;
				break;
			}
			default: throw std::runtime_error("Unsupported Gen3C camera source.");
		}

	}

	if (ImGui::CollapsingHeader("Point cloud", ImGuiTreeNodeFlags_DefaultOpen)) {
		// accum_reset |= ImGui::Checkbox("Enable reprojection", &m_reproject_enable);
		if (m_reproject_enable) {
			int max_views = (int)m_reproject_src_views.size();

			int prev_min_src_view_index = m_reproject_min_src_view_index;
			int prev_max_src_view_index = m_reproject_max_src_view_index;
			int prev_n_frames_shown = std::max(0, prev_max_src_view_index - prev_min_src_view_index);

			if (ImGui::SliderInt("Min view index", &m_reproject_min_src_view_index, 0, max_views)) {
				// If shift, move the range synchronously.
				if (ImGui::GetIO().KeyShift) {
					m_reproject_max_src_view_index =
						std::min(m_reproject_max_src_view_index + m_reproject_min_src_view_index - prev_min_src_view_index, max_views);
					// Keep the number of frames shown constant.
					m_reproject_min_src_view_index = m_reproject_max_src_view_index - prev_n_frames_shown;
				}

				// Ensure that range remains valid (max index >= min index).
				m_reproject_max_src_view_index = std::max(m_reproject_max_src_view_index, m_reproject_min_src_view_index);
				accum_reset = true;
			}

			if (ImGui::SliderInt("Max view index", &m_reproject_max_src_view_index, 0, max_views)) {
				// If shift, move the range synchronously.
				if (ImGui::GetIO().KeyShift) {
					m_reproject_min_src_view_index =
						std::max(m_reproject_min_src_view_index + m_reproject_max_src_view_index - prev_max_src_view_index, 0);
					// Keep the number of frames shown constant.
					m_reproject_max_src_view_index = m_reproject_min_src_view_index + prev_n_frames_shown;
				}
				// Ensure that range remains valid (max index >= min index).
				m_reproject_min_src_view_index = std::min(m_reproject_max_src_view_index, m_reproject_min_src_view_index);
				accum_reset = true;
			}

			if (max_views > 0 && ImGui::SliderInt("Snap to view", (int*)&m_reproject_selected_src_view, 0, max_views - 1)) {
				m_camera = m_smoothed_camera =
					m_reproject_src_views[std::min((size_t)m_reproject_selected_src_view, m_reproject_src_views.size() - 1)].camera0;
				accum_reset = true;
			}

			accum_reset |= ImGui::Checkbox("Visualize views", &m_reproject_visualize_src_views);
			ImGui::SameLine();
			if (ImGui::Button("Delete views")) {
				clear_src_views();
			}

			if (ImGui::TreeNodeEx("Advanced reprojection settings")) {
				accum_reset |= ImGui::SliderFloat(
					"Reproject min t", &m_reproject_min_t, 0.01f, 16.0f, "%.01f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat
				);
				accum_reset |= ImGui::SliderFloat(
					"Reproject scaling", &m_reproject_step_factor, 1.003f, 1.5f, "%.001f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat
				);

				accum_reset |= ImGui::Combo("Reproject render mode", (int*)&m_pm_viz_mode, PmVizModeStr);

				ImGui::TreePop();
			}

		}
	}

	if (ImGui::CollapsingHeader("Rendering", m_testbed_mode == ETestbedMode::Gen3c ? 0 : ImGuiTreeNodeFlags_DefaultOpen)) {

		ImGui::Checkbox("Render", &m_render);
		ImGui::SameLine();

		const auto& render_buffer = m_views.front().render_buffer;
		std::string spp_string = m_dlss ? std::string{""} : fmt::format("({} spp)", std::max(render_buffer->spp(), 1u));
		ImGui::Text(
			": %.01fms for %dx%d %s",
			m_render_ms.ema_val(),
			render_buffer->in_resolution().x,
			render_buffer->in_resolution().y,
			spp_string.c_str()
		);

		ImGui::SameLine();
		if (ImGui::Checkbox("VSync", &m_vsync)) {
			glfwSwapInterval(m_vsync ? 1 : 0);
		}


		ImGui::Checkbox("Dynamic resolution", &m_dynamic_res);
		ImGui::SameLine();
		ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.3f);
		if (m_dynamic_res) {
			ImGui::SliderFloat(
				"Target FPS", &m_dynamic_res_target_fps, 2.0f, 144.0f, "%.01f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat
			);
		} else {
			ImGui::SliderInt("Resolution factor", &m_fixed_res_factor, 8, 64);
		}
		ImGui::PopItemWidth();

		if (ImGui::TreeNode("Advanced rendering options")) {
			accum_reset |= ImGui::Combo("Render mode", (int*)&m_render_mode, RenderModeStr);
			accum_reset |= ImGui::Combo("Tonemap curve", (int*)&m_tonemap_curve, TonemapCurveStr);
			accum_reset |= ImGui::ColorEdit4("Background", &m_background_color[0]);

			if (ImGui::SliderFloat("Exposure", &m_exposure, -5.f, 5.f)) {
				set_exposure(m_exposure);
			}

			ImGui::SliderInt("Max spp", &m_max_spp, 0, 1024, "%d", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat);
			accum_reset |= ImGui::Checkbox("Render transparency as checkerboard", &m_render_transparency_as_checkerboard);
			accum_reset |= ImGui::Combo("Color space", (int*)&m_color_space, ColorSpaceStr);
			accum_reset |= ImGui::Checkbox("Snap to pixel centers", &m_snap_to_pixel_centers);

			ImGui::TreePop();
		}
	}

	if (ImGui::CollapsingHeader("Camera")) {
		ImGui::Checkbox("First person controls", &m_fps_camera);
		ImGui::SameLine();
		ImGui::Checkbox("Smooth motion", &m_camera_smoothing);

		float local_fov = fov();
		if (ImGui::SliderFloat("Field of view", &local_fov, 0.0f, 120.0f)) {
			set_fov(local_fov);
			accum_reset = true;
		}

		if (ImGui::TreeNode("Advanced camera settings")) {
			accum_reset |= ImGui::SliderFloat2("Screen center", &m_screen_center.x, 0.f, 1.f);
			accum_reset |= ImGui::SliderFloat2("Parallax shift", &m_parallax_shift.x, -1.f, 1.f);
			accum_reset |= ImGui::SliderFloat("Slice / focus depth", &m_slice_plane_z, -m_bounding_radius, m_bounding_radius);
			accum_reset |= ImGui::SliderFloat(
				"Render near distance", &m_render_near_distance, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat
			);

			bool lens_changed = ImGui::Checkbox("Apply lens distortion", &m_render_with_lens_distortion);
			if (m_render_with_lens_distortion) {
				lens_changed |= ImGui::Combo("Lens mode", (int*)&m_render_lens.mode, LensModeStr);
				if (m_render_lens.mode == ELensMode::OpenCV) {
					accum_reset |= ImGui::InputFloat("k1", &m_render_lens.params[0], 0.f, 0.f, "%.5f");
					accum_reset |= ImGui::InputFloat("k2", &m_render_lens.params[1], 0.f, 0.f, "%.5f");
					accum_reset |= ImGui::InputFloat("p1", &m_render_lens.params[2], 0.f, 0.f, "%.5f");
					accum_reset |= ImGui::InputFloat("p2", &m_render_lens.params[3], 0.f, 0.f, "%.5f");
				} else if (m_render_lens.mode == ELensMode::OpenCVFisheye) {
					accum_reset |= ImGui::InputFloat("k1", &m_render_lens.params[0], 0.f, 0.f, "%.5f");
					accum_reset |= ImGui::InputFloat("k2", &m_render_lens.params[1], 0.f, 0.f, "%.5f");
					accum_reset |= ImGui::InputFloat("k3", &m_render_lens.params[2], 0.f, 0.f, "%.5f");
					accum_reset |= ImGui::InputFloat("k4", &m_render_lens.params[3], 0.f, 0.f, "%.5f");
				} else if (m_render_lens.mode == ELensMode::FTheta) {
					accum_reset |= ImGui::InputFloat("width", &m_render_lens.params[5], 0.f, 0.f, "%.0f");
					accum_reset |= ImGui::InputFloat("height", &m_render_lens.params[6], 0.f, 0.f, "%.0f");
					accum_reset |= ImGui::InputFloat("f_theta p0", &m_render_lens.params[0], 0.f, 0.f, "%.5f");
					accum_reset |= ImGui::InputFloat("f_theta p1", &m_render_lens.params[1], 0.f, 0.f, "%.5f");
					accum_reset |= ImGui::InputFloat("f_theta p2", &m_render_lens.params[2], 0.f, 0.f, "%.5f");
					accum_reset |= ImGui::InputFloat("f_theta p3", &m_render_lens.params[3], 0.f, 0.f, "%.5f");
					accum_reset |= ImGui::InputFloat("f_theta p4", &m_render_lens.params[4], 0.f, 0.f, "%.5f");
				}

				if (lens_changed && !m_render_lens.supports_dlss()) {
					m_dlss = false;
				}
			}
			ImGui::Spacing();

			accum_reset |= lens_changed;

			char buf[2048];
			vec3 v = view_dir();
			vec3 p = look_at();
			vec3 s = m_sun_dir;
			vec3 u = m_up_dir;
			vec4 b = m_background_color;
			snprintf(
				buf,
				sizeof(buf),
				"testbed.background_color = [%0.3f, %0.3f, %0.3f, %0.3f]\n"
				"testbed.exposure = %0.3f\n"
				"testbed.sun_dir = [%0.3f,%0.3f,%0.3f]\n"
				"testbed.up_dir = [%0.3f,%0.3f,%0.3f]\n"
				"testbed.view_dir = [%0.3f,%0.3f,%0.3f]\n"
				"testbed.look_at = [%0.3f,%0.3f,%0.3f]\n"
				"testbed.scale = %0.3f\n"
				"testbed.fov,testbed.aperture_size,testbed.slice_plane_z = %0.3f,%0.3f,%0.3f\n"
				"testbed.autofocus_target = [%0.3f,%0.3f,%0.3f]\n"
				"testbed.autofocus = %s\n\n",
				b.r,
				b.g,
				b.b,
				b.a,
				m_exposure,
				s.x,
				s.y,
				s.z,
				u.x,
				u.y,
				u.z,
				v.x,
				v.y,
				v.z,
				p.x,
				p.y,
				p.z,
				scale(),
				fov(),
				m_aperture_size,
				m_slice_plane_z,
				m_autofocus_target.x,
				m_autofocus_target.y,
				m_autofocus_target.z,
				m_autofocus ? "True" : "False"
			);

			ImGui::InputTextMultiline("Params", buf, sizeof(buf));
			ImGui::TreePop();
		}
	}

	if (ImGui::BeginPopupModal("Error", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
		ImGui::Text("%s", imgui_error_string.c_str());
		if (ImGui::Button("OK", ImVec2(120, 0))) {
			ImGui::CloseCurrentPopup();
		}
		ImGui::EndPopup();
	}

	if (accum_reset) {
		reset_accumulation();
	}

	if (ImGui::Button("Go to Python REPL")) {
		m_want_repl = true;
	}

	ImGui::End();
}

void Testbed::init_camera_path_from_reproject_src_cameras() {
	m_camera_path.clear();

	for (int i = m_reproject_min_src_view_index; i < std::min(m_reproject_max_src_view_index, (int)m_reproject_src_views.size()); ++i) {
		const auto& view = m_reproject_src_views[i];
		m_camera_path.add_camera(
			view.camera0,
			view.fov()[m_fov_axis],
			0.0f // timestamp set to zero: camera path treats keyframes as temporally equidistant
		);
	}

	m_camera_path.keyframe_subsampling = (int)m_camera_path.keyframes.size();
	m_camera_path.editing_kernel_type = EEditingKernel::Gaussian;
}

void Testbed::visualize_reproject_src_cameras(ImDrawList* list, const mat4& world2proj) {
	for (size_t i = (size_t)m_reproject_min_src_view_index;
		 i < std::min((size_t)m_reproject_max_src_view_index, m_reproject_src_views.size());
		 ++i) {
		const auto& view = m_reproject_src_views[i];
		auto res = view.full_resolution;
		float aspect = float(res.x) / float(res.y);

		visualize_camera(list, world2proj, view.camera0, aspect, 0xffffffff);
	}
}

void Testbed::clear_src_views() {
	m_reproject_src_views.clear();
	reset_accumulation();
}

void Testbed::draw_visualizations(ImDrawList* list, const mat4x3& camera_matrix) {
	mat4 view2world = camera_matrix;
	mat4 world2view = inverse(view2world);

	auto focal = calc_focal_length(ivec2(1), m_relative_focal_length, m_fov_axis, m_zoom);
	float zscale = 1.0f / focal[m_fov_axis];

	float xyscale = (float)m_window_res[m_fov_axis];
	vec2 screen_center = render_screen_center(m_screen_center);
	mat4 view2proj = transpose(
		mat4{
			xyscale,
			0.0f,
			(float)m_window_res.x * screen_center.x * zscale,
			0.0f,
			0.0f,
			xyscale,
			(float)m_window_res.y * screen_center.y * zscale,
			0.0f,
			0.0f,
			0.0f,
			1.0f,
			0.0f,
			0.0f,
			0.0f,
			zscale,
			0.0f,
		}
	);

	mat4 world2proj = view2proj * world2view;
	float aspect = (float)m_window_res.x / (float)m_window_res.y;

	if (m_reproject_visualize_src_views) {
		visualize_reproject_src_cameras(list, world2proj);
	}

	if (m_visualize_unit_cube) {
		visualize_cube(list, world2proj, vec3(0.f), vec3(1.f), mat3::identity());
	}

	if (m_edit_render_aabb) {
		ImGuiIO& io = ImGui::GetIO();
		// float flx = focal.x;
		float fly = focal.y;
		float zfar = m_ndc_zfar;
		float znear = m_ndc_znear;
		mat4 view2proj_guizmo = transpose(
			mat4{
				fly * 2.0f / aspect,
				0.0f,
				0.0f,
				0.0f,
				0.0f,
				-fly * 2.f,
				0.0f,
				0.0f,
				0.0f,
				0.0f,
				(zfar + znear) / (zfar - znear),
				-(2.0f * zfar * znear) / (zfar - znear),
				0.0f,
				0.0f,
				1.0f,
				0.0f,
			}
		);

		ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);

		static mat4 matrix = mat4::identity();
		static mat4 world2view_guizmo = mat4::identity();

		vec3 cen = transpose(m_render_aabb_to_local) * m_render_aabb.center();
		if (!ImGuizmo::IsUsing()) {
			// The the guizmo is being used, it handles updating its matrix on its own.
			// Outside interference can only lead to trouble.
			auto rot = transpose(m_render_aabb_to_local);
			matrix = mat4(mat4x3(rot[0], rot[1], rot[2], cen));

			// Additionally, the world2view transform must stay fixed, else the guizmo will incorrectly
			// interpret the state from past frames. Special handling is necessary here, because below
			// we emulate world translation and rotation through (inverse) camera movement.
			world2view_guizmo = world2view;
		}

		auto prev_matrix = matrix;

		if (ImGuizmo::Manipulate(
				(const float*)&world2view_guizmo, (const float*)&view2proj_guizmo, m_camera_path.m_gizmo_op, ImGuizmo::LOCAL, (float*)&matrix, NULL, NULL
			)) {
			if (m_edit_world_transform) {
				// We transform the world by transforming the camera in the opposite direction.
				auto rel = prev_matrix * inverse(matrix);
				m_camera = mat3(rel) * m_camera;
				m_camera[3] += rel[3].xyz();

				m_up_dir = mat3(rel) * m_up_dir;
			} else {
				m_render_aabb_to_local = transpose(mat3(matrix));
				vec3 new_cen = m_render_aabb_to_local * matrix[3].xyz();
				vec3 old_cen = m_render_aabb.center();
				m_render_aabb.min += new_cen - old_cen;
				m_render_aabb.max += new_cen - old_cen;
			}

			reset_accumulation();
		}
	}


	if (m_camera_path.imgui_viz(
			list,
			view2proj,
			world2proj,
			world2view,
			focal,
			aspect,
			m_ndc_znear,
			m_ndc_zfar
		)) {
		m_pip_render_buffer->reset_accumulation();
	}
}

void glfw_error_callback(int error, const char* description) { tlog::error() << "GLFW error #" << error << ": " << description; }

bool Testbed::keyboard_event() {
	if (ImGui::GetIO().WantCaptureKeyboard) {
		return false;
	}

	if (m_keyboard_event_callback && m_keyboard_event_callback()) {
		return false;
	}

	if (ImGui::IsKeyPressed('Q') && ImGui::GetIO().KeyCtrl) {
		glfwSetWindowShouldClose(m_glfw_window, GLFW_TRUE);
	}

	if ((ImGui::IsKeyPressed(GLFW_KEY_TAB) || ImGui::IsKeyPressed(GLFW_KEY_GRAVE_ACCENT)) && !ImGui::GetIO().KeyCtrl) {
		m_imgui.mode = (ImGuiMode)(((uint32_t)m_imgui.mode + 1) % (uint32_t)ImGuiMode::NumModes);
	}

	for (int idx = 0; idx < std::min((int)ERenderMode::NumRenderModes, 10); ++idx) {
		char c[] = {"1234567890"};
		if (ImGui::IsKeyPressed(c[idx])) {
			m_render_mode = (ERenderMode)idx;
			reset_accumulation();
		}
	}

	bool ctrl = ImGui::GetIO().KeyCtrl;
	bool shift = ImGui::GetIO().KeyShift;

	if (ImGui::IsKeyPressed('Z')) {
		m_camera_path.m_gizmo_op = ImGuizmo::TRANSLATE;
	}

	if (ImGui::IsKeyPressed('X')) {
		m_camera_path.m_gizmo_op = ImGuizmo::ROTATE;
	}

	if (ImGui::IsKeyPressed('E')) {
		set_exposure(m_exposure + (shift ? -0.5f : 0.5f));
		redraw_next_frame();
	}

	if (ImGui::IsKeyPressed('R')) {
		reset_camera();
	}

	if (ImGui::IsKeyPressed('=') || ImGui::IsKeyPressed('+')) {
		if (m_fps_camera) {
			m_camera_velocity *= 1.5f;
		} else {
			set_scale(m_scale * 1.1f);
		}
	}

	if (ImGui::IsKeyPressed('-') || ImGui::IsKeyPressed('_')) {
		if (m_fps_camera) {
			m_camera_velocity /= 1.5f;
		} else {
			set_scale(m_scale / 1.1f);
		}
	}

	// WASD camera movement
	vec3 translate_vec = vec3(0.0f);
	if (ImGui::IsKeyDown('W')) {
		translate_vec.z += 1.0f;
	}

	if (ImGui::IsKeyDown('A')) {
		translate_vec.x += -1.0f;
	}

	if (ImGui::IsKeyDown('S')) {
		translate_vec.z += -1.0f;
	}

	if (ImGui::IsKeyDown('D')) {
		translate_vec.x += 1.0f;
	}

	if (ImGui::IsKeyDown(' ')) {
		translate_vec.y += -1.0f;
	}

	if (ImGui::IsKeyDown('C')) {
		translate_vec.y += 1.0f;
	}

	translate_vec *= m_camera_velocity * m_frame_ms.val() / 1000.0f;
	if (shift) {
		translate_vec *= 5.0f;
	}

	if (translate_vec != vec3(0.0f)) {
		m_fps_camera = true;

		// If VR is active, movement that isn't aligned with the current view
		// direction is _very_ jarring to the user, so make keyboard-based
		// movement aligned with the VR view, even though it is not an intended
		// movement mechanism. (Users should use controllers.)
		translate_camera(translate_vec, m_hmd && m_hmd->is_visible() ? mat3(m_views.front().camera0) : mat3(m_camera));
	}

	return false;
}

void Testbed::mouse_wheel() {
	float delta = ImGui::GetIO().MouseWheel;
	if (delta == 0) {
		return;
	}

	float scale_factor = pow(1.1f, -delta);
	set_scale(m_scale * scale_factor);

	reset_accumulation(true);
}

mat3 Testbed::rotation_from_angles(const vec2& angles) const {
	vec3 up = m_up_dir;
	vec3 side = m_camera[0];
	return rotmat(angles.x, up) * rotmat(angles.y, side);
}

void Testbed::mouse_drag() {
	vec2 rel = vec2{ImGui::GetIO().MouseDelta.x, ImGui::GetIO().MouseDelta.y} / (float)m_window_res[m_fov_axis];
	vec2 mouse = {ImGui::GetMousePos().x, ImGui::GetMousePos().y};

	vec3 side = m_camera[0];

	bool shift = ImGui::GetIO().KeyShift;

	// Left pressed
	if (ImGui::GetIO().MouseClicked[0] && shift) {
		m_autofocus_target = get_3d_pos_from_pixel(*m_views.front().render_buffer, mouse);
		m_autofocus = true;

		reset_accumulation();
	}

	// Left held
	if (ImGui::GetIO().MouseDown[0]) {
		float rot_sensitivity = m_fps_camera ? 0.35f : 1.0f;
		mat3 rot = rotation_from_angles(-rel * 2.0f * PI() * rot_sensitivity);

		if (m_fps_camera) {
			rot *= mat3(m_camera);
			m_camera = mat4x3(rot[0], rot[1], rot[2], m_camera[3]);
		} else {
			// Turntable
			auto old_look_at = look_at();
			set_look_at({0.0f, 0.0f, 0.0f});
			m_camera = rot * m_camera;
			set_look_at(old_look_at);
		}

		reset_accumulation(true);
	}

	// Right held
	if (ImGui::GetIO().MouseDown[1]) {
		mat3 rot = rotation_from_angles(-rel * 2.0f * PI());
		if (m_render_mode == ERenderMode::Shade) {
			m_sun_dir = transpose(rot) * m_sun_dir;
		}

		m_slice_plane_z += -rel.y * m_bounding_radius;
		reset_accumulation();
	}

	// Middle pressed
	if (ImGui::GetIO().MouseClicked[2]) {
		m_drag_depth = get_depth_from_renderbuffer(*m_views.front().render_buffer, mouse / vec2(m_window_res));
	}

	// Middle held
	if (ImGui::GetIO().MouseDown[2]) {
		vec3 translation = vec3{-rel.x, -rel.y, 0.0f} / m_zoom;
		bool is_orthographic = m_render_with_lens_distortion && m_render_lens.mode == ELensMode::Orthographic;

		translation /= m_relative_focal_length[m_fov_axis];

		// If we have a valid depth value, scale the scene translation by it such that the
		// hovered point in 3D space stays under the cursor.
		if (m_drag_depth < 256.0f && !is_orthographic) {
			translation *= m_drag_depth;
		}

		translate_camera(translation, mat3(m_camera));
	}
}

bool Testbed::begin_frame() {
	if (glfwWindowShouldClose(m_glfw_window)) {
		destroy_window();
		return false;
	}

	{
		auto now = std::chrono::steady_clock::now();
		auto elapsed = now - m_last_frame_time_point;
		m_last_frame_time_point = now;
		m_frame_ms.update(std::chrono::duration<float, std::milli>(elapsed).count());
	}

	glfwPollEvents();
	glfwGetFramebufferSize(m_glfw_window, &m_window_res.x, &m_window_res.y);

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	ImGuizmo::BeginFrame();

	return true;
}

void Testbed::handle_user_input() {
	// Only respond to mouse inputs when not interacting with ImGui
	if (!ImGui::IsAnyItemActive() && !ImGuizmo::IsUsing() && !ImGui::GetIO().WantCaptureMouse) {
		mouse_wheel();
		mouse_drag();
	}

	keyboard_event();

	switch (m_imgui.mode) {
		case ImGuiMode::Enabled: imgui(); break;
		case ImGuiMode::FpsOverlay: overlay_fps(); break;
		case ImGuiMode::Disabled: break;
		default: throw std::runtime_error{fmt::format("Invalid imgui mode: {}", (uint32_t)m_imgui.mode)};
	}
}

vec3 Testbed::vr_to_world(const vec3& pos) const { return mat3(m_camera) * pos * m_scale + m_camera[3]; }

void Testbed::begin_vr_frame_and_handle_vr_input() {
	if (!m_hmd) {
		m_vr_frame_info = nullptr;
		return;
	}

	m_hmd->poll_events();
	if (!m_hmd->must_run_frame_loop()) {
		m_vr_frame_info = nullptr;
		return;
	}

	m_vr_frame_info = m_hmd->begin_frame();

	const auto& views = m_vr_frame_info->views;
	size_t n_views = views.size();
	size_t n_devices = m_devices.size();
	if (n_views > 0) {
		set_n_views(n_views);

		ivec2 total_size = 0;
		for (size_t i = 0; i < n_views; ++i) {
			ivec2 view_resolution = {views[i].view.subImage.imageRect.extent.width, views[i].view.subImage.imageRect.extent.height};
			total_size += view_resolution;

			m_views[i].full_resolution = view_resolution;

			// Apply the VR pose relative to the world camera transform.
			m_views[i].camera0 = mat3(m_camera) * views[i].pose;
			m_views[i].camera0[3] = vr_to_world(views[i].pose[3]);
			m_views[i].camera1 = m_views[i].camera0;

			m_views[i].visualized_dimension = m_visualized_dimension;

			const auto& xr_fov = views[i].view.fov;

			// Compute the distance on the image plane (1 unit away from the camera) that an angle of the respective FOV spans
			vec2 rel_focal_length_left_down = 0.5f *
				fov_to_focal_length(ivec2(1), vec2{360.0f * xr_fov.angleLeft / PI(), 360.0f * xr_fov.angleDown / PI()});
			vec2 rel_focal_length_right_up = 0.5f *
				fov_to_focal_length(ivec2(1), vec2{360.0f * xr_fov.angleRight / PI(), 360.0f * xr_fov.angleUp / PI()});

			// Compute total distance (for X and Y) that is spanned on the image plane.
			m_views[i].relative_focal_length = rel_focal_length_right_up - rel_focal_length_left_down;

			// Compute fraction of that distance that is spanned by the right-up part and set screen center accordingly.
			vec2 ratio = rel_focal_length_right_up / m_views[i].relative_focal_length;
			m_views[i].screen_center = {1.0f - ratio.x, ratio.y};

			// Fix up weirdness in the rendering pipeline
			m_views[i].relative_focal_length[(m_fov_axis + 1) % 2] *= (float)view_resolution[(m_fov_axis + 1) % 2] /
				(float)view_resolution[m_fov_axis];
			m_views[i].render_buffer->set_hidden_area_mask(m_vr_use_hidden_area_mask ? views[i].hidden_area_mask : nullptr);

			// Render each view on a different GPU (if available)
			m_views[i].device = m_use_aux_devices ? &m_devices.at(i % m_devices.size()) : &primary_device();
		}

		// Put all the views next to each other, but at half size
		glfwSetWindowSize(m_glfw_window, total_size.x / 2, (total_size.y / 2) / n_views);

		// VR controller input
		const auto& hands = m_vr_frame_info->hands;
		m_fps_camera = true;

		// TRANSLATE BY STICK (if not pressing the stick)
		if (!hands[0].pressing) {
			vec3 translate_vec = vec3{hands[0].thumbstick.x, 0.0f, hands[0].thumbstick.y} * m_camera_velocity * m_frame_ms.val() / 1000.0f;
			if (translate_vec != vec3(0.0f)) {
				translate_camera(translate_vec, mat3(m_views.front().camera0), false);
			}
		}

		// TURN BY STICK (if not pressing the stick)
		if (!hands[1].pressing) {
			auto prev_camera = m_camera;

			// Turn around the up vector (equivalent to x-axis mouse drag) with right joystick left/right
			float sensitivity = 0.35f;
			auto rot = rotation_from_angles({-2.0f * PI() * sensitivity * hands[1].thumbstick.x * m_frame_ms.val() / 1000.0f, 0.0f}) *
				mat3(m_camera);
			m_camera = mat4x3(rot[0], rot[1], rot[2], m_camera[3]);

			// Translate camera such that center of rotation was about the current view
			m_camera[3] += mat3(prev_camera) * views[0].pose[3] * m_scale - mat3(m_camera) * views[0].pose[3] * m_scale;
		}

		// TRANSLATE, SCALE, AND ROTATE BY GRAB
		{
			bool both_grabbing = hands[0].grabbing && hands[1].grabbing;
			float drag_factor = both_grabbing ? 0.5f : 1.0f;

			if (both_grabbing) {
				drag_factor = 0.5f;

				vec3 prev_diff = hands[0].prev_grab_pos - hands[1].prev_grab_pos;
				vec3 diff = hands[0].grab_pos - hands[1].grab_pos;
				vec3 center = 0.5f * (hands[0].grab_pos + hands[1].grab_pos);

				vec3 center_world = vr_to_world(0.5f * (hands[0].grab_pos + hands[1].grab_pos));

				// Scale around center position of the two dragging hands. Makes the scaling feel similar to phone pinch-to-zoom
				float scale = m_scale * length(prev_diff) / length(diff);
				m_camera[3] = (view_pos() - center_world) * (scale / m_scale) + center_world;
				m_scale = scale;

				// Take rotational component and project it to the nearest rotation about the up vector.
				// We don't want to rotate the scene about any other axis.
				vec3 rot = cross(normalize(prev_diff), normalize(diff));
				float rot_radians = std::asin(dot(m_up_dir, rot));

				auto prev_camera = m_camera;
				auto rotcam = rotmat(rot_radians, m_up_dir) * mat3(m_camera);
				m_camera = mat4x3(rotcam[0], rotcam[1], rotcam[2], m_camera[3]);
				m_camera[3] += mat3(prev_camera) * center * m_scale - mat3(m_camera) * center * m_scale;
			}

			for (const auto& hand : hands) {
				if (hand.grabbing) {
					m_camera[3] -= drag_factor * mat3(m_camera) * hand.drag() * m_scale;
				}
			}
		}
	}
}

void Testbed::SecondWindow::draw(GLuint texture) {
	if (!window) {
		return;
	}
	int display_w, display_h;
	GLFWwindow* old_context = glfwGetCurrentContext();
	glfwMakeContextCurrent(window);
	glfwGetFramebufferSize(window, &display_w, &display_h);
	glViewport(0, 0, display_w, display_h);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindVertexArray(vao);
	if (program) {
		glUseProgram(program);
	}
	glDrawArrays(GL_TRIANGLES, 0, 6);
	glBindVertexArray(0);
	glUseProgram(0);
	glfwSwapBuffers(window);
	glfwMakeContextCurrent(old_context);
}

void Testbed::init_opengl_shaders() {
	static const char* shader_vert = R"glsl(#version 140
		out vec2 UVs;
		void main() {
			UVs = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
			gl_Position = vec4(UVs * 2.0 - 1.0, 0.0, 1.0);
		})glsl";

	static const char* shader_frag = R"glsl(#version 140
		in vec2 UVs;
		out vec4 frag_color;
		uniform sampler2D rgba_texture;
		uniform sampler2D depth_texture;

		struct FoveationWarp {
			float al, bl, cl;
			float am, bm;
			float ar, br, cr;
			float switch_left, switch_right;
			float inv_switch_left, inv_switch_right;
		};

		uniform FoveationWarp warp_x;
		uniform FoveationWarp warp_y;

		float unwarp(in FoveationWarp warp, float y) {
			y = clamp(y, 0.0, 1.0);
			if (y < warp.inv_switch_left) {
				return (sqrt(-4.0 * warp.al * warp.cl + 4.0 * warp.al * y + warp.bl * warp.bl) - warp.bl) / (2.0 * warp.al);
			} else if (y > warp.inv_switch_right) {
				return (sqrt(-4.0 * warp.ar * warp.cr + 4.0 * warp.ar * y + warp.br * warp.br) - warp.br) / (2.0 * warp.ar);
			} else {
				return (y - warp.bm) / warp.am;
			}
		}

		vec2 unwarp(in vec2 pos) {
			return vec2(unwarp(warp_x, pos.x), unwarp(warp_y, pos.y));
		}

		void main() {
			vec2 tex_coords = UVs;
			tex_coords.y = 1.0 - tex_coords.y;
			tex_coords = unwarp(tex_coords);
			frag_color = texture(rgba_texture, tex_coords.xy);
			//Uncomment the following line of code to visualize debug the depth buffer for debugging.
			// frag_color = vec4(vec3(texture(depth_texture, tex_coords.xy).r), 1.0);
			gl_FragDepth = texture(depth_texture, tex_coords.xy).r;
		})glsl";

	GLuint vert = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vert, 1, &shader_vert, NULL);
	glCompileShader(vert);
	check_shader(vert, "Blit vertex shader", false);

	GLuint frag = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(frag, 1, &shader_frag, NULL);
	glCompileShader(frag);
	check_shader(frag, "Blit fragment shader", false);

	m_blit_program = glCreateProgram();
	glAttachShader(m_blit_program, vert);
	glAttachShader(m_blit_program, frag);
	glLinkProgram(m_blit_program);
	check_shader(m_blit_program, "Blit shader program", true);

	glDeleteShader(vert);
	glDeleteShader(frag);

	glGenVertexArrays(1, &m_blit_vao);
}

void Testbed::blit_texture(
	const Foveation& foveation,
	GLint rgba_texture,
	GLint rgba_filter_mode,
	GLint depth_texture,
	GLint framebuffer,
	const ivec2& offset,
	const ivec2& resolution
) {
	if (m_blit_program == 0) {
		return;
	}

	// Blit image to OpenXR swapchain.
	// Note that the OpenXR swapchain is 8bit while the rendering is in a float texture.
	// As some XR runtimes do not support float swapchains, we can't render into it directly.

	bool tex = glIsEnabled(GL_TEXTURE_2D);
	bool depth = glIsEnabled(GL_DEPTH_TEST);
	bool cull = glIsEnabled(GL_CULL_FACE);

	if (!tex) {
		glEnable(GL_TEXTURE_2D);
	}
	if (!depth) {
		glEnable(GL_DEPTH_TEST);
	}
	if (cull) {
		glDisable(GL_CULL_FACE);
	}

	glDepthFunc(GL_ALWAYS);
	glDepthMask(GL_TRUE);

	glBindVertexArray(m_blit_vao);
	glUseProgram(m_blit_program);
	glUniform1i(glGetUniformLocation(m_blit_program, "rgba_texture"), 0);
	glUniform1i(glGetUniformLocation(m_blit_program, "depth_texture"), 1);

	auto bind_warp = [&](const FoveationPiecewiseQuadratic& warp, const std::string& uniform_name) {
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".al").c_str()), warp.al);
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".bl").c_str()), warp.bl);
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".cl").c_str()), warp.cl);

		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".am").c_str()), warp.am);
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".bm").c_str()), warp.bm);

		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".ar").c_str()), warp.ar);
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".br").c_str()), warp.br);
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".cr").c_str()), warp.cr);

		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".switch_left").c_str()), warp.switch_left);
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".switch_right").c_str()), warp.switch_right);

		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".inv_switch_left").c_str()), warp.inv_switch_left);
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".inv_switch_right").c_str()), warp.inv_switch_right);
	};

	bind_warp(foveation.warp_x, "warp_x");
	bind_warp(foveation.warp_y, "warp_y");

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, depth_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, rgba_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, rgba_filter_mode);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, rgba_filter_mode);

	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	glViewport(offset.x, offset.y, resolution.x, resolution.y);

	glDrawArrays(GL_TRIANGLES, 0, 3);

	glBindVertexArray(0);
	glUseProgram(0);

	glDepthFunc(GL_LESS);

	// restore old state
	if (!tex) {
		glDisable(GL_TEXTURE_2D);
	}
	if (!depth) {
		glDisable(GL_DEPTH_TEST);
	}
	if (cull) {
		glEnable(GL_CULL_FACE);
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Testbed::draw_gui() {
	// Make sure all the cuda code finished its business here
	CUDA_CHECK_THROW(cudaDeviceSynchronize());

	if (!m_rgba_render_textures.empty()) {
		m_second_window.draw((GLuint)m_rgba_render_textures.front()->texture());
	}

	glfwMakeContextCurrent(m_glfw_window);
	int display_w, display_h;
	glfwGetFramebufferSize(m_glfw_window, &display_w, &display_h);
	glViewport(0, 0, display_w, display_h);
	glClearColor(0.f, 0.f, 0.f, 0.f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_BLEND);
	glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
	glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

	ivec2 extent = {(int)((float)display_w / m_n_views.x), (int)((float)display_h / m_n_views.y)};

	int i = 0;
	for (int y = 0; y < m_n_views.y; ++y) {
		for (int x = 0; x < m_n_views.x; ++x) {
			if (i >= m_views.size()) {
				break;
			}

			auto& view = m_views[i];
			ivec2 top_left{x * extent.x, display_h - (y + 1) * extent.y};
			blit_texture(
				m_foveated_rendering_visualize ? Foveation{} : view.foveation,
				m_rgba_render_textures.at(i)->texture(),
				m_foveated_rendering ? GL_LINEAR : GL_NEAREST,
				m_depth_render_textures.at(i)->texture(),
				0,
				top_left,
				extent
			);

			++i;
		}
	}
	glFinish();
	glViewport(0, 0, display_w, display_h);

	ImDrawList* list = ImGui::GetBackgroundDrawList();
	list->AddCallback(ImDrawCallback_ResetRenderState, nullptr);

	// Visualizations are only meaningful when rendering a single view
	if (m_views.size() == 1) {
		draw_visualizations(list, m_smoothed_camera);
	}

	if (m_render_ground_truth) {
		list->AddText(ImVec2(4.f, 4.f), 0xffffffff, "Ground Truth");
	}

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	glfwSwapBuffers(m_glfw_window);

	// Make sure all the OGL code finished its business here.
	// Any code outside of this function needs to be able to freely write to
	// textures without being worried about interfering with rendering.
	glFinish();
}
#endif // NGP_GUI

__global__ void to_8bit_color_kernel(ivec2 resolution, EColorSpace output_color_space, cudaSurfaceObject_t surface, uint8_t* result) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x || y >= resolution.y) {
		return;
	}

	vec4 color;
	surf2Dread((float4*)&color, surface, x * sizeof(float4), y);

	if (output_color_space == EColorSpace::Linear) {
		color.rgb() = linear_to_srgb(color.rgb());
	}

	for (uint32_t i = 0; i < 3; ++i) {
		result[(x + resolution.x * y) * 3 + i] = (uint8_t)(clamp(color[i], 0.0f, 1.0f) * 255.0f + 0.5f);
	}
}

void Testbed::prepare_next_camera_path_frame() {
	if (!m_camera_path.rendering) {
		return;
	}

	// If we're rendering a video, we'd like to accumulate multiple spp
	// for motion blur. Hence dump the frame once the target spp has been reached
	// and only reset _then_.
	if (m_views.front().render_buffer->spp() == m_camera_path.render_settings.spp) {
		auto tmp_dir = fs::path{"tmp"};
		if (!tmp_dir.exists()) {
			if (!fs::create_directory(tmp_dir)) {
				m_camera_path.rendering = false;
				tlog::error() << "Failed to create temporary directory 'tmp' to hold rendered images.";
				return;
			}
		}

		ivec2 res = m_views.front().render_buffer->out_resolution();
		const dim3 threads = {16, 8, 1};
		const dim3 blocks = {div_round_up((uint32_t)res.x, threads.x), div_round_up((uint32_t)res.y, threads.y), 1};

		GPUMemory<uint8_t> image_data(product(res) * 3);
		to_8bit_color_kernel<<<blocks, threads>>>(
			res,
			EColorSpace::SRGB, // the GUI always renders in SRGB
			m_views.front().render_buffer->surface(),
			image_data.data()
		);

		m_render_futures.emplace_back(
			m_thread_pool.enqueue_task([image_data = std::move(image_data), frame_idx = m_camera_path.render_frame_idx++, res, tmp_dir] {
				std::vector<uint8_t> cpu_image_data(image_data.size());
				CUDA_CHECK_THROW(cudaMemcpy(cpu_image_data.data(), image_data.data(), image_data.bytes(), cudaMemcpyDeviceToHost));
				write_stbi(tmp_dir / fmt::format("{:06d}.jpg", frame_idx), res.x, res.y, 3, cpu_image_data.data(), 100);
			})
		);

		reset_accumulation(true);

		if (m_camera_path.render_frame_idx == m_camera_path.render_settings.n_frames(m_camera_path.duration_seconds())) {
			m_camera_path.rendering = false;

			wait_all(m_render_futures);
			m_render_futures.clear();

			tlog::success() << "Finished rendering '.jpg' video frames to '" << tmp_dir << "'. Assembling them into a video next.";

			fs::path ffmpeg = "ffmpeg";

#ifdef _WIN32
			// Under Windows, try automatically downloading FFmpeg binaries if they don't exist
			if (system(fmt::format("where {} >nul 2>nul", ffmpeg.str()).c_str()) != 0) {
				fs::path dir = root_dir();
				if ((dir / "external" / "ffmpeg").exists()) {
					for (const auto& path : fs::directory{dir / "external" / "ffmpeg"}) {
						ffmpeg = path / "bin" / "ffmpeg.exe";
					}
				}

				if (!ffmpeg.exists()) {
					tlog::info() << "FFmpeg not found. Downloading FFmpeg...";
					do_system((dir / "scripts" / "download_ffmpeg.bat").str());
				}

				for (const auto& path : fs::directory{dir / "external" / "ffmpeg"}) {
					ffmpeg = path / "bin" / "ffmpeg.exe";
				}

				if (!ffmpeg.exists()) {
					tlog::warning() << "FFmpeg download failed. Trying system-wide FFmpeg.";
				}
			}
#endif

			auto ffmpeg_command = fmt::format(
				"{} -loglevel error -y -framerate {} -i tmp/%06d.jpg -c:v libx264 -preset slow -crf {} -pix_fmt yuv420p \"{}\"",
				ffmpeg.str(),
				m_camera_path.render_settings.fps,
				// Quality goes from 0 to 10. This conversion to CRF means a quality of 10
				// is a CRF of 17 and a quality of 0 a CRF of 27, which covers the "sane"
				// range of x264 quality settings according to the FFmpeg docs:
				// https://trac.ffmpeg.org/wiki/Encode/H.264
				27 - m_camera_path.render_settings.quality,
				m_camera_path.render_settings.filename
			);
			int ffmpeg_result = do_system(ffmpeg_command);
			if (ffmpeg_result == 0) {
				tlog::success() << "Saved video '" << m_camera_path.render_settings.filename << "'";
			} else if (ffmpeg_result == -1) {
				tlog::error() << "Video could not be assembled: FFmpeg not found.";
			} else {
				tlog::error() << "Video could not be assembled: FFmpeg failed";
			}

			clear_tmp_dir();
		}
	}

	const auto& rs = m_camera_path.render_settings;
	const float duration = m_camera_path.duration_seconds();
	m_camera_path.play_time = (float)((double)m_camera_path.render_frame_idx / (double)rs.n_frames(duration));

	if (m_views.front().render_buffer->spp() == 0) {
		set_camera_from_time(m_camera_path.play_time);
		apply_camera_smoothing(rs.frame_milliseconds(duration));

		auto smoothed_camera_backup = m_smoothed_camera;

		// Compute the camera for the next frame in order to be able to compute motion blur
		// between it and the current one.
		set_camera_from_time(m_camera_path.play_time + 1.0f / rs.n_frames(duration));
		apply_camera_smoothing(rs.frame_milliseconds(duration));

		m_camera_path.render_frame_end_camera = m_smoothed_camera;

		// Revert camera such that the next frame will be computed correctly
		// (Start camera of next frame should be the same as end camera of this frame)
		set_camera_from_time(m_camera_path.play_time);
		m_smoothed_camera = smoothed_camera_backup;
	}
}

__global__ void reproject_kernel(
	BoundingBox render_aabb,
	mat3 render_aabb_to_local,
	default_rng_t rng,
	float near_t,
	float step_factor,
	uint32_t spp,
	uint32_t view_idx,
	mat4x3 src_camera,
	vec2 src_screen_center,
	vec2 src_focal_length,
	ivec2 src_resolution,
	Foveation src_foveation,
	Lens src_lens,
	MatrixView<const float> src_depth_buffer,
	mat4x3 dst_camera,
	vec2 dst_screen_center,
	vec2 dst_focal_length,
	ivec2 dst_resolution,
	Foveation dst_foveation,
	Lens dst_lens,
	vec4* __restrict__ dst_frame_buffer,
	MatrixView<float> dst_depth_buffer,
	MatrixView<uint8_t> dst_hole_mask,
	MatrixView<ViewIdx> dst_index_field,
	MatrixView<uint8_t> src_hole_mask = {},
	MatrixView<ViewIdx> src_index_field = {}
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	uint32_t is_hole = dst_hole_mask(y, x);
	if (x >= dst_resolution.x || y >= dst_resolution.y || (src_hole_mask && !is_hole)) {
		return;
	}

	auto ray = pixel_to_ray(
		spp,
		{(int)x, (int)y},
		dst_resolution,
		dst_focal_length,
		dst_camera,
		dst_screen_center,
		vec3(0.0f), // parallax
		false,      // pixel center snap
		0.0f,       // near dist
		1.0f,       // focus
		0.0f,       // aperture
		dst_foveation,
		{},
		dst_lens
	);

	uint32_t dst_idx = x + dst_resolution.x * y;

	float t = near_t;
	rng.advance(dst_idx);
	t *= std::pow(step_factor, rng.next_float());

	struct Result {
		ViewIdx idx;
		float dist;
		float t;
	};

	auto get_reprojected_dist = [&](float t) -> Result {
		vec3 p = ray(t);

		vec2 src_px = pos_to_pixel(p, src_resolution, src_focal_length, src_camera, src_screen_center, vec3(0.0f), src_foveation, src_lens);

		if (src_px.x <= 0 || src_px.x >= src_resolution.x || src_px.y <= 0 || src_px.y >= src_resolution.y) {
			return {
				{-1, 0},
                -1.0f, -1.0f
			};
		}

		ViewIdx nearest = {clamp(ivec2(floor(src_px)), 0, src_resolution - 1), view_idx};
		if (src_hole_mask) {
			if (!src_hole_mask(nearest.px.y, nearest.px.x) || src_depth_buffer(nearest.px.y, nearest.px.x) == 0.0f) {
				return {
					{-1, 0},
                    -1.0f, -1.0f
				};
			}
		}

		float d = src_depth_buffer(nearest.px.y, nearest.px.x);
		Ray src_ray = {
			src_camera[3],
			p - src_camera[3],
		};

		src_ray.d /= src_lens.is_360() ? length(src_ray.d) : dot(src_ray.d, src_camera[2]);

		vec3 src_p = src_ray(d);
		if (src_index_field) {
			nearest = src_index_field(nearest.px.y, nearest.px.x);
		}

		return {nearest, distance(p, src_p), t};
	};

	auto refine_match = [&](Result match) -> Result {
		static const uint32_t N_STEPS_PER_REFINEMENT = 10;
		static const uint32_t N_REFINEMENTS = 3;

		float prev_t = match.t / step_factor;
		float next_t = match.t * step_factor;

		NGP_PRAGMA_UNROLL
		for (uint32_t j = 0; j < N_REFINEMENTS; ++j) {
			float step_size = (next_t - prev_t) / (N_STEPS_PER_REFINEMENT - 1);
			float t = prev_t;

			NGP_PRAGMA_UNROLL
			for (uint32_t i = 0; i < N_STEPS_PER_REFINEMENT; ++i) {
				auto res = get_reprojected_dist(t);
				if (res.idx.px.x >= 0 && res.dist < match.dist) {
					match = res;
					prev_t = t - step_size;
					next_t = t + step_size;
				}

				t += step_size;
			}
		}

		return match;
	};

	Result final = {
		{-1, 0},
        std::numeric_limits<float>::infinity(), 0
	};
	Result fallback = final;

	float mint = fmaxf(render_aabb.ray_intersect(render_aabb_to_local * ray.o, render_aabb_to_local * ray.d).x, 0.0f) + 1e-6f;
	if (mint < MAX_DEPTH()) {
		while (t <= mint) {
			t *= step_factor;
		}
	}

	// float last_dist = std::numeric_limits<float>::infinity();
	for (; render_aabb.contains(render_aabb_to_local * ray(t)); t *= step_factor) {
		auto res = get_reprojected_dist(t);
		if (res.idx.px.x >= 0) {
			if (res.dist < t * (step_factor - 1.0f)) {
				res = refine_match(res);
				if (res.dist < final.dist) {
					if (res.dist / res.t < 4.0f / dst_focal_length.x) {
						final = res;
						break;
					}
				}
			}

			// if (res.dist < last_dist) {
			//	fallback = res;
			// }

			// last_dist = res.dist;
		}
	}

	if (final.idx.px.x == -1) {
		final = fallback;
	}

	float prev_depth = dst_depth_buffer(y, x);

	dst_frame_buffer[dst_idx] = vec4::zero();
	if (final.idx.px.x == -1) {
		if (is_hole) {
			dst_depth_buffer(y, x) = MAX_DEPTH();
			dst_hole_mask(y, x) = 1;
			dst_index_field(y, x) = {-1, 0};
		}
	} else {
		if (is_hole || final.t * step_factor < prev_depth) {
			dst_depth_buffer(y, x) = final.t;
			dst_hole_mask(y, x) = src_index_field ? 2 : 0;
			dst_index_field(y, x) = final.idx;
		}
	}
}

__global__ void dilate_holes_kernel(ivec2 res, MatrixView<const uint8_t> old_hole_mask, MatrixView<uint8_t> hole_mask) {
	int32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	int32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= res.x || y >= res.y) {
		return;
	}

	auto is_hole = [&](const ivec2& offset) {
		auto clamped = clamp(ivec2{x, y} + offset, 0, res - 1);
		return old_hole_mask(clamped.y, clamped.x);
	};

	hole_mask(y, x) = is_hole({1, 0}) || is_hole({-1, 0}) || is_hole({1, 1}) || is_hole({-1, 1}) || is_hole({1, -1}) || is_hole({-1, -1}) ||
		is_hole({0, 1}) || is_hole({0, -1});
}

__global__ void generate_alt_depth_kernel(
	mat4x3 src_camera,
	vec2 src_screen_center,
	vec2 src_focal_length,
	ivec2 src_resolution,
	const vec4* __restrict__ src_frame_buffer,
	const float* __restrict__ src_depth_buffer,
	Foveation src_foveation,
	Lens src_lens,
	mat4x3 dst_camera,
	Lens dst_lens,
	MatrixView<float> alt_depth_buffer
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= src_resolution.x || y >= src_resolution.y) {
		return;
	}

	auto ray = pixel_to_ray(
		0,
		{(int)x, (int)y},
		src_resolution,
		src_focal_length,
		src_camera,
		src_screen_center,
		vec3(0.0f), // parallax
		false,      // pixel center snap
		0.0f,       // near dist
		1.0f,       // focus
		0.0f,       // aperture
		src_foveation,
		{},
		src_lens
	);

	uint32_t src_idx = x + src_resolution.x * y;
	vec3 p = ray(src_depth_buffer[src_idx]);

	alt_depth_buffer(y, x) = dst_lens.is_360() ? distance(p, dst_camera[3]) : dot(p - dst_camera[3], dst_camera[2]);
}

__global__ void copy_depth_buffer_kernel(ivec2 dst_resolution, const float* __restrict__ src_depth_buffer, MatrixView<float> dst_depth_buffer) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= dst_resolution.x || y >= dst_resolution.y) {
		return;
	}

	uint32_t idx = x + dst_resolution.x * y;
	dst_depth_buffer(y, x) = src_depth_buffer[idx];
}

static constexpr float Z_NEAR = 0.1f;
static constexpr float Z_BASE = 1.03f;

inline NGP_HOST_DEVICE float to_log_depth(float d) { return logf(d / Z_NEAR) * logf(Z_BASE); }

inline NGP_HOST_DEVICE float from_log_depth(float d) { return expf(d / logf(Z_BASE)) * Z_NEAR; }

inline NGP_HOST_DEVICE vec4 from_rgbd32(uint32_t val) {
	vec4 result = rgba32_to_rgba(val);
	result.a = from_log_depth(result.a);
	return result;
}

inline NGP_HOST_DEVICE uint32_t to_rgbd32(vec4 rgbd) {
	rgbd.a = to_log_depth(rgbd.a);
	return rgba_to_rgba32(rgbd);
}

__global__ void reproject_viz_kernel(
	ivec2 dst_res,
	const ivec2* src_res,
	bool pm_enable,
	MatrixView<const uint32_t> hole_labels,
	MatrixView<const EPmPixelState> state,
	MatrixView<const ViewIdx> index_field,
	MatrixView<const uint32_t> dst_rgbd,
	MatrixView<const float> dst_depth,
	const MatrixView<const uint32_t>* src_rgba,
	const MatrixView<const float>* src_depth,
	MatrixView<vec4> frame,
	MatrixView<float> depth,
	EPmVizMode viz_mode,
	float depth_scale
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= dst_res.x || y >= dst_res.y) {
		return;
	}

	if (!pm_enable && state(y, x) == EPmPixelState::Hole) {
		if (viz_mode == EPmVizMode::Depth) {
			frame(y, x).rgb() = vec3(depth(y, x) * depth_scale);
		} else {
			frame(y, x).rgb() = vec3(0.0f);
		}

		depth(y, x) = MAX_DEPTH();
		return;
	}

	auto src_idx = index_field(y, x);

	if (viz_mode == EPmVizMode::Depth) {
		frame(y, x).rgb() = vec3(dst_depth(y, x) * depth_scale);
	} else if (viz_mode == EPmVizMode::Offset) {
		vec2 diff = vec2(x, y) / vec2(dst_res) - vec2(src_idx.px) / vec2(src_res[src_idx.view]);
		float l = length(diff);
		frame(y, x).rgb() = hsv_to_rgb({atan2(diff.y / l, diff.x / l) / (PI() * 2.0f) + 0.5f, 1.0f, l});
	} else if (viz_mode == EPmVizMode::Holes) {
		if (state(y, x) == EPmPixelState::Hole) {
			frame(y, x).rgb() = colormap_turbo(hole_labels(y, x) / (float)product(dst_res));
		}
	} else {
		vec4 rgbd = rgba32_to_rgba(src_rgba[src_idx.view](src_idx.px.y, src_idx.px.x));
		rgbd.rgb() = srgb_to_linear(rgbd.rgb());
		frame(y, x) = rgbd;
		depth(y, x) = src_depth[src_idx.view](src_idx.px.y, src_idx.px.x);
	}
}

static constexpr int32_t PM_PATCH_RADIUS = 4;

inline NGP_HOST_DEVICE ivec2 mirror(const ivec2& v, const ivec2& res) { return abs(res - abs(res - v - 1) - 1); }

__global__ void pm_prepare_padded_src_buffers(
	ivec2 padded_res,
	ivec2 res,
	MatrixView<const vec4> src_rgba,
	MatrixView<const float> src_depth,
	MatrixView<uint32_t> dst_rgbd,
	MatrixView<float> dst_depth
) {
	int32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	int32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= padded_res.x || y >= padded_res.y) {
		return;
	}

	ivec2 padding = (padded_res - res) / 2;
	ivec2 idx = {(int16_t)(x - padding.x), (int16_t)(y - padding.y)};

	// auto clamped_idx = clamp(idx, i16vec2((int16_t)0), i16vec2(res - 1));
	auto clamped_idx = mirror(idx, i16vec2(res));

	vec4 rgba = src_rgba(clamped_idx.y, clamped_idx.x);
	rgba.rgb() = linear_to_srgb(rgba.rgb());
	dst_rgbd(idx.y, idx.x) = rgba_to_rgba32(rgba);
	dst_depth(idx.y, idx.x) = src_depth(clamped_idx.y, clamped_idx.x);
}

__global__ void pm_prepare_padded_dst_buffers(
	ivec2 padded_dst_res,
	ivec2 dst_res,
	uint32_t n_src_views,
	const ivec2* src_res,
	default_rng_t fixed_seed_rng,
	const MatrixView<const uint32_t>* src_rgbd,
	const MatrixView<const float>* src_depth,
	MatrixView<EPmPixelState> dst_state,
	MatrixView<ViewIdx> dst_index_field,
	MatrixView<uint32_t> dst_rgbd,
	MatrixView<float> dst_depth,
	MatrixView<float> dst_depth_threshold,
	MatrixView<const uint8_t> hole_mask
) {
	int32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	int32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= padded_dst_res.x || y >= padded_dst_res.y) {
		return;
	}

	ivec2 padding = (padded_dst_res - dst_res) / 2;
	ivec2 idx = {x - padding.x, y - padding.y};

	// auto clamped_idx = clamp(idx, i16vec2((int16_t)0), i16vec2(res - 1));
	auto clamped_idx = mirror(idx, dst_res);

	ViewIdx src_idx;
	uint8_t is_hole = hole_mask(clamped_idx.y, clamped_idx.x);
	if (is_hole == 1) {
		fixed_seed_rng.advance((x + y * padded_dst_res.x) * 3);

		// uint32_t random_view = fixed_seed_rng.next_uint(n_src_views);
		uint32_t random_view = 0;
		auto res = src_res[random_view];
		src_idx = {
			i16vec2{(int16_t)fixed_seed_rng.next_uint(res.y), (int16_t)fixed_seed_rng.next_uint(res.x)},
            random_view
		};
	} else {
		src_idx = dst_index_field(clamped_idx.y, clamped_idx.x);
	}

	dst_index_field(idx.y, idx.x) = src_idx;

	if (is_hole == 0) {
		dst_state(idx.y, idx.x) = EPmPixelState::Reprojected;
		dst_rgbd(idx.y, idx.x) = src_rgbd[src_idx.view](src_idx.px.y, src_idx.px.x);

		float depth = src_depth[src_idx.view](src_idx.px.y, src_idx.px.x);
		dst_depth(idx.y, idx.x) = depth;
		dst_depth_threshold(idx.y, idx.x) = depth;
	} else if (is_hole == 1) {
		dst_state(idx.y, idx.x) = EPmPixelState::Hole;
		dst_rgbd(idx.y, idx.x) = 0x00FF00FF;
		dst_depth(idx.y, idx.x) = 0.0f;
		dst_depth_threshold(idx.y, idx.x) = 0.0f;
	} else {
		dst_state(idx.y, idx.x) = EPmPixelState::Reprojected;
		dst_rgbd(idx.y, idx.x) = src_rgbd[src_idx.view](src_idx.px.y, src_idx.px.x);
		dst_depth_threshold(idx.y, idx.x) = dst_depth(idx.y, idx.x);
	}
}


void Testbed::reproject_views(const std::vector<const View*> src_views, View& dst_view) {
	if (src_views.empty()) {
		dst_view.render_buffer->clear_frame(m_stream.get());
		return;
	}

	auto dst_res = dst_view.render_buffer->in_resolution();

	std::vector<ivec2> src_res(src_views.size());
	std::vector<vec2> src_screen_center(src_views.size());
	std::vector<vec2> src_focal_length(src_views.size());
	std::vector<GPUImage<float>> tmp_src_depth_buffer(src_views.size());

	for (size_t i = 0; i < src_views.size(); ++i) {
		src_res[i] = src_views[i]->render_buffer->in_resolution();

		src_screen_center[i] = render_screen_center(src_views[i]->screen_center);
		src_focal_length[i] =
			calc_focal_length(src_views[i]->render_buffer->in_resolution(), src_views[i]->relative_focal_length, m_fov_axis, m_zoom);

		// Compute the depth of every pixel in the src_view when reprojected into the dst_view.
		// This could in principle happen in parallel with the reprojection step happening below.
		tmp_src_depth_buffer[i] = GPUImage<float>(src_res[i], m_stream.get());

		const dim3 threads = {16, 8, 1};
		const dim3 blocks = {div_round_up((uint32_t)dst_res.x, threads.x), div_round_up((uint32_t)dst_res.y, threads.y), 1};

		generate_alt_depth_kernel<<<blocks, threads, 0, m_stream.get()>>>(
			src_views[i]->camera0,
			src_screen_center[i],
			src_focal_length[i],
			src_res[i],
			src_views[i]->render_buffer->frame_buffer(),
			src_views[i]->render_buffer->depth_buffer(),
			src_views[i]->foveation,
			src_views[i]->lens,
			dst_view.camera0,
			dst_view.lens,
			tmp_src_depth_buffer[i].view()
		);
	}

	dst_view.render_buffer->clear_frame(m_stream.get());

	const dim3 threads = {16, 8, 1};
	const dim3 blocks = {div_round_up((uint32_t)dst_res.x, threads.x), div_round_up((uint32_t)dst_res.y, threads.y), 1};

	auto prev_index_field = std::move(dst_view.index_field);
	dst_view.index_field = GPUImage<ViewIdx>(dst_res, PM_PATCH_RADIUS, m_stream.get());

	auto prev_hole_mask = std::move(dst_view.hole_mask);
	dst_view.hole_mask = GPUImage<uint8_t>(dst_res, m_stream.get());
	dst_view.hole_mask.image.memset_async(m_stream.get(), 1);

	auto prev_depth_buffer = std::move(dst_view.depth_buffer);
	dst_view.depth_buffer = GPUImage<float>(dst_res, PM_PATCH_RADIUS, m_stream.get());

	auto dst_screen_center = render_screen_center(dst_view.screen_center);
	auto dst_focal_length = calc_focal_length(dst_res, dst_view.relative_focal_length, m_fov_axis, m_zoom);

	// First reproject from the source images as much as possible
	for (size_t i = 0; i < src_views.size(); ++i) {
		reproject_kernel<<<blocks, threads, 0, m_stream.get()>>>(
			m_render_aabb,
			m_render_aabb_to_local,
			m_rng,
			m_reproject_min_t,
			m_reproject_step_factor,
			dst_view.render_buffer->spp(),
			i,
			src_views[i]->camera0,
			src_screen_center[i],
			src_focal_length[i],
			src_res[i],
			src_views[i]->foveation,
			src_views[i]->lens,
			MatrixView<const float>(src_views[i]->render_buffer->depth_buffer(), src_res[i].x, 1),
			dst_view.camera0,
			dst_screen_center,
			dst_focal_length,
			dst_res,
			dst_view.foveation,
			dst_view.lens,
			dst_view.render_buffer->frame_buffer(),
			dst_view.depth_buffer.view(),
			dst_view.hole_mask.view(),
			dst_view.index_field.view()
		);
	}

	// auto old_holes_mask = std::move(dst_view.hole_mask);
	// dst_view.hole_mask = GPUImage<uint8_t>(dst_res, m_stream.get());
	// dilate_holes_kernel<<<blocks, threads, 0, m_stream.get()>>>(dst_res, old_holes_mask.view(), dst_view.hole_mask.view());

	// Then try reprojecting into the remaining holes from the previous rendering
	if (m_reproject_reuse_last_frame && prev_depth_buffer.data()) {
		reproject_kernel<<<blocks, threads, 0, m_stream.get()>>>(
			m_render_aabb,
			m_render_aabb_to_local,
			m_rng,
			m_reproject_min_t,
			m_reproject_step_factor,
			dst_view.render_buffer->spp(),
			0, // Reprojecting from the most recent view will copy the previous index anyway.
			dst_view.prev_camera,
			render_screen_center(dst_view.screen_center),
			calc_focal_length(prev_hole_mask.resolution(), dst_view.relative_focal_length, m_fov_axis, m_zoom),
			prev_hole_mask.resolution(),
			dst_view.prev_foveation,
			dst_view.lens,
			prev_depth_buffer.view(),
			dst_view.camera0,
			dst_screen_center,
			dst_focal_length,
			dst_res,
			dst_view.foveation,
			dst_view.lens,
			dst_view.render_buffer->frame_buffer(),
			dst_view.depth_buffer.view(),
			dst_view.hole_mask.view(),
			dst_view.index_field.view(),
			prev_hole_mask.view(),
			prev_index_field.view()
		);
	}

	m_rng.advance();

	auto hole_labels = GPUImage<uint32_t>(dst_res, m_stream.get());

	// Detect holes and label them
	{
		init_labels<<<blocks, threads, 0, m_stream.get()>>>(
			dst_res.x, dst_res.y, hole_labels.n_elements(), hole_labels.data(), dst_view.hole_mask.data()
		);
		resolve_labels<<<blocks, threads, 0, m_stream.get()>>>(dst_res.x, dst_res.y, hole_labels.n_elements(), hole_labels.data());
		label_reduction<<<blocks, threads, 0, m_stream.get()>>>(
			dst_res.x, dst_res.y, hole_labels.n_elements(), hole_labels.data(), dst_view.hole_mask.data()
		);
		resolve_labels<<<blocks, threads, 0, m_stream.get()>>>(dst_res.x, dst_res.y, hole_labels.n_elements(), hole_labels.data());
	}

	auto dst_state_buffer = GPUImage<EPmPixelState>(dst_res, PM_PATCH_RADIUS, m_stream.get());

	std::vector<GPUImage<uint32_t>> src_rgbd_buffer(src_views.size());
	std::vector<GPUImage<float>> src_depth_buffer(src_views.size());
	std::vector<ivec2> padded_src_res(src_views.size());

	std::vector<MatrixView<const uint32_t>> src_rgbd_views(src_views.size());
	std::vector<MatrixView<const float>> src_depth_views(src_views.size());

	for (size_t i = 0; i < src_views.size(); ++i) {
		src_rgbd_buffer[i] = GPUImage<uint32_t>(src_res[i], PM_PATCH_RADIUS, m_stream.get());
		src_depth_buffer[i] = GPUImage<float>(src_res[i], PM_PATCH_RADIUS, m_stream.get());
		padded_src_res[i] = src_rgbd_buffer[i].resolution_padded();

		const dim3 threads = {16, 8, 1};
		const dim3 blocks = {div_round_up((uint32_t)padded_src_res[i].x, threads.x), div_round_up((uint32_t)padded_src_res[i].y, threads.y), 1};

		pm_prepare_padded_src_buffers<<<blocks, threads, 0, m_stream.get()>>>(
			padded_src_res[i],
			src_res[i],
			MatrixView<const vec4>(src_views[i]->render_buffer->frame_buffer(), src_res[i].x, 1),
			tmp_src_depth_buffer[i].view(),
			src_rgbd_buffer[i].view(),
			src_depth_buffer[i].view()
		);

		src_rgbd_views[i] = src_rgbd_buffer[i].view();
		src_depth_views[i] = src_depth_buffer[i].view();
	}

	GPUMemoryArena::Allocation views_alloc;
	auto views_scratch = allocate_workspace_and_distribute<MatrixView<const uint32_t>, MatrixView<const float>, ivec2>(
		m_stream.get(), &views_alloc, src_views.size(), src_views.size(), src_views.size()
	);

	auto* src_rgba_views_device = std::get<0>(views_scratch);
	auto* src_depth_views_device = std::get<1>(views_scratch);
	auto* src_res_device = std::get<2>(views_scratch);

	CUDA_CHECK_THROW(cudaMemcpyAsync(
		src_rgba_views_device,
		src_rgbd_views.data(),
		src_views.size() * sizeof(MatrixView<const uint32_t>),
		cudaMemcpyHostToDevice,
		m_stream.get()
	));
	CUDA_CHECK_THROW(cudaMemcpyAsync(
		src_depth_views_device, src_depth_views.data(), src_views.size() * sizeof(MatrixView<const float>), cudaMemcpyHostToDevice, m_stream.get()
	));
	CUDA_CHECK_THROW(cudaMemcpyAsync(src_res_device, src_res.data(), src_views.size() * sizeof(ivec2), cudaMemcpyHostToDevice, m_stream.get())
	);

	auto dst_rgba_buffer = GPUImage<uint32_t>(dst_res, PM_PATCH_RADIUS, m_stream.get());
	auto dst_depth_threshold_buffer = GPUImage<float>(dst_res, PM_PATCH_RADIUS, m_stream.get());
	ivec2 padded_dst_res = dst_rgba_buffer.resolution_padded();

	default_rng_t fixed_seed_rng{0x1337};

	{
		const dim3 threads = {16, 8, 1};
		const dim3 blocks = {div_round_up((uint32_t)padded_dst_res.x, threads.x), div_round_up((uint32_t)padded_dst_res.y, threads.y), 1};

		pm_prepare_padded_dst_buffers<<<blocks, threads, 0, m_stream.get()>>>(
			padded_dst_res,
			dst_res,
			(uint32_t)src_views.size(),
			src_res_device,
			fixed_seed_rng,
			src_rgba_views_device,
			src_depth_views_device,
			dst_state_buffer.view(),
			dst_view.index_field.view(),
			dst_rgba_buffer.view(),
			dst_view.depth_buffer.view(),
			dst_depth_threshold_buffer.view(),
			dst_view.hole_mask.view()
		);

		fixed_seed_rng.advance();
	}


	reproject_viz_kernel<<<blocks, threads, 0, m_stream.get()>>>(
		dst_res,
		src_res_device,
		m_pm_enable,
		hole_labels.view(),
		dst_state_buffer.view(),
		dst_view.index_field.view(),
		dst_rgba_buffer.view(),
		dst_view.depth_buffer.view(),
		src_rgba_views_device,
		src_depth_views_device,
		MatrixView<vec4>(dst_view.render_buffer->frame_buffer(), dst_res.x, 1),
		MatrixView<float>(dst_view.render_buffer->depth_buffer(), dst_res.x, 1),
		m_pm_viz_mode,
		1.0f
	);
}

void Testbed::render(bool skip_rendering) {
	// Don't do any smoothing here if a camera path is being rendered. It'll take care
	// of the smoothing on its own.
	float frame_ms = m_camera_path.rendering ? 0.0f : m_frame_ms.val();
	apply_camera_smoothing(frame_ms);

	if (!m_render_window || !m_render || skip_rendering) {
		return;
	}

	auto start = std::chrono::steady_clock::now();
	ScopeGuard timing_guard{[&]() {
		m_render_ms.update(std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count());
	}};

	if (frobenius_norm(m_smoothed_camera - m_camera) < 0.001f) {
		m_smoothed_camera = m_camera;
	} else if (!m_camera_path.rendering) {
		reset_accumulation(true);
	}

	if (m_autofocus) {
		autofocus();
	}

	Lens lens = m_render_with_lens_distortion ? m_render_lens : Lens{};

#ifdef NGP_GUI
	if (m_hmd && m_hmd->is_visible()) {
		for (auto& view : m_views) {
			view.visualized_dimension = m_visualized_dimension;
		}

		m_n_views = {(int)m_views.size(), 1};

		m_render_with_lens_distortion = false;
		reset_accumulation(true);
	} else {
		set_n_views(1);
		m_n_views = {1, 1};

		auto& view = m_views.front();

		view.full_resolution = m_window_res;

		view.camera0 = m_smoothed_camera;

		// Motion blur over the fraction of time that the shutter is open. Interpolate in log-space to preserve rotations.
		view.camera1 = (m_camera_path.rendering && !m_gen3c_render_with_gen3c) ?
			camera_log_lerp(m_smoothed_camera, m_camera_path.render_frame_end_camera, m_camera_path.render_settings.shutter_fraction) :
			view.camera0;

		view.visualized_dimension = m_visualized_dimension;
		view.relative_focal_length = m_relative_focal_length;
		view.screen_center = m_screen_center;
		view.render_buffer->set_hidden_area_mask(nullptr);
		view.foveation = {};
		view.lens = lens;
		view.device = &primary_device();
	}

	if (m_dlss) {
		m_aperture_size = 0.0f;
		if (!m_render_lens.supports_dlss()) {
			m_render_with_lens_distortion = false;
		}
	}

	// Update dynamic res and DLSS
	{
		// Don't count the time being spent allocating buffers and resetting DLSS as part of the frame time.
		// Otherwise the dynamic resolution calculations for following frames will be thrown out of whack
		// and may even start oscillating.
		auto skip_start = std::chrono::steady_clock::now();
		ScopeGuard skip_timing_guard{[&]() { start += std::chrono::steady_clock::now() - skip_start; }};

		size_t n_pixels = 0, n_pixels_full_res = 0;
		for (const auto& view : m_views) {
			n_pixels += product(view.render_buffer->in_resolution());
			n_pixels_full_res += product(view.full_resolution);
		}

		float pixel_ratio = n_pixels == 0 ? (1.0f / 256.0f) : ((float)n_pixels / (float)n_pixels_full_res);

		float last_factor = std::sqrt(pixel_ratio);
		float factor = std::sqrt(pixel_ratio / m_render_ms.val() * 1000.0f / m_dynamic_res_target_fps);
		if (!m_dynamic_res) {
			factor = 8.f / (float)m_fixed_res_factor;
		}

		factor = clamp(factor, 1.0f / 16.0f, 1.0f);

		vec2 avg_screen_center = vec2(0.0f);
		for (size_t i = 0; i < m_views.size(); ++i) {
			avg_screen_center += m_views[i].screen_center;
		}

		avg_screen_center /= (float)m_views.size();

		for (auto&& view : m_views) {
			if (m_dlss) {
				view.render_buffer->enable_dlss(*m_dlss_provider, view.full_resolution);
			} else {
				view.render_buffer->disable_dlss();
			}

			ivec2 render_res = view.render_buffer->in_resolution();
			ivec2 new_render_res = clamp(ivec2(vec2(view.full_resolution) * factor), view.full_resolution / 16, view.full_resolution);

			if (m_camera_path.rendering && !m_gen3c_render_with_gen3c) {
				new_render_res = m_camera_path.render_settings.resolution;
			}

			float ratio = std::sqrt((float)product(render_res) / (float)product(new_render_res));
			if (ratio > 1.2f || ratio < 0.8f || factor == 1.0f || !m_dynamic_res || (m_camera_path.rendering && !m_gen3c_render_with_gen3c)) {
				render_res = new_render_res;
			}

			if (view.render_buffer->dlss()) {
				render_res = view.render_buffer->dlss()->clamp_resolution(render_res);
				view.render_buffer->dlss()->update_feature(
					render_res, view.render_buffer->dlss()->is_hdr(), view.render_buffer->dlss()->sharpen()
				);
			}

			view.render_buffer->resize(render_res);

			if (m_foveated_rendering) {
				if (m_dynamic_foveated_rendering) {
					vec2 resolution_scale = vec2(render_res) / vec2(view.full_resolution);

					// Only start foveation when DLSS if off or if DLSS is asked to do more than 1.5x upscaling.
					// The reason for the 1.5x threshold is that DLSS can do up to 3x upscaling, at which point a
					// foveation factor of 2x = 3.0x/1.5x corresponds exactly to bilinear super sampling, which is
					// helpful in suppressing DLSS's artifacts.
					float foveation_begin_factor = m_dlss ? 1.5f : 1.0f;

					resolution_scale =
						clamp(resolution_scale * foveation_begin_factor, vec2(1.0f / m_foveated_rendering_max_scaling), vec2(1.0f));
					view.foveation = {resolution_scale, vec2(1.0f) - view.screen_center, vec2(m_foveated_rendering_full_res_diameter * 0.5f)};

					m_foveated_rendering_scaling = 2.0f / sum(resolution_scale);
				} else {
					view.foveation = {
						vec2(1.0f / m_foveated_rendering_scaling),
						vec2(1.0f) - view.screen_center,
						vec2(m_foveated_rendering_full_res_diameter * 0.5f)
					};
				}
			} else {
				view.foveation = {};
			}
		}
	}

	// Make sure all in-use auxiliary GPUs have the latest model and bitfield
	std::unordered_set<CudaDevice*> devices_in_use;
	for (auto& view : m_views) {
		if (!view.device || devices_in_use.count(view.device) != 0) {
			continue;
		}

		devices_in_use.insert(view.device);
		sync_device(*view.render_buffer, *view.device);
	}

	if (m_reproject_enable) {
		render_by_reprojection(m_stream.get(), m_views);
	} else {
		SyncedMultiStream synced_streams{m_stream.get(), m_views.size()};

		std::vector<std::future<void>> futures(m_views.size());
		for (size_t i = 0; i < m_views.size(); ++i) {
			auto& view = m_views[i];
			futures[i] = view.device->enqueue_task([this, &view, stream = synced_streams.get(i)]() {
				auto device_guard = use_device(stream, *view.render_buffer, *view.device);
				render_frame_main(
					*view.device, view.camera0, view.camera1, view.screen_center, view.relative_focal_length, view.foveation, view.lens, view.visualized_dimension
				);
			});
		}

		for (size_t i = 0; i < m_views.size(); ++i) {
			auto& view = m_views[i];

			if (futures[i].valid()) {
				futures[i].get();
			}

			render_frame_epilogue(
				synced_streams.get(i),
				view.camera0,
				view.prev_camera,
				view.screen_center,
				view.relative_focal_length,
				view.foveation,
				view.prev_foveation,
				view.lens,
				*view.render_buffer,
				true
			);

			view.prev_camera = view.camera0;
			view.prev_foveation = view.foveation;
		}
	}

	for (size_t i = 0; i < m_views.size(); ++i) {
		m_rgba_render_textures.at(i)->blit_from_cuda_mapping();
		m_depth_render_textures.at(i)->blit_from_cuda_mapping();
	}

	if (m_picture_in_picture_res > 0) {
		ivec2 res{(int)m_picture_in_picture_res, (int)(m_picture_in_picture_res * 9.0f / 16.0f)};
		m_pip_render_buffer->resize(res);
		if (m_pip_render_buffer->spp() < 8) {
			// a bit gross, but let's copy the keyframe's state into the global state in order to not have to plumb
			// through the fov etc to render_frame.
			CameraKeyframe backup = copy_camera_to_keyframe();
			CameraKeyframe pip_kf = m_camera_path.eval_camera_path(m_camera_path.play_time);
			set_camera_from_keyframe(pip_kf);

			if (m_reproject_enable) {
				std::vector<View> views(1);
				auto& view = views.front();
				view.camera0 = pip_kf.m();
				view.camera1 = pip_kf.m();
				view.prev_camera = pip_kf.m();
				view.screen_center = m_screen_center;
				view.relative_focal_length = m_relative_focal_length;
				view.foveation = {};
				view.prev_foveation = {};
				view.lens = lens;
				view.visualized_dimension = m_visualized_dimension;
				view.render_buffer = m_pip_render_buffer;

				render_by_reprojection(m_stream.get(), views);
			} else {
				render_frame(
					m_stream.get(),
					pip_kf.m(),
					pip_kf.m(),
					pip_kf.m(),
					m_screen_center,
					m_relative_focal_length,
					{}, // foveation
					{}, // prev foveation
					lens,
					m_visualized_dimension,
					*m_pip_render_buffer
				);
			}

			set_camera_from_keyframe(backup);
			m_pip_render_texture->blit_from_cuda_mapping();
		}
	}
#endif

	CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));
}

mat4x3 Testbed::view_camera(size_t view_idx) const {
	if (m_views.size() <= view_idx) {
		throw std::runtime_error{fmt::format("View #{} does not exist.", view_idx)};
	}

	auto& view = m_views.at(view_idx);
	return view.camera0;
}


#ifdef NGP_GUI
void Testbed::create_second_window() {
	if (m_second_window.window) {
		return;
	}

	bool frameless = false;
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, !frameless);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_CENTER_CURSOR, false);
	glfwWindowHint(GLFW_DECORATED, !frameless);
	glfwWindowHint(GLFW_SCALE_TO_MONITOR, frameless);
	glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, true);
	// get the window size / coordinates
	int win_w = 0, win_h = 0, win_x = 0, win_y = 0;
	GLuint ps = 0, vs = 0;

	{
		win_w = 1920;
		win_h = 1080;
		win_x = 0x40000000;
		win_y = 0x40000000;
		static const char* copy_shader_vert =
			"\
			in vec2 vertPos_data;\n\
			out vec2 texCoords;\n\
			void main(){\n\
				gl_Position = vec4(vertPos_data.xy, 0.0, 1.0);\n\
				texCoords = (vertPos_data.xy + 1.0) * 0.5; texCoords.y=1.0-texCoords.y;\n\
			}";
		static const char* copy_shader_frag =
			"\
			in vec2 texCoords;\n\
			out vec4 fragColor;\n\
			uniform sampler2D screenTex;\n\
			void main(){\n\
				fragColor = texture(screenTex, texCoords.xy);\n\
			}";
		vs = compile_shader(false, copy_shader_vert);
		ps = compile_shader(true, copy_shader_frag);
	}

	m_second_window.window = glfwCreateWindow(win_w, win_h, "Fullscreen Output", NULL, m_glfw_window);
	if (win_x != 0x40000000) {
		glfwSetWindowPos(m_second_window.window, win_x, win_y);
	}

	glfwMakeContextCurrent(m_second_window.window);
	m_second_window.program = glCreateProgram();
	glAttachShader(m_second_window.program, vs);
	glAttachShader(m_second_window.program, ps);
	glLinkProgram(m_second_window.program);
	if (!check_shader(m_second_window.program, "shader program", true)) {
		glDeleteProgram(m_second_window.program);
		m_second_window.program = 0;
	}

	// vbo and vao
	glGenVertexArrays(1, &m_second_window.vao);
	glGenBuffers(1, &m_second_window.vbo);
	glBindVertexArray(m_second_window.vao);
	const float fsquadVerts[] = {-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f};
	glBindBuffer(GL_ARRAY_BUFFER, m_second_window.vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(fsquadVerts), fsquadVerts, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void Testbed::set_n_views(size_t n_views) {
	bool changed_views = n_views != m_views.size();

	while (m_views.size() > n_views) {
		m_views.pop_back();
	}

	m_rgba_render_textures.resize(n_views);
	m_depth_render_textures.resize(n_views);

	while (m_views.size() < n_views) {
		size_t idx = m_views.size();
		m_rgba_render_textures[idx] = std::make_shared<GLTexture>();
		m_depth_render_textures[idx] = std::make_shared<GLTexture>();
		m_views.emplace_back(View{std::make_shared<CudaRenderBuffer>(m_rgba_render_textures[idx], m_depth_render_textures[idx])});
	}

};
#endif // NGP_GUI

void Testbed::init_window(int resw, int resh, bool hidden, bool second_window) {
#ifndef NGP_GUI
	throw std::runtime_error{"init_window failed: NGP was built without GUI support"};
#else
	m_window_res = {resw, resh};

	glfwSetErrorCallback(glfw_error_callback);
	if (!glfwInit()) {
		throw std::runtime_error{"GLFW could not be initialized."};
	}

#	ifdef NGP_VULKAN
	// Only try to initialize DLSS (Vulkan+NGX) if the
	// GPU is sufficiently new. Older GPUs don't support
	// DLSS, so it is preferable to not make a futile
	// attempt and emit a warning that confuses users.
	if (primary_device().compute_capability() >= 70) {
		try {
			m_dlss_provider = init_vulkan_and_ngx();
		} catch (const std::runtime_error& e) {
			tlog::warning() << "Could not initialize Vulkan and NGX. DLSS not supported. (" << e.what() << ")";
		}
	}
#	endif

	glfwWindowHint(GLFW_VISIBLE, hidden ? GLFW_FALSE : GLFW_TRUE);
	std::string title = "Gen3C GUI";
	m_glfw_window = glfwCreateWindow(m_window_res.x, m_window_res.y, title.c_str(), NULL, NULL);
	if (m_glfw_window == NULL) {
		throw std::runtime_error{"GLFW window could not be created."};
	}
	glfwMakeContextCurrent(m_glfw_window);
#	ifdef _WIN32
	if (gl3wInit()) {
		throw std::runtime_error{"GL3W could not be initialized."};
	}
#	else
	glewExperimental = 1;
	if (glewInit()) {
		throw std::runtime_error{"GLEW could not be initialized."};
	}
#	endif
	glfwSwapInterval(m_vsync ? 1 : 0); // Disable vsync

	GLint gl_version_minor, gl_version_major;
	glGetIntegerv(GL_MINOR_VERSION, &gl_version_minor);
	glGetIntegerv(GL_MAJOR_VERSION, &gl_version_major);

	if (gl_version_major < 3 || (gl_version_major == 3 && gl_version_minor < 1)) {
		throw std::runtime_error{
			fmt::format("Unsupported OpenGL version {}.{}. Gen3C requires at least OpenGL 3.1", gl_version_major, gl_version_minor)
		};
	}

	tlog::success() << "Initialized OpenGL version " << glGetString(GL_VERSION);

	glfwSetWindowUserPointer(m_glfw_window, this);
	glfwSetDropCallback(m_glfw_window, [](GLFWwindow* window, int count, const char** paths) {
		Testbed* testbed = (Testbed*)glfwGetWindowUserPointer(window);
		if (!testbed) {
			return;
		}

		if (testbed->m_file_drop_callback) {
			if (testbed->m_file_drop_callback(std::vector<std::string>(paths, paths + count))) {
				// Files were handled by the callback.
				return;
			}
		}

		for (int i = 0; i < count; i++) {
			testbed->load_file(paths[i]);
		}
	});

	glfwSetKeyCallback(m_glfw_window, [](GLFWwindow* window, int key, int scancode, int action, int mods) {
		Testbed* testbed = (Testbed*)glfwGetWindowUserPointer(window);
		if (testbed) {
			testbed->redraw_gui_next_frame();
		}
	});

	glfwSetCursorPosCallback(m_glfw_window, [](GLFWwindow* window, double xpos, double ypos) {
		Testbed* testbed = (Testbed*)glfwGetWindowUserPointer(window);
		if (testbed && (ImGui::IsAnyItemActive() || ImGui::GetIO().WantCaptureMouse || ImGuizmo::IsUsing()) &&
			(ImGui::GetIO().MouseDown[0] || ImGui::GetIO().MouseDown[1] || ImGui::GetIO().MouseDown[2])) {
			testbed->redraw_gui_next_frame();
		}
	});

	glfwSetMouseButtonCallback(m_glfw_window, [](GLFWwindow* window, int button, int action, int mods) {
		Testbed* testbed = (Testbed*)glfwGetWindowUserPointer(window);
		if (testbed) {
			testbed->redraw_gui_next_frame();
		}
	});

	glfwSetScrollCallback(m_glfw_window, [](GLFWwindow* window, double xoffset, double yoffset) {
		Testbed* testbed = (Testbed*)glfwGetWindowUserPointer(window);
		if (testbed) {
			testbed->redraw_gui_next_frame();
		}
	});

	glfwSetWindowSizeCallback(m_glfw_window, [](GLFWwindow* window, int width, int height) {
		Testbed* testbed = (Testbed*)glfwGetWindowUserPointer(window);
		if (testbed) {
			testbed->redraw_next_frame();
		}
	});

	glfwSetFramebufferSizeCallback(m_glfw_window, [](GLFWwindow* window, int width, int height) {
		Testbed* testbed = (Testbed*)glfwGetWindowUserPointer(window);
		if (testbed) {
			testbed->redraw_next_frame();
		}
	});

	float xscale, yscale;
	glfwGetWindowContentScale(m_glfw_window, &xscale, &yscale);

	// IMGUI init
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	(void)io;

	// By default, imgui places its configuration (state of the GUI -- size of windows, which regions are expanded, etc.) in ./imgui.ini
	// relative to the working directory. Instead, we would like to place imgui.ini in the directory that Gen3C project resides in.
	static std::string ini_filename;
	ini_filename = (root_dir() / "imgui.ini").str();
	io.IniFilename = ini_filename.c_str();

	// New ImGui event handling seems to make camera controls laggy if input trickling is true. So disable input trickling.
	io.ConfigInputTrickleEventQueue = false;
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(m_glfw_window, true);
	ImGui_ImplOpenGL3_Init("#version 140");

	ImGui::GetStyle().ScaleAllSizes(xscale);
	ImFontConfig font_cfg;
	font_cfg.SizePixels = 13.0f * xscale;
	io.Fonts->AddFontDefault(&font_cfg);
	ImFontConfig overlay_font_cfg;
	overlay_font_cfg.SizePixels = 128.0f * xscale;
	m_imgui.overlay_font = io.Fonts->AddFontDefault(&overlay_font_cfg);

	init_opengl_shaders();

	// Make sure there's at least one usable render texture
	set_n_views(1);
	m_views.front().full_resolution = m_window_res;
	m_views.front().render_buffer->resize(m_views.front().full_resolution);

	m_pip_render_texture = std::make_shared<GLTexture>();
	m_pip_render_buffer = std::make_shared<CudaRenderBuffer>(m_pip_render_texture);

	m_render_window = true;

	if (m_second_window.window == nullptr && second_window) {
		create_second_window();
	}
#endif // NGP_GUI
}

void Testbed::destroy_window() {
#ifndef NGP_GUI
	throw std::runtime_error{"destroy_window failed: NGP was built without GUI support"};
#else
	if (!m_render_window) {
		throw std::runtime_error{"Window must be initialized to be destroyed."};
	}

	m_hmd.reset();

	m_views.clear();
	m_rgba_render_textures.clear();
	m_depth_render_textures.clear();

	m_pip_render_buffer.reset();
	m_pip_render_texture.reset();

	m_dlss = false;
	m_dlss_provider.reset();

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glfwDestroyWindow(m_glfw_window);
	glfwTerminate();

	m_blit_program = 0;
	m_blit_vao = 0;

	m_glfw_window = nullptr;
	m_render_window = false;
#endif // NGP_GUI
}

void Testbed::init_vr() {
#ifndef NGP_GUI
	throw std::runtime_error{"init_vr failed: NGP was built without GUI support"};
#else
	try {
		if (!m_glfw_window) {
			throw std::runtime_error{"`init_window` must be called before `init_vr`"};
		}

#	if defined(XR_USE_PLATFORM_WIN32)
		m_hmd = std::make_unique<OpenXRHMD>(wglGetCurrentDC(), glfwGetWGLContext(m_glfw_window));
#	elif defined(XR_USE_PLATFORM_XLIB)
		Display* xDisplay = glfwGetX11Display();
		GLXContext glxContext = glfwGetGLXContext(m_glfw_window);

		int glxFBConfigXID = 0;
		glXQueryContext(xDisplay, glxContext, GLX_FBCONFIG_ID, &glxFBConfigXID);
		int attributes[3] = {GLX_FBCONFIG_ID, glxFBConfigXID, 0};
		int nelements = 1;
		GLXFBConfig* pglxFBConfig = glXChooseFBConfig(xDisplay, 0, attributes, &nelements);
		if (nelements != 1 || !pglxFBConfig) {
			throw std::runtime_error{"init_vr(): Couldn't obtain GLXFBConfig"};
		}

		GLXFBConfig glxFBConfig = *pglxFBConfig;

		XVisualInfo* visualInfo = glXGetVisualFromFBConfig(xDisplay, glxFBConfig);
		if (!visualInfo) {
			throw std::runtime_error{"init_vr(): Couldn't obtain XVisualInfo"};
		}

		m_hmd = std::make_unique<OpenXRHMD>(xDisplay, visualInfo->visualid, glxFBConfig, glXGetCurrentDrawable(), glxContext);
#	elif defined(XR_USE_PLATFORM_WAYLAND)
		m_hmd = std::make_unique<OpenXRHMD>(glfwGetWaylandDisplay());
#	endif

		// Enable aggressive optimizations to make the VR experience smooth.
		update_vr_performance_settings();

		// If multiple GPUs are available, shoot for 60 fps in VR.
		// Otherwise, it wouldn't be realistic to expect more than 30.
		m_dynamic_res_target_fps = m_devices.size() > 1 ? 60 : 30;
		m_background_color = {0.0f, 0.0f, 0.0f, 0.0f};
	} catch (const std::runtime_error& e) {
		if (std::string{e.what()}.find("XR_ERROR_FORM_FACTOR_UNAVAILABLE") != std::string::npos) {
			throw std::runtime_error{
				"Could not initialize VR. Ensure that SteamVR, OculusVR, or any other OpenXR-compatible runtime is running. Also set it as the active OpenXR runtime."
			};
		} else {
			throw std::runtime_error{fmt::format("Could not initialize VR: {}", e.what())};
		}
	}
#endif // NGP_GUI
}

void Testbed::update_vr_performance_settings() {
#ifdef NGP_GUI
	if (m_hmd) {
		auto blend_mode = m_hmd->environment_blend_mode();

		// DLSS is instrumental in getting VR to look good. Enable if possible.
		// If the environment is blended in (such as in XR/AR applications),
		// DLSS causes jittering at object sillhouettes (doesn't deal well with alpha),
		// and hence stays disabled.
		m_dlss = (blend_mode == EEnvironmentBlendMode::Opaque) && m_dlss_provider;

		// Foveated rendering is similarly vital in getting high performance without losing
		// resolution in the middle of the view.
		m_foveated_rendering = true;

		// Many VR runtimes perform optical flow for automatic reprojection / motion smoothing.
		// This breaks down for solid-color background, sometimes leading to artifacts. Hence:
		// set background color to transparent and, in spherical_checkerboard_kernel(...),
		// blend a checkerboard. If the user desires a solid background nonetheless, they can
		// set the background color to have an alpha value of 1.0 manually via the GUI or via Python.
		m_render_transparency_as_checkerboard = (blend_mode == EEnvironmentBlendMode::Opaque);
	} else {
		m_foveated_rendering = false;
		m_render_transparency_as_checkerboard = false;
	}
#endif // NGP_GUI
}

bool Testbed::frame() {
#ifdef NGP_GUI
	if (m_render_window) {
		if (!begin_frame()) {
			return false;
		}

		handle_user_input();
		begin_vr_frame_and_handle_vr_input();
	}
#endif

	bool skip_rendering = false;
	if (!m_dlss && m_max_spp > 0 && !m_views.empty() && m_views.front().render_buffer->spp() >= m_max_spp) {
		skip_rendering = true;
	}

	if (m_camera_path.rendering && !m_gen3c_render_with_gen3c) {
		prepare_next_camera_path_frame();
		skip_rendering = false;
	}

	if (m_record_camera_path && !m_views.empty()) {
		m_camera_path.spline_order = 1;
		const float timestamp = m_camera_path.duration_seconds() + m_frame_ms.val() / 1000.0f;
		m_camera_path.add_camera(m_views[0].camera0, focal_length_to_fov(1.0f, m_views[0].relative_focal_length[m_fov_axis]), timestamp);

		m_camera_path.keyframe_subsampling = (int)m_camera_path.keyframes.size();
		m_camera_path.editing_kernel_type = EEditingKernel::Gaussian;
	}

#ifdef NGP_GUI
	if (m_hmd && m_hmd->is_visible()) {
		skip_rendering = false;
	}
#endif

	if (!skip_rendering || std::chrono::steady_clock::now() - m_last_gui_draw_time_point > 50ms) {
		redraw_gui_next_frame();
	}

	try {
		while (true) {
			(*m_task_queue.tryPop())();
		}
	} catch (const SharedQueueEmptyException&) {}

	render(skip_rendering);

#ifdef NGP_GUI
	if (m_render_window) {
		if (m_gui_redraw) {
			draw_gui();
			m_gui_redraw = false;

			m_last_gui_draw_time_point = std::chrono::steady_clock::now();
		}

		ImGui::EndFrame();
	}

	if (m_hmd && m_vr_frame_info) {
		// If HMD is visible to the user, splat rendered images to the HMD
		if (m_hmd->is_visible()) {
			size_t n_views = std::min(m_views.size(), m_vr_frame_info->views.size());

			// Blit textures to the OpenXR-owned framebuffers (each corresponding to one eye)
			for (size_t i = 0; i < n_views; ++i) {
				const auto& vr_view = m_vr_frame_info->views.at(i);

				ivec2 resolution = {
					vr_view.view.subImage.imageRect.extent.width,
					vr_view.view.subImage.imageRect.extent.height,
				};

				blit_texture(
					m_views.at(i).foveation,
					m_rgba_render_textures.at(i)->texture(),
					GL_LINEAR,
					m_depth_render_textures.at(i)->texture(),
					vr_view.framebuffer,
					ivec2(0),
					resolution
				);
			}

			glFinish();
		}

		// Far and near planes are intentionally reversed, because we map depth inversely
		// to z. I.e. a window-space depth of 1 refers to the near plane and a depth of 0
		// to the far plane. This results in much better numeric precision.
		m_hmd->end_frame(m_vr_frame_info, m_ndc_zfar / m_scale, m_ndc_znear / m_scale, m_vr_use_depth_reproject);
	}
#endif

	return true;
}

bool Testbed::want_repl() {
	bool b = m_want_repl;
	m_want_repl = false;
	return b;
}

void Testbed::apply_camera_smoothing(float elapsed_ms) {
	// Ensure our camera rotation remains an orthogonal matrix as numeric
	// errors accumulate across frames.
	m_camera = orthogonalize(m_camera);

	if (m_camera_smoothing) {
		float decay = std::pow(0.02f, elapsed_ms / 1000.0f);
		m_smoothed_camera = orthogonalize(camera_log_lerp(m_smoothed_camera, m_camera, 1.0f - decay));
	} else {
		m_smoothed_camera = m_camera;
	}
}

CameraKeyframe Testbed::copy_camera_to_keyframe() const { return CameraKeyframe(m_camera, fov(), 0.0f); }

void Testbed::set_camera_from_keyframe(const CameraKeyframe& k) {
	m_camera = k.m();
	set_fov(k.fov);
}

void Testbed::set_camera_from_time(float t) {
	if (m_camera_path.keyframes.empty()) {
		return;
	}

	set_camera_from_keyframe(m_camera_path.eval_camera_path(t));
}

float Testbed::fov() const { return focal_length_to_fov(1.0f, m_relative_focal_length[m_fov_axis]); }

void Testbed::set_fov(float val) { m_relative_focal_length = vec2(fov_to_focal_length(1, val)); }

vec2 Testbed::fov_xy() const { return focal_length_to_fov(ivec2(1), m_relative_focal_length); }

void Testbed::set_fov_xy(const vec2& val) { m_relative_focal_length = fov_to_focal_length(ivec2(1), val); }

Testbed::Testbed(ETestbedMode mode) {
	tcnn::set_log_callback([](LogSeverity severity, const std::string& msg) {
		tlog::ESeverity s = tlog::ESeverity::Info;
		switch (severity) {
			case LogSeverity::Info: s = tlog::ESeverity::Info; break;
			case LogSeverity::Debug: s = tlog::ESeverity::Debug; break;
			case LogSeverity::Warning: s = tlog::ESeverity::Warning; break;
			case LogSeverity::Error: s = tlog::ESeverity::Error; break;
			case LogSeverity::Success: s = tlog::ESeverity::Success; break;
			default: break;
		}
		tlog::log(s) << msg;
	});

	if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2))) {
		throw std::runtime_error{"Testbed requires CUDA 10.2 or later."};
	}

#ifdef NGP_GUI
	// Ensure we're running on the GPU that'll host our GUI. To do so, try creating a dummy
	// OpenGL context, figure out the GPU it's running on, and then kill that context again.
	if (!is_wsl() && glfwInit()) {
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
		GLFWwindow* offscreen_context = glfwCreateWindow(640, 480, "", NULL, NULL);

		if (offscreen_context) {
			glfwMakeContextCurrent(offscreen_context);

			int gl_device = -1;
			unsigned int device_count = 0;
			if (cudaGLGetDevices(&device_count, &gl_device, 1, cudaGLDeviceListAll) == cudaSuccess) {
				if (device_count > 0 && gl_device >= 0) {
					set_cuda_device(gl_device);
				}
			}

			glfwDestroyWindow(offscreen_context);
		}

		glfwTerminate();
	}
#endif

	// Reset our stream, which was allocated on the originally active device,
	// to make sure it corresponds to the now active device.
	m_stream = {};

	int active_device = cuda_device();
	int active_compute_capability = cuda_compute_capability();
	tlog::success() << fmt::format(
		"Initialized CUDA {}. Active GPU is #{}: {} [{}]", cuda_runtime_version_string(), active_device, cuda_device_name(), active_compute_capability
	);

	if (active_compute_capability < MIN_GPU_ARCH) {
		tlog::warning() << "Insufficient compute capability " << active_compute_capability << " detected.";
		tlog::warning() << "This program was compiled for >=" << MIN_GPU_ARCH << " and may thus behave unexpectedly.";
	}

	m_devices.emplace_back(active_device, true);

	int n_devices = cuda_device_count();
	for (int i = 0; i < n_devices; ++i) {
		if (i == active_device) {
			continue;
		}

		if (cuda_compute_capability(i) >= MIN_GPU_ARCH) {
			m_devices.emplace_back(i, false);
		}
	}

	if (m_devices.size() > 1) {
		tlog::success() << "Detected auxiliary GPUs:";
		for (size_t i = 1; i < m_devices.size(); ++i) {
			const auto& device = m_devices[i];
			tlog::success() << "  #" << device.id() << ": " << device.name() << " [" << device.compute_capability() << "]";
		}
	}

	set_mode(mode);
	set_exposure(0);

	reset_camera();
}

Testbed::~Testbed() {
	// If any temporary file was created, make sure it's deleted
	clear_tmp_dir();

	if (m_render_window) {
		destroy_window();
	}
}

bool Testbed::clear_tmp_dir() {
	wait_all(m_render_futures);
	m_render_futures.clear();

	bool success = true;
	auto tmp_dir = fs::path{"tmp"};
	if (tmp_dir.exists()) {
		if (tmp_dir.is_directory()) {
			for (const auto& path : fs::directory{tmp_dir}) {
				if (path.is_file()) {
					success &= path.remove_file();
				}
			}
		}

		success &= tmp_dir.remove_file();
	}

	return success;
}

vec2 Testbed::calc_focal_length(const ivec2& resolution, const vec2& relative_focal_length, int fov_axis, float zoom) const {
	return relative_focal_length * (float)resolution[fov_axis] * zoom;
}

vec2 Testbed::render_screen_center(const vec2& screen_center) const {
	// see pixel_to_ray for how screen center is used; 0.5, 0.5 is 'normal'. we flip so that it becomes the point in the
	// original image we want to center on.
	return (0.5f - screen_center) * m_zoom + 0.5f;
}

__global__ void dlss_prep_kernel(
	ivec2 resolution,
	uint32_t sample_index,
	vec2 focal_length,
	vec2 screen_center,
	vec3 parallax_shift,
	bool snap_to_pixel_centers,
	float* depth_buffer,
	const float znear,
	const float zfar,
	mat4x3 camera,
	mat4x3 prev_camera,
	cudaSurfaceObject_t depth_surface,
	cudaSurfaceObject_t mvec_surface,
	cudaSurfaceObject_t exposure_surface,
	Foveation foveation,
	Foveation prev_foveation,
	Lens lens
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x || y >= resolution.y) {
		return;
	}

	uint32_t idx = x + resolution.x * y;

	uint32_t x_orig = x;
	uint32_t y_orig = y;

	const float depth = depth_buffer[idx];
	vec2 mvec = motion_vector(
		sample_index,
		{(int)x, (int)y},
		resolution,
		focal_length,
		camera,
		prev_camera,
		screen_center,
		parallax_shift,
		snap_to_pixel_centers,
		depth,
		foveation,
		prev_foveation,
		lens
	);

	surf2Dwrite(make_float2(mvec.x, mvec.y), mvec_surface, x_orig * sizeof(float2), y_orig);

	// DLSS was trained on games, which presumably used standard normalized device coordinates (ndc)
	// depth buffers. So: convert depth to NDC with reasonable near- and far planes.
	surf2Dwrite(to_ndc_depth(depth, znear, zfar), depth_surface, x_orig * sizeof(float), y_orig);

	// First thread write an exposure factor of 1. Since DLSS will run on tonemapped data,
	// exposure is assumed to already have been applied to DLSS' inputs.
	if (x_orig == 0 && y_orig == 0) {
		surf2Dwrite(1.0f, exposure_surface, 0, 0);
	}
}

__global__ void spherical_checkerboard_kernel(
	ivec2 resolution,
	vec2 focal_length,
	mat4x3 camera,
	vec2 screen_center,
	vec3 parallax_shift,
	Foveation foveation,
	Lens lens,
	vec4 background_color,
	vec4* frame_buffer
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x || y >= resolution.y) {
		return;
	}

	Ray ray = pixel_to_ray(
		0,
		{(int)x, (int)y},
		resolution,
		focal_length,
		camera,
		screen_center,
		parallax_shift,
		false,
		0.0f,
		1.0f,
		0.0f,
		foveation,
		{}, // No need for hidden area mask
		lens
	);

	// Blend with checkerboard to break up reprojection weirdness in some VR runtimes
	host_device_swap(ray.d.z, ray.d.y);
	vec2 spherical = dir_to_spherical(normalize(ray.d)) * 32.0f / PI();
	const vec4 dark_gray = {0.5f, 0.5f, 0.5f, 1.0f};
	const vec4 light_gray = {0.55f, 0.55f, 0.55f, 1.0f};
	vec4 checker = fabsf(fmodf(floorf(spherical.x) + floorf(spherical.y), 2.0f)) < 0.5f ? dark_gray : light_gray;

	// Blend background color on top of checkerboard first (checkerboard is meant to be "behind" the background,
	// representing transparency), and then blend the result behind the frame buffer.
	background_color.rgb() = srgb_to_linear(background_color.rgb());
	background_color += (1.0f - background_color.a) * checker;

	uint32_t idx = x + resolution.x * y;
	frame_buffer[idx] += (1.0f - frame_buffer[idx].a) * background_color;
}

__global__ void vr_overlay_hands_kernel(
	ivec2 resolution,
	vec2 focal_length,
	mat4x3 camera,
	vec2 screen_center,
	vec3 parallax_shift,
	Foveation foveation,
	Lens lens,
	vec3 left_hand_pos,
	float left_grab_strength,
	vec4 left_hand_color,
	vec3 right_hand_pos,
	float right_grab_strength,
	vec4 right_hand_color,
	float hand_radius,
	EColorSpace output_color_space,
	cudaSurfaceObject_t surface
	// TODO: overwrite depth buffer
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x || y >= resolution.y) {
		return;
	}

	Ray ray = pixel_to_ray(
		0,
		{(int)x, (int)y},
		resolution,
		focal_length,
		camera,
		screen_center,
		parallax_shift,
		false,
		0.0f,
		1.0f,
		0.0f,
		foveation,
		{}, // No need for hidden area mask
		lens
	);

	vec4 color = vec4(0.0f);
	auto composit_hand = [&](vec3 hand_pos, float grab_strength, vec4 hand_color) {
		// Don't render the hand indicator if it's behind the ray origin.
		if (dot(ray.d, hand_pos - ray.o) < 0.0f) {
			return;
		}

		float distance = ray.distance_to(hand_pos);

		vec4 base_color = vec4(0.0f);
		const vec4 border_color = {0.4f, 0.4f, 0.4f, 0.4f};

		// Divide hand radius into an inner part (4/5ths) and a border (1/5th).
		float radius = hand_radius * 0.8f;
		float border_width = hand_radius * 0.2f;

		// When grabbing, shrink the inner part as a visual indicator.
		radius *= 0.5f + 0.5f * (1.0f - grab_strength);

		if (distance < radius) {
			base_color = hand_color;
		} else if (distance < radius + border_width) {
			base_color = border_color;
		} else {
			return;
		}

		// Make hand color opaque when grabbing.
		base_color.a = grab_strength + (1.0f - grab_strength) * base_color.a;
		color += base_color * (1.0f - color.a);
	};

	if (dot(ray.d, left_hand_pos - ray.o) < dot(ray.d, right_hand_pos - ray.o)) {
		composit_hand(left_hand_pos, left_grab_strength, left_hand_color);
		composit_hand(right_hand_pos, right_grab_strength, right_hand_color);
	} else {
		composit_hand(right_hand_pos, right_grab_strength, right_hand_color);
		composit_hand(left_hand_pos, left_grab_strength, left_hand_color);
	}

	// Blend with existing color of pixel
	vec4 prev_color;
	surf2Dread((float4*)&prev_color, surface, x * sizeof(float4), y);
	if (output_color_space == EColorSpace::SRGB) {
		prev_color.rgb() = srgb_to_linear(prev_color.rgb());
	}

	color += (1.0f - color.a) * prev_color;

	if (output_color_space == EColorSpace::SRGB) {
		color.rgb() = linear_to_srgb(color.rgb());
	}

	surf2Dwrite(to_float4(color), surface, x * sizeof(float4), y);
}

void Testbed::render_by_reprojection(cudaStream_t stream, std::vector<View>& views) {
	// Reprojection from view cache
	int n_src_views = std::max(std::min(m_reproject_max_src_view_index, (int)m_reproject_src_views.size()) - m_reproject_min_src_view_index, 0);

	std::vector<const View*> src_views(n_src_views);
	for (int i = 0; i < n_src_views; ++i) {
		// Invert order of src views to reproject from the most recent one first and fill in the holes / closer content with older views.
		src_views[n_src_views - i - 1] = &m_reproject_src_views[i + m_reproject_min_src_view_index];
	}

	for (size_t i = 0; i < views.size(); ++i) {
		auto& view = views[i];

		reproject_views(src_views, view);

		render_frame_epilogue(
			stream,
			view.camera0,
			view.prev_camera,
			view.screen_center,
			view.relative_focal_length,
			view.foveation,
			view.prev_foveation,
			view.lens,
			*view.render_buffer,
			true
		);

		view.prev_camera = view.camera0;
		view.prev_foveation = view.foveation;
	}
}

void Testbed::render_frame(
	cudaStream_t stream,
	const mat4x3& camera_matrix0,
	const mat4x3& camera_matrix1,
	const mat4x3& prev_camera_matrix,
	const vec2& orig_screen_center,
	const vec2& relative_focal_length,
	const Foveation& foveation,
	const Foveation& prev_foveation,
	const Lens& lens,
	int visualized_dimension,
	CudaRenderBuffer& render_buffer,
	bool to_srgb,
	CudaDevice* device
) {
	if (!device) {
		device = &primary_device();
	}

	sync_device(render_buffer, *device);

	{
		auto device_guard = use_device(stream, render_buffer, *device);
		render_frame_main(
			*device, camera_matrix0, camera_matrix1, orig_screen_center, relative_focal_length, foveation, lens, visualized_dimension
		);
	}

	render_frame_epilogue(
		stream, camera_matrix0, prev_camera_matrix, orig_screen_center, relative_focal_length, foveation, prev_foveation, lens, render_buffer, to_srgb
	);
}

void Testbed::render_frame_main(
	CudaDevice& device,
	const mat4x3& camera_matrix0,
	const mat4x3& camera_matrix1,
	const vec2& orig_screen_center,
	const vec2& relative_focal_length,
	const Foveation& foveation,
	const Lens& lens,
	int visualized_dimension
) {
	device.render_buffer_view().clear(device.stream());

	vec2 focal_length = calc_focal_length(device.render_buffer_view().resolution, relative_focal_length, m_fov_axis, m_zoom);
	vec2 screen_center = render_screen_center(orig_screen_center);
}

void Testbed::render_frame_epilogue(
	cudaStream_t stream,
	const mat4x3& camera_matrix0,
	const mat4x3& prev_camera_matrix,
	const vec2& orig_screen_center,
	const vec2& relative_focal_length,
	const Foveation& foveation,
	const Foveation& prev_foveation,
	const Lens& lens,
	CudaRenderBuffer& render_buffer,
	bool to_srgb
) {
	vec2 focal_length = calc_focal_length(render_buffer.in_resolution(), relative_focal_length, m_fov_axis, m_zoom);
	vec2 screen_center = render_screen_center(orig_screen_center);

	render_buffer.set_color_space(m_color_space);
	render_buffer.set_tonemap_curve(m_tonemap_curve);

	// Prepare DLSS data: motion vectors, scaled depth, exposure
	if (render_buffer.dlss()) {
		auto res = render_buffer.in_resolution();

		const dim3 threads = {16, 8, 1};
		const dim3 blocks = {div_round_up((uint32_t)res.x, threads.x), div_round_up((uint32_t)res.y, threads.y), 1};

		dlss_prep_kernel<<<blocks, threads, 0, stream>>>(
			res,
			render_buffer.spp(),
			focal_length,
			screen_center,
			m_parallax_shift,
			m_snap_to_pixel_centers,
			render_buffer.depth_buffer(),
			m_ndc_znear,
			m_ndc_zfar,
			camera_matrix0,
			prev_camera_matrix,
			render_buffer.dlss()->depth(),
			render_buffer.dlss()->mvec(),
			render_buffer.dlss()->exposure(),
			foveation,
			prev_foveation,
			lens
		);

		render_buffer.set_dlss_sharpening(m_dlss_sharpening);
	}

	EColorSpace output_color_space = to_srgb ? EColorSpace::SRGB : EColorSpace::Linear;

	if (m_render_transparency_as_checkerboard) {
		mat4x3 checkerboard_transform = mat4x3::identity();

#ifdef NGP_GUI
		if (m_hmd && m_vr_frame_info && !m_vr_frame_info->views.empty()) {
			checkerboard_transform = m_vr_frame_info->views[0].pose;
		}
#endif

		auto res = render_buffer.in_resolution();
		const dim3 threads = {16, 8, 1};
		const dim3 blocks = {div_round_up((uint32_t)res.x, threads.x), div_round_up((uint32_t)res.y, threads.y), 1};
		spherical_checkerboard_kernel<<<blocks, threads, 0, stream>>>(
			res,
			focal_length,
			checkerboard_transform,
			screen_center,
			m_parallax_shift,
			foveation,
			lens,
			m_background_color,
			render_buffer.frame_buffer()
		);
	}

	render_buffer.accumulate(m_exposure, stream);
	render_buffer.tonemap(m_exposure, m_background_color, output_color_space, m_ndc_znear, m_ndc_zfar, m_snap_to_pixel_centers, stream);

#ifdef NGP_GUI
	// If in VR, indicate the hand position and render transparent background
	if (m_hmd && m_vr_frame_info) {
		auto& hands = m_vr_frame_info->hands;

		auto res = render_buffer.out_resolution();
		const dim3 threads = {16, 8, 1};
		const dim3 blocks = {div_round_up((uint32_t)res.x, threads.x), div_round_up((uint32_t)res.y, threads.y), 1};
		vr_overlay_hands_kernel<<<blocks, threads, 0, stream>>>(
			res,
			focal_length * vec2(render_buffer.out_resolution()) / vec2(render_buffer.in_resolution()),
			camera_matrix0,
			screen_center,
			m_parallax_shift,
			foveation,
			lens,
			vr_to_world(hands[0].pose[3]),
			hands[0].grab_strength,
			{hands[0].pressing ? 0.8f : 0.0f, 0.0f, 0.0f, 0.8f},
			vr_to_world(hands[1].pose[3]),
			hands[1].grab_strength,
			{hands[1].pressing ? 0.8f : 0.0f, 0.0f, 0.0f, 0.8f},
			0.05f * m_scale, // Hand radius
			output_color_space,
			render_buffer.surface()
		);
	}
#endif
}

float Testbed::get_depth_from_renderbuffer(const CudaRenderBuffer& render_buffer, const vec2& uv) {
	if (!render_buffer.depth_buffer()) {
		return m_scale;
	}

	float depth;
	auto res = render_buffer.in_resolution();
	ivec2 depth_pixel = clamp(ivec2(uv * vec2(res)), 0, res - 1);

	CUDA_CHECK_THROW(
		cudaMemcpy(&depth, render_buffer.depth_buffer() + depth_pixel.x + depth_pixel.y * res.x, sizeof(float), cudaMemcpyDeviceToHost)
	);
	return depth;
}

vec3 Testbed::get_3d_pos_from_pixel(const CudaRenderBuffer& render_buffer, const vec2& pixel) {
	float depth = get_depth_from_renderbuffer(render_buffer, pixel / vec2(m_window_res));
	auto ray = pixel_to_ray_pinhole(
		0,
		ivec2(pixel),
		m_window_res,
		calc_focal_length(m_window_res, m_relative_focal_length, m_fov_axis, m_zoom),
		m_smoothed_camera,
		render_screen_center(m_screen_center)
	);
	return ray(depth);
}

void Testbed::autofocus() {
	float new_slice_plane_z = std::max(dot(view_dir(), m_autofocus_target - view_pos()), 0.1f) - m_scale;
	if (new_slice_plane_z != m_slice_plane_z) {
		m_slice_plane_z = new_slice_plane_z;
		if (m_aperture_size != 0.0f) {
			reset_accumulation();
		}
	}
}

Testbed::LevelStats compute_level_stats(const float* params, size_t n_params) {
	Testbed::LevelStats s = {};
	for (size_t i = 0; i < n_params; ++i) {
		float v = params[i];
		float av = fabsf(v);
		if (av < 0.00001f) {
			s.numzero++;
		} else {
			if (s.count == 0) {
				s.min = s.max = v;
			}
			s.count++;
			s.x += v;
			s.xsquared += v * v;
			s.min = min(s.min, v);
			s.max = max(s.max, v);
		}
	}
	return s;
}

Testbed::CudaDevice::CudaDevice(int id, bool is_primary) : m_id{id}, m_is_primary{is_primary} {
	auto guard = device_guard();
	m_stream = std::make_unique<StreamAndEvent>();
	m_data = std::make_unique<Data>();
	m_render_worker = std::make_unique<ThreadPool>(is_primary ? 0u : 1u);
}

ScopeGuard Testbed::CudaDevice::device_guard() {
	int prev_device = cuda_device();
	if (prev_device == m_id) {
		return {};
	}

	set_cuda_device(m_id);
	return ScopeGuard{[prev_device]() { set_cuda_device(prev_device); }};
}

void Testbed::sync_device(CudaRenderBuffer& render_buffer, Testbed::CudaDevice& device) {
	if (!device.dirty()) {
		return;
	}

	if (device.is_primary()) {
		device.data().hidden_area_mask = render_buffer.hidden_area_mask();
		device.set_dirty(false);
		return;
	}

	m_stream.signal(device.stream());

	int active_device = cuda_device();
	auto guard = device.device_guard();

	if (render_buffer.hidden_area_mask()) {
		auto ham = std::make_shared<Buffer2D<uint8_t>>(render_buffer.hidden_area_mask()->resolution());
		CUDA_CHECK_THROW(cudaMemcpyPeerAsync(
			ham->data(), device.id(), render_buffer.hidden_area_mask()->data(), active_device, ham->bytes(), device.stream()
		));
		device.data().hidden_area_mask = ham;
	} else {
		device.data().hidden_area_mask = nullptr;
	}

	device.set_dirty(false);
	device.signal(m_stream.get());
}

// From https://stackoverflow.com/questions/20843271/passing-a-non-copyable-closure-object-to-stdfunction-parameter
template <class F> auto make_copyable_function(F&& f) {
	using dF = std::decay_t<F>;
	auto spf = std::make_shared<dF>(std::forward<F>(f));
	return [spf](auto&&... args) -> decltype(auto) { return (*spf)(decltype(args)(args)...); };
}

ScopeGuard Testbed::use_device(cudaStream_t stream, CudaRenderBuffer& render_buffer, Testbed::CudaDevice& device) {
	device.wait_for(stream);

	if (device.is_primary()) {
		device.set_render_buffer_view(render_buffer.view());
		return ScopeGuard{[&device, stream]() {
			device.set_render_buffer_view({});
			device.signal(stream);
		}};
	}

	int active_device = cuda_device();
	auto guard = device.device_guard();

	size_t n_pixels = product(render_buffer.in_resolution());

	GPUMemoryArena::Allocation alloc;
	auto scratch = allocate_workspace_and_distribute<vec4, float>(device.stream(), &alloc, n_pixels, n_pixels);

	device.set_render_buffer_view({
		std::get<0>(scratch),
		std::get<1>(scratch),
		render_buffer.in_resolution(),
		render_buffer.spp(),
		device.data().hidden_area_mask,
	});

	return ScopeGuard{
		make_copyable_function([&render_buffer, &device, guard = std::move(guard), alloc = std::move(alloc), active_device, stream]() {
			// Copy device's render buffer's data onto the original render buffer
			CUDA_CHECK_THROW(cudaMemcpyPeerAsync(
				render_buffer.frame_buffer(),
				active_device,
				device.render_buffer_view().frame_buffer,
				device.id(),
				product(render_buffer.in_resolution()) * sizeof(vec4),
				device.stream()
			));
			CUDA_CHECK_THROW(cudaMemcpyPeerAsync(
				render_buffer.depth_buffer(),
				active_device,
				device.render_buffer_view().depth_buffer,
				device.id(),
				product(render_buffer.in_resolution()) * sizeof(float),
				device.stream()
			));

			device.set_render_buffer_view({});
			device.signal(stream);
		})
	};
}

void Testbed::set_all_devices_dirty() {
	for (auto& device : m_devices) {
		device.set_dirty(true);
	}
}

void Testbed::load_camera_path(const fs::path& path) { m_camera_path.load(path, mat4x3::identity()); }

bool Testbed::loop_animation() { return m_camera_path.loop; }

void Testbed::set_loop_animation(bool value) { m_camera_path.loop = value; }

} // namespace ngp
