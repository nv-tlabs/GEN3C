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

/** @file   testbed.h
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#pragma once

#include <neural-graphics-primitives/adam_optimizer.h>
#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/camera_path.h>
#include <neural-graphics-primitives/common_host.h>
#include <neural-graphics-primitives/discrete_distribution.h>
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/shared_queue.h>
#include <neural-graphics-primitives/thread_pool.h>

#ifdef NGP_GUI
#	include <neural-graphics-primitives/openxr_hmd.h>
#endif

#include <tiny-cuda-nn/multi_stream.h>
#include <tiny-cuda-nn/random.h>

#include <json/json.hpp>

#ifdef NGP_PYTHON
#	include <pybind11/numpy.h>
#	include <pybind11/pybind11.h>
#endif

#include <deque>
#include <thread>

struct GLFWwindow;

namespace ngp {

struct Triangle;
class GLTexture;

struct ViewIdx {
	i16vec2 px;
	uint32_t view;
};

class Testbed {
public:
	Testbed(ETestbedMode mode = ETestbedMode::None);
	~Testbed();

	bool clear_tmp_dir();
	void update_imgui_paths();

	void set_mode(ETestbedMode mode);

	using distance_fun_t = std::function<void(uint32_t, const vec3*, float*, cudaStream_t)>;
	using normals_fun_t = std::function<void(uint32_t, const vec3*, vec3*, cudaStream_t)>;

	struct LevelStats {
		float mean() { return count ? (x / (float)count) : 0.f; }
		float variance() { return count ? (xsquared - (x * x) / (float)count) / (float)count : 0.f; }
		float sigma() { return sqrtf(variance()); }
		float fraczero() { return (float)numzero / float(count + numzero); }
		float fracquant() { return (float)numquant / float(count); }

		float x;
		float xsquared;
		float min;
		float max;
		int numzero;
		int numquant;
		int count;
	};

	class CudaDevice;

	struct View {
		std::shared_ptr<CudaRenderBuffer> render_buffer = nullptr;
		ivec2 full_resolution = {1, 1};
		int visualized_dimension = 0;

		mat4x3 camera0 = mat4x3::identity();
		mat4x3 camera1 = mat4x3::identity();
		mat4x3 prev_camera = mat4x3::identity();

		Foveation foveation;
		Foveation prev_foveation;

		vec2 relative_focal_length;
		vec2 screen_center;

		Lens lens;

		CudaDevice* device = nullptr;

		GPUImage<ViewIdx> index_field;
		GPUImage<uint8_t> hole_mask;
		GPUImage<float> depth_buffer;


		vec2 fov() const { return relative_focal_length_to_fov(relative_focal_length); }

		uint32_t uid = 0;
	};

	void render_by_reprojection(cudaStream_t stream, std::vector<View>& views);

	void render_frame(
		cudaStream_t stream,
		const mat4x3& camera_matrix0,
		const mat4x3& camera_matrix1,
		const mat4x3& prev_camera_matrix,
		const vec2& screen_center,
		const vec2& relative_focal_length,
		const Foveation& foveation,
		const Foveation& prev_foveation,
		const Lens& lens,
		int visualized_dimension,
		CudaRenderBuffer& render_buffer,
		bool to_srgb = true,
		CudaDevice* device = nullptr
	);
	void render_frame_main(
		CudaDevice& device,
		const mat4x3& camera_matrix0,
		const mat4x3& camera_matrix1,
		const vec2& screen_center,
		const vec2& relative_focal_length,
		const Foveation& foveation,
		const Lens& lens,
		int visualized_dimension
	);
	void render_frame_epilogue(
		cudaStream_t stream,
		const mat4x3& camera_matrix0,
		const mat4x3& prev_camera_matrix,
		const vec2& screen_center,
		const vec2& relative_focal_length,
		const Foveation& foveation,
		const Foveation& prev_foveation,
		const Lens& lens,
		CudaRenderBuffer& render_buffer,
		bool to_srgb = true
	);

	void init_camera_path_from_reproject_src_cameras();
	void visualize_reproject_src_cameras(ImDrawList* list, const mat4& world2proj);
	void clear_src_views();

	void reset_accumulation(bool due_to_camera_movement = false, bool immediate_redraw = true, bool reset_pip = false);
	void redraw_next_frame() { m_render_skip_due_to_lack_of_camera_movement_counter = 0; }
	bool reprojection_available() { return m_dlss; }
	void load_mesh(const fs::path& data_path);
	void set_exposure(float exposure) { m_exposure = exposure; }
	void translate_camera(const vec3& rel, const mat3& rot, bool allow_up_down = true);
	mat3 rotation_from_angles(const vec2& angles) const;
	void mouse_drag();
	void mouse_wheel();
	void load_file(const fs::path& path);
	vec3 look_at() const;
	void set_look_at(const vec3& pos);
	float scale() const { return m_scale; }
	void set_scale(float scale);
	vec3 view_pos() const { return m_camera[3]; }
	vec3 view_dir() const { return m_camera[2]; }
	vec3 view_up() const { return m_camera[1]; }
	vec3 view_side() const { return m_camera[0]; }
	void set_view_dir(const vec3& dir);
	void reset_camera();
	bool keyboard_event();
	void update_density_grid_mean_and_bitfield(cudaStream_t stream);
	void mark_density_grid_in_sphere_empty(const vec3& pos, float radius, cudaStream_t stream);

	void prepare_next_camera_path_frame();
	void overlay_fps();
	void imgui();
	vec2 calc_focal_length(const ivec2& resolution, const vec2& relative_focal_length, int fov_axis, float zoom) const;
	vec2 render_screen_center(const vec2& screen_center) const;
	void optimise_mesh_step(uint32_t N_STEPS);
	void compute_mesh_vertex_colors();

	float get_depth_from_renderbuffer(const CudaRenderBuffer& render_buffer, const vec2& uv);
	vec3 get_3d_pos_from_pixel(const CudaRenderBuffer& render_buffer, const vec2& focus_pixel);
	void autofocus();

#ifdef NGP_PYTHON
	std::pair<pybind11::array_t<float>, pybind11::array_t<float>>
		render_to_cpu(int width, int height, int spp, bool linear, float start_t, float end_t, float fps, float shutter_fraction);
	pybind11::array_t<float>
		render_to_cpu_rgba(int width, int height, int spp, bool linear, float start_t, float end_t, float fps, float shutter_fraction);
	pybind11::array_t<float> view(bool linear, size_t view) const;
	std::pair<pybind11::array_t<float>, pybind11::array_t<uint32_t>>
		reproject(const mat4x3& src, const pybind11::array_t<float>& src_img, const pybind11::array_t<float>& src_depth, const mat4x3& dst);
	uint32_t add_src_view(
		mat4x3 camera_to_world,
		float fx,
		float fy,
		float cx,
		float cy,
		Lens lens,
		pybind11::array_t<float> img,
		pybind11::array_t<float> depth,
		float timestamp,
		bool is_srgb = false
	);
	pybind11::array_t<uint32_t> src_view_ids() const;
#	ifdef NGP_GUI
	pybind11::array_t<float> screenshot(bool linear, bool front_buffer) const;
#	endif
#endif

	mat4x3 view_camera(size_t view) const;


	void draw_visualizations(ImDrawList* list, const mat4x3& camera_matrix);
	void reproject_views(const std::vector<const View*> src, View& dst);
	void render(bool skip_rendering);
	void init_window(int resw, int resh, bool hidden = false, bool second_window = false);
	void destroy_window();
	void init_vr();
	void update_vr_performance_settings();
	void apply_camera_smoothing(float elapsed_ms);
	bool begin_frame();
	void handle_user_input();
	vec3 vr_to_world(const vec3& pos) const;
	void begin_vr_frame_and_handle_vr_input();
	void draw_gui();
	bool frame();
	bool want_repl();
	void load_image(const fs::path& data_path);
	void load_exr_image(const fs::path& data_path);
	void load_stbi_image(const fs::path& data_path);
	void load_binary_image(const fs::path& data_path);
	float fov() const;
	void set_fov(float val);
	vec2 fov_xy() const;
	void set_fov_xy(const vec2& val);
	CameraKeyframe copy_camera_to_keyframe() const;
	void set_camera_from_keyframe(const CameraKeyframe& k);
	void set_camera_from_time(float t);
	void load_camera_path(const fs::path& path);
	bool loop_animation();
	void set_loop_animation(bool value);

	fs::path root_dir();
	void set_root_dir(const fs::path& dir);

	bool m_want_repl = false;

	bool m_render_window = false;
	bool m_gather_histograms = false;

	bool m_render_ground_truth = false;
	EGroundTruthRenderMode m_ground_truth_render_mode = EGroundTruthRenderMode::Shade;
	float m_ground_truth_alpha = 1.0f;

	bool m_render = true;
	int m_max_spp = 0;
	ETestbedMode m_testbed_mode = ETestbedMode::None;

	// Rendering stuff
	ivec2 m_window_res = ivec2(0);
	bool m_dynamic_res = false;
	float m_dynamic_res_target_fps = 20.0f;
	int m_fixed_res_factor = 8;
	float m_scale = 1.0;
	float m_aperture_size = 0.0f;
	vec2 m_relative_focal_length = vec2(1.0f);
	uint32_t m_fov_axis = 1;
	float m_zoom = 1.f;                // 2d zoom factor (for insets?)
	vec2 m_screen_center = vec2(0.5f); // center of 2d zoom

	float m_ndc_znear = 1.0f / 32.0f;
	float m_ndc_zfar = 128.0f;

	mat4x3 m_camera = mat4x3::identity();
	mat4x3 m_default_camera = transpose(mat3x4{1.0f, 0.0f, 0.0f, 0.5f, 0.0f, -1.0f, 0.0f, 0.5f, 0.0f, 0.0f, -1.0f, 0.5f});
	mat4x3 m_smoothed_camera = mat4x3::identity();
	size_t m_render_skip_due_to_lack_of_camera_movement_counter = 0;

	bool m_fps_camera = false;
	bool m_camera_smoothing = false;
	bool m_autofocus = false;
	vec3 m_autofocus_target = vec3(0.5f);

	bool m_render_with_lens_distortion = false;
	Lens m_render_lens = {};

	CameraPath m_camera_path = {};
	bool m_record_camera_path = false;

	vec3 m_up_dir = {0.0f, 1.0f, 0.0f};
	vec3 m_sun_dir = normalize(vec3(1.0f));
	float m_bounding_radius = 1;
	float m_exposure = 0.f;

	ERenderMode m_render_mode = ERenderMode::Shade;

	uint32_t m_seed = 1337;

#ifdef NGP_GUI
	GLFWwindow* m_glfw_window = nullptr;
	struct SecondWindow {
		GLFWwindow* window = nullptr;
		GLuint program = 0;
		GLuint vao = 0, vbo = 0;
		void draw(GLuint texture);
	} m_second_window;

	float m_drag_depth = 1.0f;

	// The VAO will be empty, but we need a valid one for attribute-less rendering
	GLuint m_blit_vao = 0;
	GLuint m_blit_program = 0;

	void init_opengl_shaders();
	void blit_texture(
		const Foveation& foveation,
		GLint rgba_texture,
		GLint rgba_filter_mode,
		GLint depth_texture,
		GLint framebuffer,
		const ivec2& offset,
		const ivec2& resolution
	);

	void create_second_window();

	std::unique_ptr<OpenXRHMD> m_hmd;
	OpenXRHMD::FrameInfoPtr m_vr_frame_info;

	bool m_vr_use_depth_reproject = false;
	bool m_vr_use_hidden_area_mask = false;

	std::deque<View> m_reproject_src_views;
	View m_reproject_pending_view;

	int m_reproject_min_src_view_index = 0;
	int m_reproject_max_src_view_index = 1;
	int m_reproject_max_src_view_count = -1;  // -1 indicates unlimited
	uint32_t m_reproject_selected_src_view = 0;
	bool m_reproject_freeze_src_views = false;
	int m_reproject_n_views_to_cache = 1;
	bool m_reproject_visualize_src_views = false;

	float m_reproject_min_t = 0.1f;
	float m_reproject_step_factor = 1.05f;
	vec3 m_reproject_parallax = vec3(0.0f, 0.0f, 0.0f);
	bool m_reproject_enable = false;
	bool m_reproject_reuse_last_frame = true;

	float m_reproject_lazy_render_ms = 100.0f;
	float m_reproject_lazy_render_res_factor = 1.25f;


	bool m_pm_enable = false;
	EPmVizMode m_pm_viz_mode = EPmVizMode::Shade;

	void set_n_views(size_t n_views);

	// Callback invoked when a keyboard event is detected.
	// If the callback returns `true`, the event is considered handled and the default behavior will not occur.
	std::function<bool()> m_keyboard_event_callback;

	// Callback invoked when a file is dropped onto the window.
	// If the callback returns `true`, the files are considered handled and the default behavior will not occur.
	std::function<bool(const std::vector<std::string>&)> m_file_drop_callback;

	std::shared_ptr<GLTexture> m_pip_render_texture;
	std::vector<std::shared_ptr<GLTexture>> m_rgba_render_textures;
	std::vector<std::shared_ptr<GLTexture>> m_depth_render_textures;
#endif

	std::shared_ptr<CudaRenderBuffer> m_pip_render_buffer;

	SharedQueue<std::unique_ptr<ICallable>> m_task_queue;

	void redraw_gui_next_frame() { m_gui_redraw = true; }

	bool m_gui_redraw = true;

	enum EDataType {
		Float,
		Half,
	};

	struct VolPayload {
		vec3 dir;
		vec4 col;
		uint32_t pixidx;
	};

	float m_camera_velocity = 1.0f;
	EColorSpace m_color_space = EColorSpace::Linear;
	ETonemapCurve m_tonemap_curve = ETonemapCurve::Identity;
	bool m_dlss = false;
	std::shared_ptr<IDlssProvider> m_dlss_provider;
	float m_dlss_sharpening = 0.0f;

	// 3D stuff
	float m_render_near_distance = 0.0f;
	float m_slice_plane_z = 0.0f;
	bool m_floor_enable = false;
	inline float get_floor_y() const { return m_floor_enable ? m_aabb.min.y + 0.001f : -10000.f; }
	BoundingBox m_raw_aabb;
	BoundingBox m_aabb = {vec3(0.0f), vec3(1.0f)};
	BoundingBox m_render_aabb = {vec3(0.0f), vec3(1.0f)};
	mat3 m_render_aabb_to_local = mat3::identity();

	// Rendering/UI bookkeeping
	Ema<float> m_render_ms = {EEmaType::Time, 100};
	// The frame contains everything, i.e. rendering + GUI and buffer swapping
	Ema<float> m_frame_ms = {EEmaType::Time, 100};
	std::chrono::time_point<std::chrono::steady_clock> m_last_frame_time_point;
	std::chrono::time_point<std::chrono::steady_clock> m_last_gui_draw_time_point;
	vec4 m_background_color = {0.0f, 0.0f, 0.0f, 1.0f};

	bool m_vsync = true;
	bool m_render_transparency_as_checkerboard = false;

	// Visualization of neuron activations
	int m_visualized_dimension = -1;
	int m_visualized_layer = 0;

	std::vector<View> m_views;
	ivec2 m_n_views = {1, 1};

	float m_picture_in_picture_res = 0.f; // if non zero, requests a small second picture :)

	enum class ImGuiMode : uint32_t {
		Enabled,
		FpsOverlay,
		Disabled,
		// Don't set the below
		NumModes,
	};

	struct ImGuiVars {
		static const uint32_t MAX_PATH_LEN = 1024;

		ImGuiMode mode = ImGuiMode::Enabled; // tab to cycle
		char cam_path_path[MAX_PATH_LEN] = "cam.json";
		char video_path[MAX_PATH_LEN] = "video.mp4";
		char cam_export_path[MAX_PATH_LEN] = "cam_export.json";

		void* overlay_font = nullptr;
	} m_imgui;

	fs::path m_root_dir = "";

	bool m_visualize_unit_cube = false;
	bool m_edit_render_aabb = false;
	bool m_edit_world_transform = true;

	bool m_snap_to_pixel_centers = false;

	vec3 m_parallax_shift = {0.0f, 0.0f, 0.0f}; // to shift the viewer's origin by some amount in camera space

	StreamAndEvent m_stream;

	class CudaDevice {
	public:
		struct Data {
			std::shared_ptr<Buffer2D<uint8_t>> hidden_area_mask;
		};

		CudaDevice(int id, bool is_primary);

		CudaDevice(const CudaDevice&) = delete;
		CudaDevice& operator=(const CudaDevice&) = delete;

		CudaDevice(CudaDevice&&) = default;
		CudaDevice& operator=(CudaDevice&&) = default;

		ScopeGuard device_guard();

		int id() const { return m_id; }

		bool is_primary() const { return m_is_primary; }

		std::string name() const { return cuda_device_name(m_id); }

		int compute_capability() const { return cuda_compute_capability(m_id); }

		cudaStream_t stream() const { return m_stream->get(); }

		void wait_for(cudaStream_t stream) const {
			CUDA_CHECK_THROW(cudaEventRecord(m_primary_device_event.event, stream));
			m_stream->wait_for(m_primary_device_event.event);
		}

		void signal(cudaStream_t stream) const { m_stream->signal(stream); }

		const CudaRenderBufferView& render_buffer_view() const { return m_render_buffer_view; }

		void set_render_buffer_view(const CudaRenderBufferView& view) { m_render_buffer_view = view; }

		Data& data() const { return *m_data; }

		bool dirty() const { return m_dirty; }

		void set_dirty(bool value) { m_dirty = value; }

		void clear() {
			m_data = std::make_unique<Data>();
			m_render_buffer_view = {};
			set_dirty(true);
		}

		template <class F> auto enqueue_task(F&& f) -> std::future<std::result_of_t<F()>> {
			if (is_primary()) {
				return std::async(std::launch::deferred, std::forward<F>(f));
			} else {
				return m_render_worker->enqueue_task(std::forward<F>(f));
			}
		}

	private:
		int m_id;
		bool m_is_primary;
		std::unique_ptr<StreamAndEvent> m_stream;
		struct Event {
			Event() { CUDA_CHECK_THROW(cudaEventCreate(&event)); }

			~Event() { cudaEventDestroy(event); }

			Event(const Event&) = delete;
			Event& operator=(const Event&) = delete;
			Event(Event&& other) { *this = std::move(other); }
			Event& operator=(Event&& other) {
				std::swap(event, other.event);
				return *this;
			}

			cudaEvent_t event = {};
		};
		Event m_primary_device_event;
		std::unique_ptr<Data> m_data;
		CudaRenderBufferView m_render_buffer_view = {};

		bool m_dirty = true;

		std::unique_ptr<ThreadPool> m_render_worker;
	};

	void sync_device(CudaRenderBuffer& render_buffer, CudaDevice& device);
	ScopeGuard use_device(cudaStream_t stream, CudaRenderBuffer& render_buffer, CudaDevice& device);
	void set_all_devices_dirty();

	std::vector<CudaDevice> m_devices;
	CudaDevice& primary_device() { return m_devices.front(); }

	ThreadPool m_thread_pool;
	std::vector<std::future<void>> m_render_futures;

	bool m_use_aux_devices = false;
	bool m_foveated_rendering = false;
	bool m_dynamic_foveated_rendering = true;
	float m_foveated_rendering_full_res_diameter = 0.55f;
	float m_foveated_rendering_scaling = 1.0f;
	float m_foveated_rendering_max_scaling = 2.0f;
	bool m_foveated_rendering_visualize = false;

	default_rng_t m_rng;

	CudaRenderBuffer m_windowless_render_surface{std::make_shared<CudaSurface2D>()};

	// ---------- Gen3C stuff
	/**
	 * Common signature for Gen3C-related UI callback functions, to be implemented
	 * in Python.
	 *
	 * Inputs:
	 *   name: name of the UI event (e.g. name of the button pressed).
	 *
	 * Returns: bool, whether the operation was successful.
	 */
	using gen3c_cb_t = std::function<bool(const std::string&)>;
	gen3c_cb_t m_gen3c_cb;

	// Info string to be displayed in the Gen3C UI window.
	std::string m_gen3c_info;
	// Path to an image or directory to use to seed the generative model.
	// The specific format is guessed based on what the path points to.
	std::string m_gen3c_seed_path;
	// Whether to automatically launch new inference requests.
	bool m_gen3c_auto_inference = false;

	EGen3cCameraSource m_gen3c_camera_source = EGen3cCameraSource::Authored;
	// Fake translation speed in scene unit / frame.
	vec3 m_gen3c_translation_speed = {0.05f, 0.f, 0.f};
	// Fake rotation speed around (x, y, z) in radians / frame.
	vec3 m_gen3c_rotation_speed = {0.f, 0.05f, 0.f};

	// Number of frames to request for each inference request.
	std::string m_gen3c_inference_info = "";

	// Progress of seeding-related things (scale 0..1). Set to a negative value to hide the progress bar.
	float m_gen3c_seeding_progress = -1.0f;
	// Progress of inference-related things (scale 0..1). Set to a negative value to hide the progress bar.
	float m_gen3c_inference_progress = -1.0f;

	// Saving Gen3C inference outputs
	bool m_gen3c_save_frames = false;
	// Whether or not to display generated frames in the UI.
	// No display means that we can save some time by not de-compressing
	// the result video from the server, and even skip depth prediction for most frames.
	bool m_gen3c_display_frames = false;
	std::string m_gen3c_output_dir = "";

	// When rendering with Gen3C, whether to include the rendered cache in the generated video (for debugging / visualization)
	bool m_gen3c_show_cache_renderings = false;

	bool m_gen3c_inference_is_connected = false;
	// Either we render the camera path from the local pointcloud or we use the inference server to get a photoreal video
	bool m_gen3c_render_with_gen3c = true;
};

} // namespace ngp
