# GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control

<!-- Note: this video is hosted by GitHub and gets embedded automatically when viewing in the GitHub UI -->

https://github.com/user-attachments/assets/247e1719-9f8f-4504-bfa3-f9706bd8682d


**GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control**<br>
[Xuanchi Ren*](https://xuanchiren.com/),
[Tianchang Shen*](https://www.cs.toronto.edu/~shenti11/),
[Jiahui Huang](https://huangjh-pub.github.io/),
[Huan Ling](https://www.cs.toronto.edu/~linghuan/),
[Yifan Lu](https://yifanlu0227.github.io/),
[Merlin Nimier-David](https://merlin.nimierdavid.fr/),
[Thomas Müller](https://research.nvidia.com/person/thomas-muller),
[Alexander Keller](https://research.nvidia.com/person/alex-keller),
[Sanja Fidler](https://www.cs.toronto.edu/~fidler/),
[Jun Gao](https://www.cs.toronto.edu/~jungao/) <br>
\* indicates equal contribution <br>
**[Paper](https://arxiv.org/pdf/2503.03751), [Project Page](https://research.nvidia.com/labs/toronto-ai/GEN3C/), [HuggingFace](https://huggingface.co/collections/nvidia/gen3c-683f3f9540a8f9c98cf46a8d)**

Abstract: We present GEN3C, a generative video model with precise Camera Control and
temporal 3D Consistency. Prior video models already generate realistic videos,
but they tend to leverage little 3D information, leading to inconsistencies,
such as objects popping in and out of existence. Camera control, if implemented
at all, is imprecise, because camera parameters are mere inputs to the neural
network which must then infer how the video depends on the camera. In contrast,
GEN3C is guided by a 3D cache: point clouds obtained by predicting the
pixel-wise depth of seed images or previously generated frames. When gpromptenerating
the next frames, GEN3C is conditioned on the 2D renderings of the 3D cache with
the new camera trajectory provided by the user. Crucially, this means that
GEN3C neither has to remember what it previously generated nor does it have to
infer the image structure from the camera pose. The model, instead, can focus
all its generative power on previously unobserved regions, as well as advancing
the scene state to the next frame. Our results demonstrate more precise camera
control than prior work, as well as state-of-the-art results in sparse-view
novel view synthesis, even in challenging settings such as driving scenes and
monocular dynamic video. Results are best viewed in videos.

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/).
For any other questions related to the model, please contact Xuanchi, Tianchang or Jun.

## News
- 2025-06-06 Code and model released! In a future update, we plan to include the pipeline for jointly predicting depth and camera pose from video, as well as a driving-finetuned model. Stay tuned!

## Installation
Please follow the "Inference" section in [INSTALL.md](INSTALL.md) to set up your environment.

## Inference

### Download checkpoints
1. Generate a [Hugging Face](https://huggingface.co/settings/tokens) access token (if you haven't done so already). Set the access token to `Read` permission (default is `Fine-grained`).

2. Log in to Hugging Face with the access token:
   ```bash
   huggingface-cli login
   ```

3. Download the GEN3C model weights from [Hugging Face](https://huggingface.co/nvidia/GEN3C-Cosmos-7B):
   ```bash
   CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_gen3c_checkpoints.py --checkpoint_dir checkpoints
   ```

### Interactive GUI usage

<div align="center">
  <img src="gui/assets/gui_preview.webp" alt="GEN3C interactive GUI"  width="1080px"/>
</div>

GEN3C can be used through an interactive GUI, allowing to visualize the inputs in 3D, author arbitrary camera trajectories, and start inference from a single window.
Please see the [dedicated instructions](gui/README.md).


### Command-line usage
GEN3C supports both images and videos as input. Below are examples of running GEN3C on single images and videos with predefined camera trajectory patterns.

### Example 1: Single Image to Video Generation

#### Single GPU
Generate a 121-frame video from a single image:
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/gen3c_single_image.py \
    --checkpoint_dir checkpoints \
    --input_image_path assets/diffusion/000000.png \
    --video_save_name test_single_image \
    --guidance 1 \
    --foreground_masking
```

Running with cheap GPU usage:

```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/gen3c_single_image.py \
    --checkpoint_dir checkpoints \
    --input_image_path assets/diffusion/000000.png \
    --video_save_name test_single_image \
    --guidance 1 \
    --foreground_masking \
    --offload_diffusion_transformer \
    --offload_tokenizer \
    --offload_text_encoder_model \
    --offload_prompt_upsampler \
    --offload_guardrail_models
```


#### Static multiple view support with max pooling 

This is a walkaround support for reconstructing a scene from multiple views. Currently only single view is supported. The rest will be in [README_MULTIVIEW.md](README_MULTIVIEW.md).




#### Multi-GPU (8 GPUs)
```bash
NUM_GPUS=8
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=${NUM_GPUS} cosmos_predict1/diffusion/inference/gen3c_single_image.py \
    --checkpoint_dir checkpoints \
    --input_image_path assets/diffusion/000000.png \
    --video_save_name test_single_image_multigpu \
    --num_gpus ${NUM_GPUS} \
    --guidance 1 \
    --foreground_masking
```

#### Additional Options
- To generate longer videos autoregressively, specify the number of frames using `--num_video_frames`. The number of frames must follow the pattern: 121 * N - 1 (e.g., 241, 361, etc.)
- To save buffer images alongside the output video, add the `--save_buffer` flag
- You can control camera trajectories using `--trajectory`, `--camera_rotation`, and `--movement_distance` arguments. See the "Camera Movement Options" section below for details.

#### Camera Movement Options

##### Trajectory Types
The `--trajectory` argument controls the path the camera takes during video generation. Available options:

| Option | Description |
|--------|-------------|
| `left` | Camera moves to the left (default) |
| `right` | Camera moves to the right |
| `up` | Camera moves upward |
| `down` | Camera moves downward |
| `zoom_in` | Camera moves closer to the scene |
| `zoom_out` | Camera moves away from the scene |
| `clockwise` | Camera moves in a clockwise circular path |
| `counterclockwise` | Camera moves in a counterclockwise circular path |

##### Camera Rotation Modes
The `--camera_rotation` argument controls how the camera rotates during movement. Available options:

| Option | Description |
|--------|-------------|
| `center_facing` | Camera always rotates to look at the (estimated) center of the scene (default) |
| `no_rotation` | Camera maintains its original orientation while moving |
| `trajectory_aligned` | Camera rotates to align with the direction of movement |

##### Movement Distance
The `--movement_distance` argument controls how far the camera moves from its initial position. The default value is 0.3. A larger value will result in more dramatic camera movement, while a smaller value will create more subtle movement.

##### GPU Memory Requirements

We have tested GEN3C only on H100 and A100 GPUs. For GPUs with limited memory, you can fully offload all models by appending the following flags to your command:

```bash
--offload_diffusion_transformer \
--offload_tokenizer \
--offload_text_encoder_model \
--offload_prompt_upsampler \
--offload_guardrail_models \
--disable_guardrail \
--disable_prompt_encoder
```
Maximum observed memory during inference with full offloading: ~43GB. Note: Memory usage may vary depending on system specifications and is provided for reference only.


### Example 2: Video to Video Generation
For video input, GEN3C requires additional depth information, camera intrinsics, and extrinsics. These can be obtained using your choice of SLAM packages. For testing purposes, we provide example data.

First, you need to download the test samples:
```bash
# Download test samples from Hugging Face
huggingface-cli download nvidia/GEN3C-Testing-Example --repo-type dataset --local-dir assets/diffusion/dynamic_video_samples
```

#### Single GPU
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/gen3c_dynamic.py \
    --checkpoint_dir checkpoints \
    --input_image_path assets/diffusion/dynamic_video_samples/batch_0000 \
    --video_save_name test_dynamic_video \
    --guidance 1
```


#### Multi-GPU (8 GPUs)
```bash
NUM_GPUS=8
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=${NUM_GPUS} cosmos_predict1/diffusion/inference/gen3c_dynamic.py \
    --checkpoint_dir checkpoints \
    --input_image_path assets/diffusion/dynamic_video_samples/batch_0000 \
    --video_save_name test_dynamic_video_multigpu \
    --num_gpus ${NUM_GPUS} \
    --guidance 1
```


## Gallery

- **GEN3C** can be easily applied to video/scene creation from a single image
<div align="center">
  <img src="assets/demo_3.gif" alt=""  width="1100" />
</div>

- ... or sparse-view images (we use 5 images here)
<div align="center">
  <img src="assets/demo_2.gif" alt=""  width="1100" />
</div>


- .. and dynamic videos
<div align="center">
  <img src="assets/demo_dynamic.gif" alt=""  width="1100" />
</div>

## Acknowledgement
Our model is based on [NVIDIA Cosmos](https://github.com/NVIDIA/Cosmos) and [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid).

We are also grateful to several other open-source repositories that we drew inspiration from or built upon during the development of our pipeline:
- [MoGe](https://github.com/microsoft/MoGe)
- [TrajectoryCrafter](https://github.com/TrajectoryCrafter/TrajectoryCrafter)
- [DimensionX](https://github.com/wenqsun/DimensionX)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything)

## Citation
```
 @inproceedings{ren2025gen3c,
    title={GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control},
    author={Ren, Xuanchi and Shen, Tianchang and Huang, Jiahui and Ling, Huan and
        Lu, Yifan and Nimier-David, Merlin and Müller, Thomas and Keller, Alexander and
        Fidler, Sanja and Gao, Jun},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2025}
}
```

## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.


GEN3C source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

GEN3C models are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). For a custom license, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/).
