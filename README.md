# GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control

<div align="center">
  <img src="assets/demo_1.gif" alt=""  width="1100" />
</div>

**GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control**<br>
[Xuanchi Ren](https://xuanchiren.com/),
[Tianchang Shen](https://www.cs.toronto.edu/~shenti11/)
[Jiahui Huang](https://huangjh-pub.github.io/),
[Huan Ling](https://www.cs.toronto.edu/~linghuan/),
[Yifan Lu](https://yifanlu0227.github.io/),
[Merlin Nimier-David](https://yifanlu0227.github.io/),
[Thomas Müller](https://merlin.nimierdavid.fr/),
[Alexander Keller](https://research.nvidia.com/person/alex-keller),
[Sanja Fidler](https://www.cs.toronto.edu/~fidler/),
[Jun Gao](https://www.cs.toronto.edu/~jungao/) <br>
**[Paper](), [Project Page](https://research.nvidia.com/labs/toronto-ai/GEN3C/)**

Abstract: We present GEN3C, a generative video model with precise Camera Control and
temporal 3D Consistency. Prior video models already generate realistic videos,
but they tend to leverage little 3D information, leading to inconsistencies,
such as objects popping in and out of existence. Camera control, if implemented
at all, is imprecise, because camera parameters are mere inputs to the neural
network which must then infer how the video depends on the camera. In contrast,
GEN3C is guided by a 3D cache: point clouds obtained by predicting the
pixel-wise depth of seed images or previously generated frames. When generating
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


## Installation
<p align="center">:construction: :pick: :hammer_and_wrench: :construction_worker:</p>
<p align="center">Under construction. Stay tuned!</p>

## Acknowledgement
Our model is based on [NVIDIA Cosmos](https://github.com/NVIDIA/Cosmos) and [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid).

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