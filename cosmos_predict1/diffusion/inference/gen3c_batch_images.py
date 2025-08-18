# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import glob
import cv2
import numpy as np
import torch
import trimesh
from typing import List, Dict, Any, Union
import json
from time import time
from tqdm import tqdm
from PIL import Image


from cosmos_predict1.diffusion.inference.gen3c_persistent import Gen3cPersistentModel, create_parser
from cosmos_predict1.diffusion.inference.camera_utils import generate_camera_trajectory
from cosmos_predict1.diffusion.inference.gen3c_single_image import _predict_moge_depth
from cosmos_predict1.utils import log, misc
from cosmos_predict1.utils.io import save_video


from vggt_wrapper import VGGTWrapper
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def parse_arguments() -> argparse.Namespace:
    parser = create_parser()
    
    # Override/add specific arguments for batch processing
    parser.add_argument(
        "--input_images_dir",
        type=str,
        help="Directory containing input RGB images"
    )
    parser.add_argument(
        "--input_images_pattern",
        type=str,
        default="*.jpg",
        help="Pattern to match input images (e.g., *.jpg, *.png)"
    )
    parser.add_argument(
        "--input_videos_dir",
        type=str,
        help="Directory containing input videos"
    )
    parser.add_argument(
        "--input_videos_pattern",
        type=str,
        default="*.mp4",
        help="Pattern to match input videos (e.g., *.mp4, *.avi)"
    )
    parser.add_argument(
        "--output_images_dir",
        type=str,
        required=True,
        help="Directory to save output RGB images"
    )
    parser.add_argument(
        "--trajectory_config",
        type=str,
        help="JSON file with custom trajectory configuration"
    )
    parser.add_argument(
        "--custom_trajectory",
        type=str,
        choices=[
            "left", "right", "up", "down", 
            "zoom_in", "zoom_out", 
            "clockwise", "counterclockwise",
            "none"
        ],
        default="left",
        help="Predefined trajectory type"
    )
    parser.add_argument(
        "--save_as_video",
        action="store_true",
        help="Also save output as video files"
    )
    parser.add_argument(
        "--output_fps",
        type=int,
        default=24, 
        help="FPS for output videos"
    )
    parser.add_argument(
        "--frame_extraction_method",
        type=str,
        choices=["first", "middle", "last", "all", "interval", "first_max_frames"],
        default="first",
        help="Method to extract frames from videos: first, middle, last, all, or interval"
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=30,
        help="Frame interval for extraction when using 'interval' method"
    )

    parser.add_argument(
        "--step_size",
        type=int,
        default=1,
        help="Step size for frame extraction when using 'first_max_frames' method"
    )
    parser.add_argument(
        "--max_frames_per_video",
        type=int,
        default=10,
        help="Maximum frames to extract per video when using 'all' method"
    )
    parser.add_argument(
        "--max_inputs",
        type=int,
        default=None,
        help="Maximum number of input images/videos to process"
    )
    parser.add_argument(
        "--use_vggt",
        action="store_true",
        help="Use VGG-T for depth estimation"
    )

    return parser.parse_args()


def load_input_list(args: argparse.Namespace) -> List[Dict[str, Union[str, int]]]:
    """Load list of input files (images or videos) from directories."""
    inputs = []
    
    # Load images if specified
    if args.input_images_dir:
        pattern_path = os.path.join(args.input_images_dir, args.input_images_pattern)
        image_paths = sorted(glob.glob(pattern_path))
        
        for img_path in image_paths:
            inputs.append({
                "path": img_path,
                "type": "image",
                "frame_idx": 0
            })
    
    # Load videos if specified
    if args.input_videos_dir:
        pattern_path = os.path.join(args.input_videos_dir, args.input_videos_pattern)
        video_paths = sorted(glob.glob(pattern_path))
        
        for vid_path in video_paths:
            # Extract frames from video based on method
            frame_indices = extract_frame_indices_from_video(vid_path, args)
            
            for frame_idx in frame_indices:
                inputs.append({
                    "path": vid_path,
                    "type": "video",
                    "frame_idx": frame_idx
                })
    
    if not inputs:
        if args.input_images_dir and args.input_videos_dir:
            raise ValueError(f"No images or videos found in specified directories")
        elif args.input_images_dir:
            raise ValueError(f"No images found matching pattern: {os.path.join(args.input_images_dir, args.input_images_pattern)}")
        elif args.input_videos_dir:
            raise ValueError(f"No videos found matching pattern: {os.path.join(args.input_videos_dir, args.input_videos_pattern)}")
        else:
            raise ValueError("No input directory specified. Use --input_images_dir or --input_videos_dir")
    
    if args.max_inputs is not None:
        inputs = inputs[:args.max_inputs]
    
    log.info(f"Found {len(inputs)} input items to process")
    return inputs


def extract_frame_indices_from_video(video_path: str, args: argparse.Namespace) -> List[int]:
    """Extract frame indices from video based on specified method."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.warning(f"Could not open video: {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if total_frames == 0:
        log.warning(f"Video has no frames: {video_path}")
        return []
    
    if args.frame_extraction_method == "first":
        return [0]
    elif args.frame_extraction_method == "first_max_frames":
        return [i*args.step_size for i in range(0, min(total_frames, args.max_frames_per_video))]
    elif args.frame_extraction_method == "middle":
        return [total_frames // 2]
    elif args.frame_extraction_method == "last":
        return [total_frames - 1]
    elif args.frame_extraction_method == "all":
        # Extract all frames up to max limit
        max_frames = min(total_frames, args.max_frames_per_video)
        step = max(1, total_frames // max_frames)
        return list(range(0, total_frames, step))[:max_frames]
    elif args.frame_extraction_method == "interval":
        # Extract frames at specified intervals
        return list(range(0, total_frames, args.frame_interval))
    else:
        return [0]  # Default to first frame


def load_and_preprocess_input(input_item: Dict[str, Union[str, int]], target_height: int, target_width: int) -> np.ndarray:
    """Load and preprocess a single input (image or video frame).

    Args:
        input_item: Dictionary with keys 'path', 'type', and 'frame_idx'.
        target_height: Target height for resizing.
        target_width: Target width for resizing.
    Returns:
        Preprocessed image as a numpy array of shape [H, W, C].
        Original size of the image as a tuple (height, width).
    """

    file_path = input_item["path"]
    input_type = input_item["type"]
    frame_idx = input_item["frame_idx"]
    
    if input_type == "image":
        image_bgr = cv2.imread(file_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not load image: {file_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
    elif input_type == "video":
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {file_path}")
        
        # Seek to specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not read frame {frame_idx} from video: {file_path}")
        
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    else:
        raise ValueError(f"Unknown input type: {input_type}")
    
    # Resize to target dimensions

    original_height, original_width = image_rgb.shape[:2]

    image_original = image_rgb.copy()  # Keep original for reference

    if original_height != target_height or original_width != target_width:
        image_resized = cv2.resize(image_rgb, (target_width, target_height))
        log.info(f"Resized image from {original_width}x{original_height} to {target_width}x{target_height}")
        print(f"Resized image from {original_width}x{original_height} to {target_width}x{target_height}")
    else:
        image_resized = image_rgb

    # Convert to float [0, 1]
    image_float = image_resized.astype(np.float32) / 255.0
    image_original = image_original.astype(np.float32) / 255.0  # Also return original image in float format
    
    return image_float, image_original



def export_rgb_pointcloud_from_cache(
    cache_object,
    output_path: str,
    mask_keep_value: float = 0.5,
    subsample_step: int = 1,   # e.g. 4 if you only want every 4-th pixel
) -> None:
    """
    Convert the current contents of a Cache3D_* instance into an RGB point cloud on disk.

    Parameters
    ----------
    cache_object : Cache3D_Base
        The populated cache (Buffer or 4D work just as well).
    output_path : str | Path
        Destination *.ply file.
    mask_keep_value : float, default 0.5
        Pixels whose cache_object.input_mask < mask_keep_value are discarded.
        If the cache has no mask, everything is kept.
    subsample_step : int, default 1
        Use >1 to thin out very dense clouds (keep every subsample_step-th pixel
        along height and width to save RAM / disk space).
    """
    from pathlib import Path
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Shapes: B, F, N, V, H, W, 3  and  B, F, N, V, 3, H, W
    point_tensor = cache_object.input_points
    color_tensor = cache_object.input_image
    mask_tensor = cache_object.input_mask  # may be None

    if point_tensor is None:
        raise ValueError("cache_object.input_points is None – did you forget to feed depth or points?")

    # Bring tensors to CPU and flatten all dimensions except spatial
    point_tensor = point_tensor.cpu()[:, :, :, :, ::subsample_step, ::subsample_step].contiguous()
    color_tensor = color_tensor.cpu()[:, :, :, :, :, ::subsample_step, ::subsample_step].contiguous()

    batch_size, frame_count, buffer_count, view_count, height, width, _ = point_tensor.shape
    total_pixels = batch_size * frame_count * buffer_count * view_count * height * width

    point_coordinates = point_tensor.view(total_pixels, 3)  # (N_total, 3)
    color_channels_first = color_tensor.permute(0, 1, 2, 3, 5, 6, 4)  # move C from 5th pos to last
    # rgb_colors = (color_channels_first * 255.0).clamp(0.0, 255.0).to(torch.uint8).view(total_pixels, 3)
    rgb_colors = color_channels_first * 0.5 + 0.5

    rgb_colors = rgb_colors.reshape(total_pixels, 3)  # (N_total, 3)

    # Apply mask if present
    if mask_tensor is not None:
        mask_tensor = mask_tensor.cpu()[:, :, :, :, :, ::subsample_step, ::subsample_step]
        valid_mask = (mask_tensor.view(total_pixels) >= mask_keep_value)
        point_coordinates = point_coordinates[valid_mask]
        rgb_colors = rgb_colors[valid_mask]

    # Convert tensors to NumPy so plyfile can consume them
    point_coordinates = point_coordinates.numpy()
    rgb_colors = rgb_colors.numpy()


    trimesh_cache_points = trimesh.PointCloud(point_coordinates, rgb_colors)
    saved = trimesh_cache_points.export(str(output_path))



def main():
    args = parse_arguments()
    
    # Override some arguments for batch processing
    args.trajectory = "none"  # We'll handle trajectory generation ourselves
    args.batch_input_path = None
    args.input_image_path = None
    
    if args.prompt is None:
        args.prompt = ""
    args.disable_guardrail = True
    args.disable_prompt_upsampler = True
    args.offload_text_encoder_model = True
    args.offload_guardrail_models = True
    args.offload_diffusion_transformer = True
    args.offload_tokenizer = True
    args.offload_prompt_upsampler = True        # if you still want nicer prompts
    args.disable_prompt_encoder   = True        # maximal savings, poorer text control

    # args.use_vggt = True
    # args.use_vggt = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(args)

    vggt = VGGTWrapper()

    args.video_save_name = None 
    # Ensure output directory exists
    os.makedirs(args.output_images_dir, exist_ok=True)
    
    # Load input files (images and/or videos)
    input_items = load_input_list(args)
    
    # # Process each input
    images_np_bhwc_gen3c_resized = []
    images_np_original_shape_bhwc = []  # Store original shapes for resizing later
    # world_to_camera_np_b44 = []  # Shape: [B, 4, 4]
    # focal_lengths_np_b2 = [] # Shape: [B, 2]
    # principal_point_rel_np_b2 = []# Shape: [B, 2]
    # resolutions_np_b2 = []  # Shape: [B, 2]
    # masks_np_bhw = []  # Shape: [B, H, W]

    for idx, input_item in enumerate(input_items):
        log.info(f"Processing input {idx + 1}/{len(input_items)}: {input_item['path']} (type: {input_item['type']}, frame: {input_item['frame_idx']})")
        
        try:
            # Load and preprocess input
            image_np, image_original = load_and_preprocess_input(input_item, args.height, args.width)
            
            images_np_bhwc_gen3c_resized.append(image_np)  # Append to list
            images_np_original_shape_bhwc.append(image_original)  # Append original shape

        except Exception as e:
            log.error(f"Error processing input {input_item['path']}: {e}")


    # fixed Gen3C raster
    gen3c_height, gen3c_width = 704, 1280
    args.height = gen3c_height          # used by image loader
    args.width  = gen3c_width
    model = None # initialize later
    
    args.guidance = 1

    images_np_original_shape_bhwc = np.stack(images_np_original_shape_bhwc, axis=0)  # Shape: [B, H, W, C]
    images_np_bhwc_gen3c_resized = np.stack(images_np_bhwc_gen3c_resized, axis=0)  # Shape: [B, H, W, C]



    if args.use_vggt:
        log.info("Using VGGT for image processing...")
        # Resize images to 512x512 for VGGT

        output = vggt.infer_batch(frames=images_np_original_shape_bhwc, enable_downscaling=True)
        
        points, colors = vggt.infer_pointcloud(images_np_original_shape_bhwc, confidence_threshold=0.0005, enable_downscaling=True)
        point_cloud = trimesh.PointCloud(points, colors)
        point_cloud.export("vggt_input_scaled_test_point_cloud.ply")
        
        images_np_bchw_vggt_resized_and_padded_shape = output["images"].squeeze(0).cpu().numpy()  # Shape:
        images_np_bhwc_vggt_resized_and_padded_shape = images_np_bchw_vggt_resized_and_padded_shape.transpose(0, 2, 3, 1)  # Convert to [B, H, W, C]
        height, width = images_np_bhwc_vggt_resized_and_padded_shape.shape[1:3]


        print(f"Downscaling images from {images_np_bhwc_vggt_resized_and_padded_shape.shape[1:3]} to {height, width}")

        ex, intrinsics = pose_encoding_to_extri_intri(output["pose_enc"], image_size_hw=(height, width))

        B = ex.shape[1]
        pad = torch.tensor([[0, 0, 0, 1]], device=ex.device).repeat(1, B, 1, 1)
        ex = torch.cat([ex, pad], dim=2)
        w2cs = ex.squeeze(0).cpu().numpy().astype(np.float32)  # Shape: [B, 4, 4]

        depths_np = output["depth"].squeeze(0).squeeze(-1).cpu().numpy()  # Shape: [B, H, W]

        world_points_conf = output["world_points_conf"].squeeze(0).cpu().numpy()  # Shape: [B, H, W]
        world_points_conf = (world_points_conf - world_points_conf.min()) / (world_points_conf.max() - world_points_conf.min())  # Normalize to [0, 1]
        world_points_conf = world_points_conf > 0.0005  # Apply confidence threshold

        masks_np = world_points_conf.astype(np.float32)  # Convert to boolean mask
        print(f"Currently masked {masks_np.sum()} pixels out of {masks_np.size} total pixels, in percentage: {masks_np.sum() / masks_np.size * 100:.2f}%")

        world_to_cameras_np_b44 = w2cs  # Shape: [B, 4, 4]


        # ---------- VGGT-to-Gen3C: stretch directly to 704×1280 ------------------

        log.info(f"Resizing images to Gen3C dimensions (704x1280), original shape: {images_np_bhwc_vggt_resized_and_padded_shape.shape}")
        _, vggt_h, vggt_w, _ = images_np_bhwc_vggt_resized_and_padded_shape.shape

        scale_w = gen3c_width  / vggt_w
        scale_h = gen3c_height / vggt_h

        images_np_bhwc = np.array(
            [cv2.resize(img, (gen3c_width, gen3c_height), interpolation=cv2.INTER_LINEAR)
            for img in images_np_bhwc_vggt_resized_and_padded_shape]
        )

        # images_np_bhwc = images_np_bhwc_gen3c_resized

        depths_np = np.array(
            [cv2.resize(depth, (gen3c_width, gen3c_height), interpolation=cv2.INTER_LINEAR)
            for depth in depths_np]
        )
        # masks_np_bhw = np.array(
        #     [cv2.resize(mask.astype(np.float32), (gen3c_width, gen3c_height), interpolation=cv2.INTER_NEAREST)
        #         for mask in masks_np]   
        # ).astype(np.float32)
        # dilate masks
        # masks_np_bhw = np.array([cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=3) for mask in masks_np_bhw])

        # Use ones mask for Gen3C
        masks_np_bhw = np.ones_like(depths_np, dtype=np.float32)  # Assuming all pixels are valid for Gen3C


        Image.fromarray((images_np_bhwc[0] * 255).astype(np.uint8)).save("gen3c_input_scaled_test_image.png")   
        Image.fromarray((depths_np[0] * 50).astype(np.uint8)).save("gen3c_input_scaled_test_depth.png")    
        Image.fromarray((masks_np_bhw[0] * 255).astype(np.uint8)).save("gen3c_input_scaled_test_mask.png")

        focal_lengths_np_b2 = []
        principal_point_np_b2 = []

        for i in range(B):
            fx_orig = intrinsics.squeeze(0)[i, 0, 0].item()
            fy_orig = intrinsics.squeeze(0)[i, 1, 1].item()
            cx_orig = intrinsics.squeeze(0)[i, 0, 2].item()
            cy_orig = intrinsics.squeeze(0)[i, 1, 2].item()

            fx = fx_orig * scale_w
            fy = fy_orig * scale_h
            cx = cx_orig * scale_w
            cy = cy_orig * scale_h

            focal_lengths_np_b2.append([fx, fy])
            principal_point_np_b2.append([cx, cy])

        focal_lengths_np_b2 = np.array(focal_lengths_np_b2, dtype=np.float32)  # Shape: [B, 2]
        principal_point_np_b2 = np.array(principal_point_np_b2, dtype=np.float32)  # Shape: [B, 2]
        resolutions_np_b2 = np.array([[args.width, args.height]] * B, dtype=np.float32)  # Shape: [B, 2]

        # print(f"Scaled fx,fy → {focal_lengths_np_b2[0]}")
        log.info(f"Scaled fx,fy → {focal_lengths_np_b2[0]}")
        # print(f"Scaled cx,cy → {principal_point_np_b2[0]}")
        log.info(f"Scaled cx,cy → {principal_point_np_b2[0]}")

        # print(f"Resolutions before scaling: {resolutions_np_b2[0]}")

        log.info(f"Resolutions before scaling: {resolutions_np_b2[0]}")
        resolutions_np_b2 = np.array(
            [[gen3c_width, gen3c_height]] * B,
            dtype=np.float32
        )
        # print(f"Resolutions after scaling: {resolutions_np_b2[0]}")
        log.info(f"Resolutions after scaling: {resolutions_np_b2[0]}")

        log.info("Initializing Gen3C persistent model...")

        model = Gen3cPersistentModel(args) 
    

    else:
        log.info("Using MoGE for image processing...")
        log.info("Initializing Gen3C persistent model...")
        model = Gen3cPersistentModel(args) 

        (
            moge_image_b1chw_float,
            moge_depth_b11hw,
            moge_mask_b11hw,
            moge_initial_w2c_b144,
            moge_intrinsics_b133,
        ) = _predict_moge_depth(
            images_np_original_shape_bhwc.squeeze(0) * 255.0,  # Scale to [0, 255] for MoGE
            args.height, args.width, device, model.moge_model
        )

        moge_image_b1chw_float = (moge_image_b1chw_float + 1) / 2.0

        images_np_bhwc = moge_image_b1chw_float.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()  # Shape: [B, H, W, C]
        # log.info("images_np range:",float(images_np_bhwc.min()),float(images_np_bhwc.max()))
        depths_np = moge_depth_b11hw.squeeze(1).squeeze(1).cpu().numpy()  # Shape: [H, W]
        # masks_np_bhw = moge_mask_b11hw.squeeze(1).squeeze(1).cpu().numpy()  # Shape: [H, W]
        masks_np_bhw = np.ones_like(depths_np, dtype=np.float32)  # Assuming all pixels are valid for MoGE
        world_to_cameras_np_b44 = moge_initial_w2c_b144.squeeze(0).cpu().numpy()  # Shape: [4, 4]
        
        # focal_lengths_np_b2 = []
        # principal_point_np_b2 = []

        focal_lengths_np_b2 = np.array([[moge_intrinsics_b133[0][0, 0, 0].item(), moge_intrinsics_b133[0][0, 1, 1].item()]], dtype=np.float32)  # Shape: [1, 2]
        principal_point_np_b2 = np.array([[moge_intrinsics_b133[0][0, 0, 2].item(), moge_intrinsics_b133[0][0, 1, 2].item()]], dtype=np.float32)  # Shape: [1, 2]
        resolutions_np_b2 = np.array([[args.width, args.height]], dtype=np.float32)  # Shape: [1, 2]




#################### SEED MODEL MULTIPLE ####################
    depths_np_moge = depths_np.copy()  # Copy depths for MoGE


    model.seed_model_from_values(
                # images_np=images_batch,
                images_np=images_np_bhwc, # in [0,1] range please
                depths_np=depths_np_moge,
                # world_to_cameras_np=world_to_cameras_np,
                world_to_cameras_np=world_to_cameras_np_b44,
                # focal_lengths_np=focal_lengths_np,
                focal_lengths_np=focal_lengths_np_b2,
                # principal_point_rel_np=principal_point_rel_np,
                principal_point_rel_np=principal_point_np_b2 / np.array([gen3c_width, gen3c_height]),
                # resolutions=resolutions_np
                resolutions=resolutions_np_b2,
                masks_np=masks_np_bhw,
                # input_format=["B", "N", "C", "H", "W"],  # Assuming input format is [B, V, C, H, W], which is different from the defaul [F, C, H, W]
                treat_multi_inputs_as_views=True
            )
    




################### END OF SEED MODEL MULTIPLE ####################


    export_rgb_pointcloud_from_cache(
        model.cache,
        output_path="gen3c_cache_scaled_test_point_cloud.ply",
        mask_keep_value=0.5,  # Keep points with mask value >= 0.5
        subsample_step=1,  # Keep all points
    )

    intrinsics_np_b33 = np.array([
        [
            [focal_lengths_np_b2[0][0], 0, principal_point_np_b2[0][0]],
            [0, focal_lengths_np_b2[0][1], principal_point_np_b2[0][1]],
            [0, 0, 1]
        ]
    ], dtype=np.float32)  # Shape: [B, 3, 3]



    generated_w2cs, generated_intrinsics = generate_camera_trajectory(
        trajectory_type=args.custom_trajectory,
        initial_w2c=torch.from_numpy(world_to_cameras_np_b44[0]).to(device),
        initial_intrinsics=torch.from_numpy(intrinsics_np_b33[0]).to(device),
        num_frames=args.num_video_frames,
        movement_distance=args.movement_distance,
        camera_rotation=args.camera_rotation,
        center_depth=1.0,
        device=device,
    )
    
    view_cameras_w2cs = generated_w2cs.cpu().numpy().squeeze(0)  # Shape: [B, 4, 4]
    view_camera_intrinsics = generated_intrinsics.cpu().numpy().squeeze(0)  # Shape: [B

    result = model.inference_on_cameras(
        view_cameras_w2cs=view_cameras_w2cs,
        view_camera_intrinsics=view_camera_intrinsics,
        fps=args.output_fps,
        return_estimated_depths=False,
        save_buffer=args.save_buffer
    )

  
    if result is None:
        log.warning(f"Failed to generate video for input {input_item['path']}")
        return

    # Extract output name
    # base_name = get_output_name(input_item)
    base_name = os.path.splitext(os.path.basename(input_item["path"]))[0]
    
    # Save output images
    output_subdir = os.path.join(args.output_images_dir, base_name)

    
    # Save as video if requested
    if args.save_as_video:
        video_output_dir = os.path.join(args.output_images_dir, "videos")
        os.makedirs(video_output_dir, exist_ok=True)
        video_path = os.path.join(video_output_dir, f"{base_name}.mp4")
        
        # Convert video format for saving
        video_frames = result["video"][0]  # Remove batch dimension
        video_frames_uint8 = video_frames.transpose(0, 2, 3, 1)  # [T, H, W, C]

        save_video(
            video=video_frames_uint8,
            fps=args.output_fps,
            H=args.height,
            W=args.width,
            video_save_quality=5,
            video_save_path=video_path,
        )
        log.info(f"Saved video to {video_path}")
    
    # Clear cache for next input
    model.clear_cache()

   
    
    # Cleanup
    model.cleanup()
    log.info("Batch processing completed!")


if __name__ == "__main__":
    main()
