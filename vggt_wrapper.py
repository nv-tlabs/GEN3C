#!/usr/bin/env python
# vggt_interface.py

import argparse
import gc
import os
import shutil
from pathlib import Path
from time import time

import cv2
import numpy as np
import pdb
import torch
from beartype import beartype
from typeguard import typechecked
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# from utils.sfm_utils import (
#     compute_co_vis_masks,
#     get_sorted_image_files,
#     init_filestructure,
#     save_extrinsic,
#     save_images_and_masks,
#     save_intrinsics,
#     save_points3D,
#     save_time,
#     split_train_test,
# )


class VGGTWrapper:
    """
    Unified VGGT interface preserving original signatures and outputs:
      - per-frame depth inference
      - batch inference
      - pointcloud extraction
      - camera pose extraction
      - full SFM init (_run_vggt_sfm)
      - directory processing (process_images_directory)
      - results loading
    """

    @beartype
    def __init__(self, model_path: str = "facebook/VGGT-1B", device: str = "cuda:0"):
        """
        Args:
            model_path: pretrained VGGT identifier or path
            device: torch device string
        """

        self.device = torch.device(device)
        self.model: VGGT = VGGT.from_pretrained(model_path).to(self.device).eval()
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.batch_size = 50
        self.downscale = 2
        # model attributes
        # self.max_image_size = 2048
        self.max_image_size = self.model.aggregator.patch_embed.patch_embed.img_size[0]
        self.patch_size = self.model.aggregator.patch_embed.patch_size

    # ------------------ new padding / unpadding helpers ------------------
    def _pad_images_to_multiple_of_patch_size(
        self, image_tensor: np.ndarray
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """
        Pad (not crop) a batch of images to be multiples of patch_size.
        image_tensor: [B, C, H, W]
        Returns padded images and padding tuple (pad_top, pad_bottom, pad_left, pad_right)
        """
        batch_size, channels, height, width = image_tensor.shape
        pad_height = (-height) % self.patch_size
        pad_width = (-width) % self.patch_size

        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        if pad_height == 0 and pad_width == 0:
            return image_tensor, (0, 0, 0, 0)

        padded = np.pad(
            image_tensor,
            ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode="reflect",
        )
        return padded, (pad_top, pad_bottom, pad_left, pad_right)

    # def _unpad_tensor(self, tensor: torch.Tensor, padding: tuple[int, int, int, int]) -> torch.Tensor:
    #     """
    #     Remove padding from a tensor with spatial dims at the end.
    #     Works for shapes like [..., H, W] or [..., C, H, W] etc.
    #     """
    #     pad_top, pad_bottom, pad_left, pad_right = padding

    #     # handle height
    #     if pad_top or pad_bottom:
    #         end_h = -pad_bottom if pad_bottom != 0 else None
    #         tensor = tensor[..., pad_top:end_h, :]

    #     # handle width
    #     if pad_left or pad_right:
    #         end_w = -pad_right if pad_right != 0 else None
    #         tensor = tensor[..., :, pad_left:end_w]

    #     return tensor

    # def _unpad_tensor(self,
    #                 tensor: torch.Tensor,
    #                 padding: tuple[int, int, int, int]) -> torch.Tensor:
    #     """
    #     Remove symmetric padding that was added by
    #     _pad_images_to_multiple_of_patch_size.
    #     Works for tensors whose *last* two dims are (H, W):
    #         [..., H, W]              e.g. [B, H, W]
    #         [..., C, H, W]           e.g. [B, C, H, W]
    #     """
    #     pad_top, pad_bottom, pad_left, pad_right = padding

    #     # Build slices once, apply in one shot – avoids axis-shift bugs
    #     h_slice = slice(pad_top, -pad_bottom if pad_bottom else None)
    #     w_slice = slice(pad_left, -pad_right if pad_right else None)

    #     return tensor[..., h_slice, w_slice]



    def _unpad_tensor(
            self,
            tensor: torch.Tensor,
            padding: tuple[int, int, int, int]) -> torch.Tensor:
        """
        Remove the symmetric padding we added earlier.

        Works for:
        • [B, N,  H,  W, C]   channel-last (VGGT depth, world_points)
        • [B, N, 3, H, W]     channel-first (images)
        • […,      H, W]      confidence maps, etc.
        """
        pad_top, pad_bottom, pad_left, pad_right = padding

        dims = tensor.ndim
        # Identify the two spatial axes:
        #   If last dim is 1 or 3 (small channel count) → assume [ …, H, W, C ]
        #   else                                   → assume [ …, H, W ]
        if tensor.shape[-1] in (1, 3):
            h_axis = -3
            w_axis = -2
        else:
            h_axis = -2
            w_axis = -1

        # Build a list of slices for every axis
        slicers = [slice(None)] * dims
        slicers[h_axis] = slice(pad_top,   -pad_bottom or None)
        slicers[w_axis] = slice(pad_left, -pad_right  or None)

        return tensor[tuple(slicers)]
            
    @typechecked
    def infer_original_depth(
        self,
        frames: np.ndarray | torch.Tensor,
        near: float = 1.0,
        far: float = 10000.0,
        num_denoising_steps: int = 50,
        guidance_scale: float = 6.0,
        window_size: int = 110,
        overlap: bool = False,
        use_original_scales: bool = False,
        enable_downscaling: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Full-sequence depth inference.
        Signatures unchanged from original VGGTDemo.infer.
        Returns depths tensor of shape [N,1,H,W] on self.device, so same size as input frames.

        Arguments:
            frames: numpy array or torch tensor of shape (N, C, H, W) or (N, H, W, C)
            near: near depth limit
            far: far depth limit
            num_denoising_steps: number of denoising steps for depth refinement
            guidance_scale: guidance scale for depth refinement
            window_size: window size for depth refinement
            overlap: whether to use overlapping windows for depth refinement
            use_original_scales: whether to use original scales for depth maps
            **kwargs: additional arguments passed to infer_batch
        Returns:
            depths: tensor of shape [N,1,H,W] with depth maps for each frame
        """
        # ensure numpy array of shape (N, C, H, W)
        assert frames.ndim == 4, "Input frames must be a 4D tensor or numpy array of shape (N, C, H, W) or (N, H, W, C)"
        if isinstance(frames, torch.Tensor):
            arr = frames.cpu().numpy()
        else:
            arr = frames
        # if channel last, convert to ensure (N, C, H, W) shape
        if arr.ndim == 4 and arr.shape[-1] in (1, 3):
            arr = np.transpose(arr, (0, 3, 1, 2))
        depths_parts = []
        outputs: dict[str, list[torch.Tensor]] = {}

        os.makedirs("vggt_outputs", exist_ok=True)

        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=self.dtype):
            for i in range(0, len(arr), self.batch_size):
                batch = arr[i : i + self.batch_size]
                if i != 0:
                    batch = np.concatenate([arr[0:1], batch], axis=0)

                # import pdb; pdb.set_trace()  # Debugging breakpoint
                preds = self.infer_batch(batch, use_cached=False, enable_downscaling=enable_downscaling, **kwargs)
                # accumulate outputs
                for key, val in preds.items():
                    if key.startswith("_"):
                        continue  # skip metadata
                    tensor = val.squeeze(0).cpu()
                    outputs.setdefault(key, []).append(tensor if i == 0 else tensor[1:])
                # depth postprocess (same as original)
                d = preds["depth"].squeeze(0)

                if use_original_scales == False:
                    inv = 1.0 / (d + 1e-6)
                    inv = (inv - inv.min()) / (inv.max() - inv.min()) * 3900
                    inv = 10000.0 / inv.clip(1e-5, None)
                    inv = inv.clip(near, far).squeeze(-1)
                else:
                    if d.ndim == 4:
                        inv = d.squeeze(1)
                    else:
                        inv = d
                depths_parts.append(inv if i == 0 else inv[1:])
                del preds
                torch.cuda.empty_cache()
                gc.collect()

        # save outputs
        for key, lst in outputs.items():
            torch.save(torch.cat(lst, 0), f"vggt_outputs/{key}.pt")

        # stack and return in original resolution (padding logic ensures depth is already correct size)
        depths = torch.cat(depths_parts, 0)  # [N,H,W, 1]
        depths = depths.unsqueeze(1).squeeze(-1)  # [N,1,H,W]
        return depths.to(self.device)

    @typechecked
    def infer_batch(
        self, frames: np.ndarray | list[np.ndarray], enable_downscaling: bool = False, use_cached: bool = False, **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Batch inference: same signature as original.
        frames: array or list of arrays of shape [C,H,W] or [H,W,C]
        Returns raw model outputs dict of tensors. 
        Warning: this downscales images to max_image_size, so check your outputs.

        Args:
            frames: numpy array or list of numpy arrays of shape (N, C, H, W) or (N, H, W, C)
            enable_downscaling: whether to downscale images to max_image_size
            use_cached: whether to use cached results (not implemented)
            **kwargs: additional arguments passed to the model
        Returns:
            output: dictionary with model outputs, including:
                - "depth": tensor of shape [1,N,1,H,W] with depth maps
                - "images": tensor of shape [1, N,3,H,W] with input images
                - "pose_enc": tensor of shape [1, N,4,4] with pose encodings
                - "world_points": tensor of shape [1, N,3,H,W] with world points
                - "world_points_conf": tensor of shape [1, N,H,W] with confidence scores for world points
        """
        # stack and preprocess
        imgs = []
        original_shape = frames[0].shape # (C, H, W) or (H, W, C)

        if original_shape[0] == 3:
            height_index = 1
            width_index = 2

        else:
            height_index = 0
            width_index = 1
        

        if original_shape[height_index] > self.max_image_size or original_shape[width_index] > self.max_image_size:
            if enable_downscaling == False:
                raise ValueError(
                    f"Original image shape {original_shape} is larger than max_image_size {self.max_image_size}, but downscaling is disabled. "
                    "Please enable downscaling with enable_downscaling=True or provide smaller images."
                )
        for img in frames:
            x = self._downscale_image_by_max_size(img, self.max_image_size)
            imgs.append(x)

        # Check if original images were downscaled
        if original_shape != imgs[0].transpose(1,2,0).shape:
            print(f"[DEBUG] Original image shape: {original_shape}, Downscaled shape: {imgs[0].shape}")
            print(f"This will cause the intrinsics/extrinsics to be according to new shape, not original.")

        imgs = np.stack(imgs)  # [B,3, H',W']
        imgs, padding = self._adjust_resolution_batch(imgs)
        tensor = torch.from_numpy(imgs).to(self.device)

        print(f"[DEBUG] Input tensor dtype before autocast: {tensor.dtype}")
        print(f"[DEBUG] Model param dtype: {next(self.model.parameters()).dtype}")
        print(f"D type: {self.dtype}, Device: {self.device}")

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                output = self.model(tensor)

        # import pdb; pdb.set_trace()  # Debugging breakpoint

        # unpad spatial outputs so they are back to original (pre-pad) resolution
        for spatial_key in ("depth", "world_points", "world_points_conf", "images"):
            if spatial_key in output:
                # if spatial_key == "depth" or spatial_key == "world_points": 
                    # pdb.set_trace()
                    # output[spatial_key] = output[spatial_key].permute(0, 1, 4, 2, 3) # [N, 1, H, W, C] -> [N, 1, C, H, W] 
                output[spatial_key] = self._unpad_tensor(output[spatial_key], padding)

                # if spatial_key == "depth" or spatial_key == "world_points":
                    # output[spatial_key] = output[spatial_key].permute(0, 1, 3, 4, 2) # [N, 1, C, H, W] -> [N, 1, H, W, C]


        # attach padding info so callers (e.g., camera pose) can adjust intrinsics
        output["_padding"] = torch.tensor(padding, device=self.device)
        return output

    @typechecked
    def infer_pointcloud(
        self, frames: np.ndarray | list[np.ndarray], confidence_threshold: float = 0.05,
        enable_downscaling: bool = False, 
         **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Same signature as original infer_pointcloud.
        Returns points and colors numpy arrays.
        Args:
            frames: list of numpy arrays of shape (H, W, C) or (C, H, W)
            confidence_threshold: threshold for point cloud confidence
            enable_downscaling: whether to downscale images to max_image_size
            **kwargs: additional arguments passed to infer_batch
        Returns:
            pts: numpy array of shape (N, 3) with 3D points
            rgb: numpy array of shape (N, 3) with RGB colors for each point
        """
        preds = self.infer_batch(frames, enable_downscaling=enable_downscaling, **kwargs)
        wp = preds["world_points"].squeeze(0)
        conf = preds["world_points_conf"].squeeze(0)
        img = preds["images"].squeeze(0).permute(0, 2, 3, 1)
        pts = wp.reshape(-1, 3)
        c = conf.reshape(-1)
        rgb = img.reshape(-1, 3)
        norm = (c - c.min()) / (c.max() - c.min())
        mask = norm > confidence_threshold
        return pts[mask].cpu().numpy(), rgb[mask].cpu().numpy()

    @typechecked
    def infer_camera_poses(
        self, frames: np.ndarray, use_cached: bool = False, 
        enable_downscaling: bool = False, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return c2w and intrinsics from model outputs.
        Args:
            frames: numpy array of shape (N, C, H, W) or (N, H, W, C)
            use_cached: whether to use cached results (not implemented)
            enable_downscaling: whether to downscale images to max_image_size
            **kwargs: additional arguments passed to infer_batch
        Returns:
            c2w: numpy array of shape (N, 4, 4) with camera-to-world matrices
            intr: numpy array of shape (N, 3, 3) with intrinsic matrices
        """
        preds = self.infer_batch(frames, use_cached=use_cached, enable_downscaling=enable_downscaling, **kwargs)
        padding_tensor = preds.get("_padding", torch.tensor((0, 0, 0, 0), device=self.device))
        pad_top, pad_bottom, pad_left, pad_right = tuple(padding_tensor.cpu().tolist())

        images = preds["images"]
        
        # compute padded size used during inference
        height_unpadded, width_unpadded = images.shape[3], images.shape[4]
        padded_height = height_unpadded + pad_top + pad_bottom
        padded_width = width_unpadded + pad_left + pad_right

        # get extrinsic/intrinsic using the padded resolution (what model actually saw)
        ex, intr = pose_encoding_to_extri_intri(preds["pose_enc"], image_size_hw=(padded_height, padded_width))
        B = ex.shape[1]
        pad = torch.tensor([[0, 0, 0, 1]], device=ex.device).repeat(1, B, 1, 1)
        ex = torch.cat([ex, pad], dim=2)
        c2w = torch.linalg.inv(ex)

        # adjust intrinsics principal point for removed padding
        intr_numpy = intr.squeeze().cpu().numpy()
        intr_numpy[..., 0, 2] -= pad_left
        intr_numpy[..., 1, 2] -= pad_top

        return c2w.squeeze().cpu().numpy(), intr_numpy

    @typechecked
    def infer_confidence_masks(
        self, 
        frames: np.ndarray | list[np.ndarray], 
        confidence_threshold: float = 0.1,
        enable_downscaling: bool = False,
        mask_type: str = "depth",  # Values: "depth", "points"
        use_original_shapes: bool = False,  # Whether to use original shapes for upscaling
        **kwargs
    ) -> torch.Tensor:
        """
        Extract confidence masks from VGGT inference.
        
        Args:
            frames: Input frames
            confidence_threshold: Threshold for converting confidence to binary masks. Drop confidence_threshold*100% values below this.
            enable_downscaling: Whether to downscale images
            mask_type: Type of mask to extract. Currently only "depth" is supported.
            use_original_shapes: Whether to use original shapes for upscaling masks
            
        Returns:
            confidence_masks: Binary masks of shape [N, 1, H, W] where 1 indicates high confidence
        """
        preds = self.infer_batch(frames, enable_downscaling=enable_downscaling, **kwargs)
        
        if f"{mask_type}_conf" in preds:
            conf = preds[f"{mask_type}_conf"].squeeze(0)  # Remove batch dim

            # Get quantile for confidence threshold
            confidence_quantile = np.quantile(conf.cpu().numpy(), confidence_threshold)
            print("Cutting off confidence at quantile:", confidence_quantile)
            print(f"Value is {confidence_threshold*100}% of all confidence values and it's <{confidence_quantile}>.")

            # Normalize confidence to [0, 1]
            # conf_normalized = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)

            # Create binary masks
            # confidence_masks = (conf_normalized > confidence_threshold).float()
            confidence_masks = (conf > confidence_quantile).float()  # Binary mask based on quantile
            # Add channel dimension: [N, H, W] -> [N, 1, H, W]
            confidence_masks = confidence_masks.unsqueeze(1)
            
            
            # Upscale to original image size if needed

            if use_original_shapes:
                original_shape = frames[0].shape if isinstance(frames, list) else frames.shape[1:]
                if confidence_masks.shape[-2:] != original_shape[:2]:

                    confidence_masks_upscaled = []

                    for mask in confidence_masks:
                        upscaled_mask = self._upscale_image_to_original_size(
                            mask, 
                            (original_shape[0], original_shape[1])
                        )
                        confidence_masks_upscaled.append(upscaled_mask)

                    confidence_masks = torch.stack(confidence_masks_upscaled, dim=0)
                    confidence_masks = confidence_masks > 0.0 # Convert to binary mask
                
            return confidence_masks.to(self.device)
        else:
            # Fallback: return all-ones masks
            raise NotImplementedError(
                f"Mask type '{mask_type}' not supported. Currently only 'depth' and 'points' is implemented."
            )
            # n_frames = len(frames) if isinstance(frames, list) else frames.shape[0]
            # original_shape = frames[0].shape if isinstance(frames, list) else frames.shape[1:]
            # return torch.ones(n_frames, 1, original_shape[0], original_shape[1], device=self.device)
        
    @typechecked
    def _run_vggt_sfm(
        self,
        source_path: str,
        model_path: str,
        device: str = "cuda",
        image_size: int = 512,
        llffhold: int = 8,
        n_views: int | None = None,
        co_vis_dsp: bool = False,
        depth_thre: float = 0.01,
        conf_aware_ranking: bool = True,
        focal_avg: bool = False,
        split_train_test_function: bool = False,
    ) -> str:
        """

        Run VGGT on a directory of images and saves data in COLMAP and npy format.
        Args:
            source_path: path to the directory containing images
            model_path: path to save the output
            device: device to run the model on
            image_size: size of the images to be processed
            llffhold: number of images to hold out for testing
            n_views: number of views to process
            co_vis_dsp: whether to compute co-visibility masks
            depth_thre: depth threshold for co-visibility masks
            conf_aware_ranking: whether to use confidence-aware ranking
            focal_avg: whether to average the focal length
            split_train_test_function: whether to split the dataset into train and test sets
        Returns:
            model_path: path to the saved output"""
        dev = torch.device(device)
        os.makedirs(model_path, exist_ok=True)
        model = self.model
        img_dir = Path(source_path) / "images"
        files, suffix = get_sorted_image_files(img_dir)
        if n_views is None:
            n_views = len(files)
        save_base, sparse_0, sparse_1 = init_filestructure(Path(source_path), n_views)
        if not split_train_test_function:
            train_files = files
        else:
            train_files, _ = split_train_test(files, llffhold, n_views, verbose=True)
        imgs = load_and_preprocess_images([str(f) for f in train_files]).to(dev)
        H, W = imgs.shape[-2], imgs.shape[-1]
        new_H = (H // self.patch_size) * self.patch_size
        new_W = (W // self.patch_size) * self.patch_size
        if new_H != H or new_W != W:
            imgs = imgs[:, :, :new_H, :new_W]
        start = time()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16 if dev.type == "cuda" else torch.float32):
            preds = model(imgs)
        for k in preds:
            preds[k] = preds[k].squeeze(0)
        # extrinsics/intrinsics
        if "extrinsic" not in preds:
            exs, ins = [], []
            for pe in preds["pose_enc"]:
                e, i = pose_encoding_to_extri_intri(pe[None], image_size_hw=(new_H, new_W))
                exs.append(e)
                ins.append(i[:, 0])
            preds["extrinsic"] = torch.cat(exs, 0)
            preds["intrinsic"] = torch.cat(ins, 0)
        extrinsics_np = preds["extrinsic"].cpu().numpy()
        intrinsics_np = preds["intrinsic"].cpu().numpy()
        # points3d
        wp = preds["world_points"].cpu().numpy()
        cf = preds["world_points_conf"].cpu().numpy()
        mask = cf > np.quantile(cf, 0.0)
        pts3d = np.zeros_like(wp)
        pts3d[mask] = wp[mask]
        # depthmaps
        if "depth" not in preds:
            raise ValueError("No depth maps found in VGGT output")
        depthmaps = preds["depth"].cpu().numpy()
        # co-visibility masks
        if conf_aware_ranking:
            avg_conf = cf.mean(axis=(1, 2))
            idxs = np.argsort(avg_conf)[::-1]
            overlaps = (
                compute_co_vis_masks(
                    idxs,
                    depthmaps,
                    pts3d,
                    intrinsics_np,
                    extrinsics_np,
                    imgs.permute(0, 2, 3, 1).shape,
                    depth_threshold=depth_thre,
                )
                if depth_thre > 0
                else None
            )
        save_time(model_path, "[1] coarse_init", time() - start)
        save_extrinsic(sparse_0, extrinsics_np, train_files, suffix)
        save_intrinsics(
            sparse_0,
            np.array([i[0, 0] for i in intrinsics_np]),
            (W, H),
            imgs.permute(0, 2, 3, 1).shape,
            save_focals=True,
        )
        save_points3D(
            sparse_0,
            imgs.permute(0, 2, 3, 1).cpu().numpy(),
            pts3d,
            cf.reshape(len(train_files), -1),
            overlaps,
            use_masks=co_vis_dsp,
            save_all_pts=True,
            save_txt_path=model_path,
            depth_threshold=depth_thre,
        )
        save_images_and_masks(
            sparse_0,
            len(train_files),
            imgs.permute(0, 2, 3, 1).cpu().numpy(),
            overlaps,
            train_files,
            suffix,
        )
        shutil.copy(Path(sparse_0) / "points3D.ply", Path(model_path) / "points3D.ply")
        np.save(os.path.join(model_path, "pts3d.npy"), pts3d[mask])
        colors = imgs.permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()[mask.reshape(-1)]
        np.save(os.path.join(model_path, "points3d_colors.npy"), colors)
        np.save(os.path.join(model_path, "extrinsics.npy"), extrinsics_np)
        np.save(os.path.join(model_path, "intrinsics.npy"), intrinsics_np)
        np.save(os.path.join(model_path, "depthmaps.npy"), depthmaps)
        np.save(
            os.path.join(model_path, "images.npy"),
            imgs.permute(0, 2, 3, 1).cpu().numpy(),
        )
        print(f"[✔] SFM init done. arrays saved to {model_path}")
        return model_path

    @typechecked
    def process_images_directory(
        self,
        source_path: str,
        output_path: str | None = None,
        image_size: int = 512,
        n_views: int | None = None,
        co_vis_dsp: bool = False,
        depth_thre: float = 0.01,
        focal_avg: bool = False,
        llffhold: int = 8,
        split_train_test: bool = False,
    ) -> dict[str, str | np.ndarray]:
        """
        Process a directory of images and save the results.
        Calls _run_vggt_sfm with the specified parameters.


        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source path {source_path} does not exist.")
        if output_path is None:
            output_path = os.path.join(os.path.dirname(source_path), "vggt_output")
        model_path = self._run_vggt_sfm(
            source_path=source_path,
            model_path=output_path,
            device=str(self.device),
            image_size=image_size,
            llffhold=llffhold,
            n_views=n_views,
            co_vis_dsp=co_vis_dsp,
            depth_thre=depth_thre,
            conf_aware_ranking=True,
            focal_avg=focal_avg,
            split_train_test_function=split_train_test,
        )
        return {
            "output_path": model_path,
            "extrinsics": np.load(os.path.join(model_path, "extrinsics.npy")),
            "intrinsics": np.load(os.path.join(model_path, "intrinsics.npy")),
            "points3d": np.load(os.path.join(model_path, "pts3d.npy")),
            "points3d_colors": np.load(os.path.join(model_path, "points3d_colors.npy")),
            "depthmaps": np.load(os.path.join(model_path, "depthmaps.npy")),
            "images_tensor": np.load(os.path.join(model_path, "images.npy")),
        }

    def load_results(self, model_path: str) -> dict[str, np.ndarray]:
        """
        Same signature as original load_results.
        """
        return {
            "extrinsics": np.load(os.path.join(model_path, "extrinsics.npy")),
            "intrinsics": np.load(os.path.join(model_path, "intrinsics.npy")),
            "points3d": np.load(os.path.join(model_path, "pts3d.npy")),
            "points3d_colors": np.load(os.path.join(model_path, "points3d_colors.npy")),
            "depthmaps": np.load(os.path.join(model_path, "depthmaps.npy")),
            "images_tensor": np.load(os.path.join(model_path, "images.npy")),
        }

    # Private helpers
    def _adjust_resolution_batch(self, imgs: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        B, C, H, W = imgs.shape
        # PAD to nearest multiple of patch_size instead of cropping
        padded_imgs, padding = self._pad_images_to_multiple_of_patch_size(imgs)

        if padding != (0, 0, 0, 0):
            print(f"Adjusting resolution from ({H},{W}) to ({padded_imgs.shape[2]},{padded_imgs.shape[3]}) with padding {padding}")
        return padded_imgs, padding

    @typechecked
    def _downscale_image_by_max_size(self, image: np.ndarray, max_size: int) -> np.ndarray:
        # handle (C,H,W) or (H,W,C)
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        h, w = image.shape[:2]

        try:
            if h <= max_size and w <= max_size:
                print(f"Image shape ({h},{w}) already smaller than max size {max_size}, skipping downscale.")
                return image.transpose(2, 0, 1)
            
        except Exception as e:
            print(f"Error checking image size: {e}")
            pdb.set_trace()  # Debugging breakpoint
            raise

        
        if h > w:
            new_h = max_size
            new_w = int(w * max_size / h)
        else:
            new_w = max_size
            new_h = int(h * max_size / w)

        if new_h == h and new_w == w:
            print(f"Image already at max size {max_size}, skipping downscale.")
            resized = image
        else:
            print(f"Downscaling image from ({h},{w}) to ({new_h},{new_w})")
            resized = cv2.resize(image, (new_w, new_h))

        return resized.transpose(2, 0, 1)

    @typechecked
    def _upscale_image_to_original_size(self, img: torch.Tensor, original_size: tuple[int, int]) -> torch.Tensor:
        arr = img.detach().cpu().numpy()
        # img is [1,H,W]
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        H, W = original_size
        up = cv2.resize(arr, (W, H))
        if up.ndim == 2:
            up = up[..., None]
        out = np.transpose(up, (2, 0, 1))
        return torch.from_numpy(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--n_views", type=int, default=None)
    parser.add_argument("--co_vis_dsp", action="store_true")
    parser.add_argument("--depth_thre", type=float, default=0.01)
    parser.add_argument("--focal_avg", action="store_true")
    parser.add_argument("--split_train_test", action="store_true")
    args = parser.parse_args()

    interface = VGGTWrapper(device=args.device)
    interface.process_images_directory(
        source_path=args.source_path,
        output_path=args.output_path,
        image_size=args.image_size,
        llffhold=args.llffhold,
        n_views=args.n_views,
        co_vis_dsp=args.co_vis_dsp,
        depth_thre=args.depth_thre,
        focal_avg=args.focal_avg,
        split_train_test=args.split_train_test,
    )
    print("[✔] VGGT full pipeline complete. Outputs at", args.output_path)
