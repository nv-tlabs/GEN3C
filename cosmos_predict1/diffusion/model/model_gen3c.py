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

from typing import Optional

import torch
from megatron.core import parallel_state
from torch import Tensor
from tqdm import tqdm
from cosmos_predict1.diffusion.conditioner import VideoExtendCondition
from cosmos_predict1.diffusion.model.model_v2w import DiffusionV2WModel, broadcast_condition
import torch.nn.functional as F


class DiffusionGen3CModel(DiffusionV2WModel):
    def __init__(self, config):
        super().__init__(config)
        self.frame_buffer_max = config.frame_buffer_max
        self.chunk_size = 121

    # def encode_warped_frames(
    #     self, 
    #     condition_state: torch.Tensor, 
    #     condition_state_mask: torch.Tensor, 
    #     dtype: torch.dtype,
    #     ):
    #     """
    #     Encode the warped frames and their masks into latent space.
    #     The left out space is filled with zeros and is defined by the frame_buffer_max, which is 2 by default.
    #     """

    #     assert condition_state.dim() == 6
    #     condition_state_mask = (condition_state_mask * 2 - 1).repeat(1, 1, 1, 3, 1, 1)
    #     latent_condition = []

    #     import pdb; pdb.set_trace()
    #     for i in tqdm(range(condition_state.shape[2]), desc="Encoding warped frames"):
    #         current_video_latent = self.encode(
    #             condition_state[:, :, i].permute(0, 2, 1, 3, 4).to(dtype)
    #         ).contiguous()  # 1, 16, 8, 88, 160

    #         current_mask_latent = self.encode(
    #             condition_state_mask[:, :, i].permute(0, 2, 1, 3, 4).to(dtype)
    #         ).contiguous()
    #         latent_condition.append(current_video_latent)
    #         latent_condition.append(current_mask_latent)
    #     for _ in range(self.frame_buffer_max - condition_state.shape[2]):
    #         latent_condition.append(torch.zeros_like(current_video_latent))
    #         latent_condition.append(torch.zeros_like(current_mask_latent))

    #     latent_condition = torch.cat(latent_condition, dim=1)
    #     return latent_condition



    def encode_warped_frames(
        self, 
        condition_state: torch.Tensor, # shape of (B, T, V, 3, H, W)
        condition_state_mask: torch.Tensor,  # shape of (B, T, V, 3, H, W)
        dtype: torch.dtype,
        ):
        """
        Encode the warped frames and their masks into latent space.
        The left out space is filled with zeros and is defined by the frame_buffer_max, which is 2 by default.

        From deleted Gen3C paper explanation:
        “For each pixel we compute the ℓ₂ norm of the 16-d latent vector coming from each warp buffer and keep 
        the entire vector from the buffer with the largest norm. Empirically this ‘vector-max’ selection 
        gives slightly sharper results than per-channel max pooling or simple averaging.”

        """

        assert condition_state.dim() == 6
        condition_state_mask = (condition_state_mask * 2 - 1).repeat(1, 1, 1, 3, 1, 1)
        latent_condition_video = []
        latent_condition_mask = []

        B, T, V, C_in, H, W = condition_state.shape

        for i in tqdm(range(condition_state.shape[2]), desc="Encoding warped frames"):

            condition_state_B_C_T_H_W = condition_state[:, :, i].permute(0, 2, 1, 3, 4).to(dtype)  # (B, V, C, T, H, W)
            condition_state_mask_B_C_T_H_W = condition_state_mask[:, :, i].permute(0, 2, 1, 3, 4).to(dtype)  # (B, V, C, T, H, W)
            
            current_video_latent_B_Cl_Tl_Hl_Wl = self.encode(
                condition_state_B_C_T_H_W
            ).contiguous()  # 1, 16, 8, 88, 160

            current_mask_latent_B_Cl_Tl_Hl_Wl = self.encode(
                condition_state_mask_B_C_T_H_W
            ).contiguous()

            latent_condition_video.append(current_video_latent_B_Cl_Tl_Hl_Wl)
            latent_condition_mask.append(current_mask_latent_B_Cl_Tl_Hl_Wl)


        latent_video_B_V_Cl_Tl_Hl_Wl = torch.stack(latent_condition_video, dim=1)   # (B, V, Cl, Tl, Hl, Wl)
        latent_mask_B_V_Cl_Tl_Hl_Wl = torch.stack(latent_condition_mask, dim=1)     # (B, V, Cl, Tl, Hl, Wl)

        B, V, Cl, Tl, Hl, Wl = latent_video_B_V_Cl_Tl_Hl_Wl.shape
        
        # Get the feature norm for each view and time step
        feature_norm_B_V_Tl_Hl_Wl = latent_video_B_V_Cl_Tl_Hl_Wl.norm(dim=2)  # (B, V, Tl, Hl, Wl)

        # Find the view with the maximum feature norm for each pixel
        winner_view_B_Tl_Hl_Wl = feature_norm_B_V_Tl_Hl_Wl.argmax(dim=1, keepdim=True)   # (B, V, Tl, Hl, Wl) -> (B, 1, Tl, Hl, Wl)

        view_index = winner_view_B_Tl_Hl_Wl.unsqueeze(2).expand(-1, -1, latent_video_B_V_Cl_Tl_Hl_Wl.shape[2], -1, -1, -1)  # (B, 1, Cl, Tl, Hl, Wl)


        pooled_latent_B_1_Cl_Tl_Hl_Wl = latent_video_B_V_Cl_Tl_Hl_Wl.gather(dim=1, index=view_index)  # Gather the features based on the winner view index
        pooled_latent_B_Cl_Tl_Hl_Wl = pooled_latent_B_1_Cl_Tl_Hl_Wl.squeeze(1)  # Remove the view dimension

        pooled_mask_B_1_Cl_Tl_Hl_Wl = latent_mask_B_V_Cl_Tl_Hl_Wl.gather(dim=1, index=view_index)  # Gather the mask based on the winner view index
        pooled_mask_B_Cl_Tl_Hl_Wl = pooled_mask_B_1_Cl_Tl_Hl_Wl.squeeze(1)  # Remove the view dimension


        assert pooled_latent_B_Cl_Tl_Hl_Wl.shape == current_video_latent_B_Cl_Tl_Hl_Wl.shape, \
            f"Shape pooled latent mismatch: {pooled_latent_B_Cl_Tl_Hl_Wl.shape} vs {current_video_latent_B_Cl_Tl_Hl_Wl.shape}"    
        
        assert pooled_mask_B_Cl_Tl_Hl_Wl.shape == current_mask_latent_B_Cl_Tl_Hl_Wl.shape, \
            f"Shape pooled mask mismatch: {pooled_mask_B_Cl_Tl_Hl_Wl.shape} vs {current_mask_latent_B_Cl_Tl_Hl_Wl.shape}"

        latent_condition = []
        latent_condition.append(pooled_latent_B_Cl_Tl_Hl_Wl)  # Add the max values as the video latent
        latent_condition.append(pooled_mask_B_Cl_Tl_Hl_Wl)    # Add the selected masks


        for _ in range(self.frame_buffer_max - 1):
            latent_condition.append(torch.zeros_like(current_video_latent_B_Cl_Tl_Hl_Wl))
            latent_condition.append(torch.zeros_like(current_mask_latent_B_Cl_Tl_Hl_Wl))

        L = 2 * self.frame_buffer_max * Cl  # Cl is the number of channels in the latent space
        latent_condition_B_L_Tl_Hl_Wl = torch.cat(latent_condition, dim=1)

        return latent_condition_B_L_Tl_Hl_Wl # torch.Size([1, 64, 16, 88, 160]) # [B, L, Tl, Hl, Wl] where L = 2 * frame_buffer_max * Cl




    # def encode_warped_frames(
    #     self,
    #     condition_state: torch.Tensor,          # (B, T, V, 3, H, W)
    #     condition_state_mask: torch.Tensor,     # (B, T, V, 3, H, W)
    #     dtype: torch.dtype,
    #     *,
    #     spatial_kernel_size: int = 3            # 3 × 3 spatial window
    # ) -> torch.Tensor:
    #     """
    #     1. Encode each of the V warped views → latent (Cl,Tl,Hl,Wl)
    #     2. Fuse views with max over V (no spatial pooling yet)
    #     3. **Spatial** max-pool on every Tl slice (stride 1, pad = k//2)
    #     4. Pack into (B, 2*frame_buffer_max*Cl, Tl, Hl, Wl)
    #     """

    #     # --- preprocessing -------------------------------------------------
    #     assert condition_state.dim() == 6, "expect (B,T,V,3,H,W)"
    #     condition_state_mask = (condition_state_mask * 2 - 1).repeat(1, 1, 1, 3, 1, 1)

    #     latent_video_per_view = []
    #     latent_mask_per_view  = []

    #     B, T, V, _, H, W = condition_state.shape

    #     # --- 1) encode each view separately --------------------------------
    #     for v in tqdm(range(V), desc="Encoding warped frames"):
    #         rgb_B_C_T_H_W   = condition_state[:, :, v].permute(0, 2, 1, 3, 4).to(dtype)
    #         mask_B_C_T_H_W  = condition_state_mask[:, :, v].permute(0, 2, 1, 3, 4).to(dtype)

    #         latent_rgb   = self.encode(rgb_B_C_T_H_W).contiguous()   # (B,Cl,Tl,Hl,Wl)
    #         latent_mask  = self.encode(mask_B_C_T_H_W).contiguous()

    #         latent_video_per_view.append(latent_rgb)
    #         latent_mask_per_view.append(latent_mask)

    #     # stack → (B,V,Cl,Tl,Hl,Wl)
    #     latent_rgb_B_V_Cl_Tl_Hl_Wl  = torch.stack(latent_video_per_view, dim=1)
    #     latent_mask_B_V_Cl_Tl_Hl_Wl = torch.stack(latent_mask_per_view,  dim=1)

    #     # --- 2) fuse views with max over V ---------------------------------
    #     fused_rgb_B_Cl_Tl_Hl_Wl  = torch.amax(latent_rgb_B_V_Cl_Tl_Hl_Wl,  dim=1)  # drop V
    #     fused_mask_B_Cl_Tl_Hl_Wl = torch.amax(latent_mask_B_V_Cl_Tl_Hl_Wl, dim=1)

    #     # --- 3) **spatial** max-pool on each (Tl) frame --------------------
    #     #      reshape so each Tl slice is processed independently
    #     B, Cl, Tl, Hl, Wl = fused_rgb_B_Cl_Tl_Hl_Wl.shape
    #     fused_rgb_flat  = fused_rgb_B_Cl_Tl_Hl_Wl.permute(0, 2, 1, 3, 4).reshape(B * Tl, Cl, Hl, Wl)
    #     fused_mask_flat = fused_mask_B_Cl_Tl_Hl_Wl.permute(0, 2, 1, 3, 4).reshape(B * Tl, Cl, Hl, Wl)

    #     pad = spatial_kernel_size // 2
    #     pooled_rgb_flat  = F.max_pool2d(fused_rgb_flat,  kernel_size=spatial_kernel_size,
    #                                     stride=1, padding=pad)
    #     pooled_mask_flat = F.max_pool2d(fused_mask_flat, kernel_size=spatial_kernel_size,
    #                                     stride=1, padding=pad)

    #     # restore (B,Cl,Tl,Hl,Wl)
    #     pooled_rgb_B_Cl_Tl_Hl_Wl  = pooled_rgb_flat.view(B, Tl, Cl, Hl, Wl).permute(0, 2, 1, 3, 4)
    #     pooled_mask_B_Cl_Tl_Hl_Wl = pooled_mask_flat.view(B, Tl, Cl, Hl, Wl).permute(0, 2, 1, 3, 4)

    #     # --- 4) pack for the diffusion U-Net -------------------------------
    #     latent_condition = [pooled_rgb_B_Cl_Tl_Hl_Wl, pooled_mask_B_Cl_Tl_Hl_Wl]

    #     for _ in range(self.frame_buffer_max - 1):
    #         latent_condition += [
    #             torch.zeros_like(pooled_rgb_B_Cl_Tl_Hl_Wl),
    #             torch.zeros_like(pooled_mask_B_Cl_Tl_Hl_Wl),
    #         ]

    #     L = 2 * self.frame_buffer_max * Cl
    #     return torch.cat(latent_condition, dim=1)                     # (B,L,Tl,Hl,Wl)




    def _get_conditions(
        self,
        data_batch: dict,
        is_negative_prompt: bool = False,
        condition_latent: Optional[torch.Tensor] = None,
        num_condition_t: Optional[int] = None,
        add_input_frames_guidance: bool = False,
    ):
        """Get the conditions for the model.

        Args:
            data_batch: Input data dictionary
            is_negative_prompt: Whether to use negative prompting
            condition_latent: Conditioning frames tensor (B,C,T,H,W)
            num_condition_t: Number of frames to condition on
            add_input_frames_guidance: Whether to apply guidance to input frames

        Returns:
            condition: Input conditions
            uncondition: Conditions removed/reduced to minimum (unconditioned)
        """
        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        # encode warped frames
        condition_state, condition_state_mask = (
            data_batch["condition_state"],
            data_batch["condition_state_mask"],
        )
        latent_condition = self.encode_warped_frames(
            condition_state, condition_state_mask, self.tensor_kwargs["dtype"]
        )

        condition.video_cond_bool = True

        # Adding input image/video as gt_latent into condition : VideoExtendCondition object
        # This is not warped frames, but the original input frames
        condition = self.add_condition_video_indicator_and_video_input_mask(
            condition_latent, condition, num_condition_t 
        )
        # Here we add warped latent frames as condition
        condition = self.add_condition_pose(latent_condition, condition)

        uncondition.video_cond_bool = False if add_input_frames_guidance else True
        uncondition = self.add_condition_video_indicator_and_video_input_mask(
            condition_latent, uncondition, num_condition_t
        )
        # uncodition goes without warped frames 
        uncondition = self.add_condition_pose(latent_condition, uncondition, drop_out_latent = True)
        assert condition.gt_latent.allclose(uncondition.gt_latent)

        # For inference, check if parallel_state is initialized
        to_cp = self.net.is_context_parallel_enabled
        if parallel_state.is_initialized():
            condition = broadcast_condition(condition, to_tp=False, to_cp=to_cp)
            uncondition = broadcast_condition(uncondition, to_tp=False, to_cp=to_cp)

        return condition, uncondition

    def add_condition_pose(self, latent_condition: torch.Tensor, condition: VideoExtendCondition,
                           drop_out_latent: bool = False) -> VideoExtendCondition:
        """Add pose condition to the condition object. For camera control model
        Args:
            data_batch (Dict): data batch, with key "plucker_embeddings", in shape B,T,C,H,W
            latent_state (torch.Tensor): latent state tensor in shape B,C,T,H,W
            condition (VideoExtendCondition): condition object
            num_condition_t (int): number of condition latent T, used in inference to decide the condition region and config.conditioner.video_cond_bool.condition_location == "first_n"
        Returns:
            VideoExtendCondition: updated condition object
        """
        if drop_out_latent:
            condition.condition_video_pose = torch.zeros_like(latent_condition.contiguous())
        else:
            condition.condition_video_pose = latent_condition.contiguous()

        to_cp = self.net.is_context_parallel_enabled

        # For inference, check if parallel_state is initialized
        if parallel_state.is_initialized():
            condition = broadcast_condition(condition, to_tp=True, to_cp=to_cp)
        else:
            assert not to_cp, "parallel_state is not initialized, context parallel should be turned off."

        return condition
