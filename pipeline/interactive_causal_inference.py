# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
from pipeline.causal_inference import CausalInferencePipeline
import torch.distributed as dist
from utils.debug_option import DEBUG
from .latency_logger import LatencyLogger


class InteractiveCausalInferencePipeline(CausalInferencePipeline):
    def __init__(
        self,
        args,
        device,
        *,
        generator: WanDiffusionWrapper | None = None,
        text_encoder: WanTextEncoder | None = None,
        vae: WanVAEWrapper | None = None,
        latency_logger: LatencyLogger | None = None,
    ):
        super().__init__(args, device, generator=generator, text_encoder=text_encoder, vae=vae)
        self.global_sink = getattr(args, "global_sink", False)
        # Override with provided logger or use the parent's logger
        if latency_logger is not None:
            self.latency = latency_logger

    # Internal helpers
    def _recache_after_switch(self, output, current_start_frame, new_conditional_dict):
        with self.latency.batch_timer("recache_total"):
            with self.latency.batch_timer("recache_reset_cache"):
                if not self.global_sink:
                    # reset kv cache
                    for block_idx in range(self.num_transformer_blocks):
                        cache = self.kv_cache1[block_idx]
                        cache["k"].zero_()
                        cache["v"].zero_()
                        # cache["global_end_index"].zero_()
                        # cache["local_end_index"].zero_()
                
                # reset cross-attention cache
                for blk in self.crossattn_cache:
                    blk["k"].zero_()
                    blk["v"].zero_()
                    blk["is_init"] = False

            # recache
            if current_start_frame == 0:
                return
            
            with self.latency.batch_timer("recache_prepare"):
                num_recache_frames = current_start_frame if self.local_attn_size == -1 else min(self.local_attn_size, current_start_frame)
                recache_start_frame = current_start_frame - num_recache_frames
                
                frames_to_recache = output[:, recache_start_frame:current_start_frame]
                
                # move to gpu if needed
                if frames_to_recache.device.type == 'cpu':
                    with self.latency.batch_timer("recache_h2d"):
                        target_device = next(self.generator.parameters()).device
                        frames_to_recache = frames_to_recache.to(target_device)
                
                batch_size = frames_to_recache.shape[0]
                print(f"num_recache_frames: {num_recache_frames}, recache_start_frame: {recache_start_frame}, current_start_frame: {current_start_frame}")
                
                # prepare blockwise causal mask
                device = frames_to_recache.device
                block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
                    device=device,
                    num_frames=num_recache_frames,
                    frame_seqlen=self.frame_seq_length,
                    num_frame_per_block=self.num_frame_per_block,
                    local_attn_size=self.local_attn_size
                )
                
                context_timestep = torch.ones([batch_size, num_recache_frames], 
                                            device=device, dtype=torch.int64) * self.args.context_noise
                
                self.generator.model.block_mask = block_mask
            
            # recache forward pass
            with self.latency.batch_timer("recache_forward"):
                with torch.no_grad():
                    self.generator(
                        noisy_image_or_video=frames_to_recache,
                        conditional_dict=new_conditional_dict,
                        timestep=context_timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=recache_start_frame * self.frame_seq_length,
                    )
            
            # reset cross-attention cache again
            with self.latency.batch_timer("recache_reset_crossattn_final"):
                for blk in self.crossattn_cache:
                    blk["k"].zero_()
                    blk["v"].zero_()
                    blk["is_init"] = False

    def inference(
        self,
        noise: torch.Tensor,
        *,
        text_prompts_list: List[List[str]],
        switch_frame_indices: List[int],
        return_latents: bool = False,
        low_memory: bool = False,
    ):
        """Generate a video and switch prompts at specified frame indices.

        Args:
            noise: Noise tensor, shape = (B, T_out, C, H, W).
            text_prompts_list: List[List[str]], length = N_seg. Prompt list used for segment i (aligned with batch).
            switch_frame_indices: List[int], length = N_seg - 1. The i-th value indicates that when generation reaches this frame (inclusive)
                we start using the prompts for segment i+1.
            return_latents: Whether to also return the latent tensor.
            low_memory: Enable low-memory mode.
        """
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert len(text_prompts_list) >= 1, "text_prompts_list must not be empty"
        assert len(switch_frame_indices) == len(text_prompts_list) - 1, (
            "length of switch_frame_indices should be one less than text_prompts_list"
        )
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        # Start batch-level timing with comprehensive metadata
        with self.latency.batch_scope(
            batch_idx=0,
            batch_size=batch_size,
            num_output_frames=num_output_frames,
            num_frame_per_block=self.num_frame_per_block,
            local_attn_size=self.local_attn_size,
            denoising_steps=len(self.denoising_step_list),
            low_memory=low_memory,
            num_prompt_segments=len(text_prompts_list),
            switch_frame_indices=switch_frame_indices,
        ):
            with self.latency.batch_timer("batch_total"):
                # Text encoding stage - encode all prompts
                with self.latency.batch_timer("text_encode_all_prompts"):
                    print(text_prompts_list)
                    cond_list = [self.text_encoder(text_prompts=p) for p in text_prompts_list]

                if low_memory:
                    with self.latency.batch_timer("text_encoder_offload"):
                        gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
                        move_model_to_device_with_memory_preservation(
                            self.text_encoder,
                            target_device=gpu,
                            preserved_memory_gb=gpu_memory_preservation,
                        )

                output_device = torch.device('cpu') if low_memory else noise.device
                output = torch.zeros(
                    [batch_size, num_output_frames, num_channels, height, width],
                    device=output_device,
                    dtype=noise.dtype
                )

                # Initialize caches
                with self.latency.batch_timer("kv_init"):
                    local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
                    kv_policy = ""
                    if local_attn_cfg != -1:
                        # local attention
                        kv_cache_size = local_attn_cfg * self.frame_seq_length
                        kv_policy = f"int->local, size={local_attn_cfg}"
                    else:
                        # global attention
                        kv_cache_size = num_output_frames * self.frame_seq_length
                        kv_policy = "global (-1)"
                    print(f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})")

                    self._initialize_kv_cache(
                        batch_size,
                        dtype=noise.dtype,
                        device=noise.device,
                        kv_cache_size_override=kv_cache_size
                    )
                    self._initialize_crossattn_cache(
                        batch_size=batch_size,
                        dtype=noise.dtype,
                        device=noise.device
                    )

                current_start_frame = 0
                self.generator.model.local_attn_size = self.local_attn_size
                print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
                self._set_all_modules_max_attention_size(self.local_attn_size)

                # temporal denoising by blocks
                all_num_frames = [self.num_frame_per_block] * num_blocks
                segment_idx = 0  # current segment index
                next_switch_pos = (
                    switch_frame_indices[segment_idx]
                    if segment_idx < len(switch_frame_indices)
                    else None
                )

                if DEBUG:
                    print("[MultipleSwitch] all_num_frames", all_num_frames)
                    print("[MultipleSwitch] switch_frame_indices", switch_frame_indices)

                for block_idx, current_num_frames in enumerate(all_num_frames):
                    # Check for prompt switch
                    if next_switch_pos is not None and current_start_frame >= next_switch_pos:
                        old_segment = segment_idx
                        segment_idx += 1
                        
                        # Mark the prompt switch event
                        self.latency.mark_prompt_switch(
                            frame_idx=current_start_frame,
                            old_prompt=text_prompts_list[old_segment][0] if text_prompts_list[old_segment] else "",
                            new_prompt=text_prompts_list[segment_idx][0] if text_prompts_list[segment_idx] else ""
                        )
                        
                        # Time the recache operation
                        with self.latency.batch_timer("prompt_switch_recache"):
                            self._recache_after_switch(output, current_start_frame, cond_list[segment_idx])
                        
                        if DEBUG:
                            print(
                                f"[MultipleSwitch] Switch to segment {segment_idx} at frame {current_start_frame}"
                            )
                        next_switch_pos = (
                            switch_frame_indices[segment_idx]
                            if segment_idx < len(switch_frame_indices)
                            else None
                        )
                        print(f"segment_idx: {segment_idx}")
                        print(f"text_prompts_list[segment_idx]: {text_prompts_list[segment_idx]}")
                    
                    cond_in_use = cond_list[segment_idx]

                    noisy_input = noise[
                        :, current_start_frame : current_start_frame + current_num_frames
                    ]

                    # Start frame timing for each frame in this block
                    for frame_offset in range(current_num_frames):
                        frame_idx = current_start_frame + frame_offset
                        self.latency.start_frame(
                            frame_idx,
                            prompt=text_prompts_list[segment_idx][0] if text_prompts_list[segment_idx] else "",
                            block_idx=block_idx,
                            segment_idx=segment_idx,
                        )

                    # ---------------- Spatial denoising loop ----------------
                    with self.latency.timer("denoise_loop_total"):
                        for index, current_timestep in enumerate(self.denoising_step_list):
                            with self.latency.step_timer(index) as step:
                                # Prepare timestep
                                with step.timer("prepare_timestep"):
                                    timestep = (
                                        torch.ones([batch_size, current_num_frames],
                                        device=noise.device,
                                        dtype=torch.int64)
                                        * current_timestep
                                    )

                                if index < len(self.denoising_step_list) - 1:
                                    # Forward pass with KV cache operations
                                    with step.timer("denoise_forward"):
                                        _, denoised_pred = self.generator(
                                            noisy_image_or_video=noisy_input,
                                            conditional_dict=cond_in_use,
                                            timestep=timestep,
                                            kv_cache=self.kv_cache1,
                                            crossattn_cache=self.crossattn_cache,
                                            current_start=current_start_frame * self.frame_seq_length,
                                        )
                                    
                                    # Scheduler add noise
                                    with step.timer("scheduler_add_noise"):
                                        next_timestep = self.denoising_step_list[index + 1]
                                        noisy_input = self.scheduler.add_noise(
                                            denoised_pred.flatten(0, 1),
                                            torch.randn_like(denoised_pred.flatten(0, 1)),
                                            next_timestep
                                            * torch.ones(
                                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long
                                            ),
                                        ).unflatten(0, denoised_pred.shape[:2])
                                else:
                                    # Final denoising step
                                    with step.timer("denoise_forward"):
                                        _, denoised_pred = self.generator(
                                            noisy_image_or_video=noisy_input,
                                            conditional_dict=cond_in_use,
                                            timestep=timestep,
                                            kv_cache=self.kv_cache1,
                                            crossattn_cache=self.crossattn_cache,
                                            current_start=current_start_frame * self.frame_seq_length,
                                        )

                    # Record output
                    if low_memory and output.device.type == 'cpu':
                        with self.latency.timer("d2h_transfer"):
                            output[:, current_start_frame : current_start_frame + current_num_frames] = denoised_pred.to(output.device)
                    else:
                        output[:, current_start_frame : current_start_frame + current_num_frames] = denoised_pred.to(output.device)

                    # rerun with clean context to update cache
                    with self.latency.timer("kv_recache"):
                        context_timestep = torch.ones_like(timestep) * self.args.context_noise
                        self.generator(
                            noisy_image_or_video=denoised_pred,
                            conditional_dict=cond_in_use,
                            timestep=context_timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                        )

                    # Finalize frame timing for each frame in this block
                    for frame_offset in range(current_num_frames):
                        self.latency.finalize_frame()

                    # Update frame pointer
                    current_start_frame += current_num_frames

                # Standard decoding
                with self.latency.batch_timer("vae_decode"):
                    if low_memory and output.device.type == 'cpu':
                        with self.latency.batch_timer("h2d_transfer_for_vae"):
                            output_for_vae = output.to(noise.device)
                        video = self.vae.decode_to_pixel(output_for_vae, use_cache=False)
                    else:
                        video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
                    video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output
        return video