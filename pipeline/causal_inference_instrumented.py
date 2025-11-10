# This is the instrumented inference method - to be integrated into causal_inference.py
from typing import List
import torch
from .utils.memory_utils import (
    get_cuda_free_memory_gb,
    move_model_to_device_with_memory_preservation,
)
def inference_instrumented(
    self,
    noise: torch.Tensor,
    text_prompts: List[str],
    return_latents: bool = False,
    profile: bool = False,
    low_memory: bool = False,
) -> torch.Tensor:
    """
    Perform inference on the given noise and text prompts with latency logging.
    """
    batch_size, num_output_frames, num_channels, height, width = noise.shape
    assert num_output_frames % self.num_frame_per_block == 0
    num_blocks = num_output_frames // self.num_frame_per_block

    # Start batch-level timing
    with self.latency.batch_scope(
        batch_idx=0, 
        batch_size=batch_size, 
        num_output_frames=num_output_frames,
        num_frame_per_block=self.num_frame_per_block,
        local_attn_size=self.local_attn_size,
        denoising_steps=len(self.denoising_step_list)
    ):
        with self.latency.batch_timer("batch_total"):
            # Text encoding
            with self.latency.batch_timer("text_encode"):
                conditional_dict = self.text_encoder(text_prompts=text_prompts)

            if low_memory:
                gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
                move_model_to_device_with_memory_preservation(
                    self.text_encoder, target_device=gpu, 
                    preserved_memory_gb=gpu_memory_preservation
                )

            # Decide the device for output
            output_device = torch.device('cpu') if low_memory else noise.device
            output = torch.zeros(
                [batch_size, num_output_frames, num_channels, height, width],
                device=output_device,
                dtype=noise.dtype
            )

            # Set up profiling if requested
            if profile:
                init_start = torch.cuda.Event(enable_timing=True)
                init_end = torch.cuda.Event(enable_timing=True)
                diffusion_start = torch.cuda.Event(enable_timing=True)
                diffusion_end = torch.cuda.Event(enable_timing=True)
                vae_start = torch.cuda.Event(enable_timing=True)
                vae_end = torch.cuda.Event(enable_timing=True)
                block_times = []
                block_start = torch.cuda.Event(enable_timing=True)
                block_end = torch.cuda.Event(enable_timing=True)
                init_start.record()

            # Step 1: Initialize KV cache
            with self.latency.batch_timer("kv_init"):
                local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
                kv_policy = ""
                if local_attn_cfg != -1:
                    kv_cache_size = local_attn_cfg * self.frame_seq_length
                    kv_policy = f"int->local, size={local_attn_cfg}"
                else:
                    kv_cache_size = num_output_frames * self.frame_seq_length
                    kv_policy = "global (-1)"
                print(f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})")

                self._initialize_kv_cache(
                    batch_size=batch_size,
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

            if profile:
                init_end.record()
                torch.cuda.synchronize()
                diffusion_start.record()

            # Step 2: Temporal denoising loop
            all_num_frames = [self.num_frame_per_block] * num_blocks
            for block_idx, current_num_frames in enumerate(all_num_frames):
                if profile:
                    block_start.record()

                noisy_input = noise[:, current_start_frame:current_start_frame + current_num_frames]

                # Start frame timing for each frame in this block
                for frame_offset in range(current_num_frames):
                    frame_idx = current_start_frame + frame_offset
                    self.latency.start_frame(
                        frame_idx,
                        prompt=text_prompts[0] if isinstance(text_prompts, list) else str(text_prompts),
                        block_idx=block_idx,
                    )

                # Step 2.1: Spatial denoising loop
                with self.latency.timer("denoise_loop_total"):
                    for index, current_timestep in enumerate(self.denoising_step_list):
                        with self.latency.step_timer(index) as step:
                            # Prepare timestep
                            with step.timer("prepare_timestep"):
                                timestep = torch.ones(
                                    [batch_size, current_num_frames],
                                    device=noise.device,
                                    dtype=torch.int64
                                ) * current_timestep

                            if index < len(self.denoising_step_list) - 1:
                                # Forward pass
                                with step.timer("denoise_forward"):
                                    _, denoised_pred = self.generator(
                                        noisy_image_or_video=noisy_input,
                                        conditional_dict=conditional_dict,
                                        timestep=timestep,
                                        kv_cache=self.kv_cache1,
                                        crossattn_cache=self.crossattn_cache,
                                        current_start=current_start_frame * self.frame_seq_length
                                    )
                                
                                # Scheduler add noise
                                with step.timer("scheduler_add_noise"):
                                    next_timestep = self.denoising_step_list[index + 1]
                                    noisy_input = self.scheduler.add_noise(
                                        denoised_pred.flatten(0, 1),
                                        torch.randn_like(denoised_pred.flatten(0, 1)),
                                        next_timestep * torch.ones(
                                            [batch_size * current_num_frames], 
                                            device=noise.device, 
                                            dtype=torch.long
                                        )
                                    ).unflatten(0, denoised_pred.shape[:2])
                            else:
                                # Final denoising step
                                with step.timer("denoise_forward"):
                                    _, denoised_pred = self.generator(
                                        noisy_image_or_video=noisy_input,
                                        conditional_dict=conditional_dict,
                                        timestep=timestep,
                                        kv_cache=self.kv_cache1,
                                        crossattn_cache=self.crossattn_cache,
                                        current_start=current_start_frame * self.frame_seq_length
                                    )

                # Step 2.2: record the model's output
                output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred.to(output.device)
                
                # Step 2.3: rerun with timestep zero to update KV cache using clean context
                with self.latency.timer("kv_recache"):
                    context_timestep = torch.ones_like(timestep) * self.args.context_noise
                    self.generator(
                        noisy_image_or_video=denoised_pred,
                        conditional_dict=conditional_dict,
                        timestep=context_timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )

                # Finalize frame timing for each frame in this block
                for frame_offset in range(current_num_frames):
                    self.latency.finalize_frame()

                if profile:
                    block_end.record()
                    torch.cuda.synchronize()
                    block_time = block_start.elapsed_time(block_end)
                    block_times.append(block_time)

                # Update frame pointer
                current_start_frame += current_num_frames

            if profile:
                diffusion_end.record()
                torch.cuda.synchronize()
                diffusion_time = diffusion_start.elapsed_time(diffusion_end)
                init_time = init_start.elapsed_time(init_end)
                vae_start.record()

            # Step 3: Decode the output
            with self.latency.batch_timer("vae_decode"):
                video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
                video = (video * 0.5 + 0.5).clamp(0, 1)
            
            if profile:
                vae_end.record()
                torch.cuda.synchronize()
                vae_time = vae_start.elapsed_time(vae_end)
                total_time = init_time + diffusion_time + vae_time

                print("Profiling results:")
                print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
                print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
                for i, block_time in enumerate(block_times):
                    print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
                print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
                print(f"  - Total time: {total_time:.2f} ms")

    if return_latents:
        return video, output.to(noise.device)
    else:
        return video
