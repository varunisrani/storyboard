from typing import Callable, List, Optional, Union
import torch
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline, FluxPipelineOutput
from diffusers.utils.torch_utils import randn_tensor

class Story2BoardPipeline(FluxPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_steps: int = 1,
        # Argumenter for attention mixing
        first_mixing_block: int = 30,
        last_mixing_block: int = 57,
        first_mixing_denoising_step: int = 1,
        last_mixing_denoising_step: int = 21,
    ):
        """Minimal docstring to prevent errors."""
        
        self.check_inputs(
            prompt=prompt, prompt_2=None, height=height, width=width,
            negative_prompt=negative_prompt, prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds, pooled_prompt_embeds, _ = self.encode_prompt(
            prompt=prompt,
            prompt_2="",
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
        )

        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = ""
            
            negative_prompt_embeds, negative_pooled_prompt_embeds, _ = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2="",
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                prompt_embeds=negative_prompt_embeds,
            )
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])
        
        pooled_prompt_embeds = pooled_prompt_embeds.unsqueeze(1).expand(-1, prompt_embeds.shape[1], -1)
        final_prompt_embeds = torch.cat([pooled_prompt_embeds, prompt_embeds], dim=-1)

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            self.transformer.config.in_channels,
            height,
            width,
            final_prompt_embeds.dtype,
            device,
            generator,
            latents,
        )[0]

        self.scheduler.set_timesteps(num_inference_steps, device=device, mu=4.0)
        timesteps = self.scheduler.timesteps

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = latents

                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latent_model_input] * 2)

                with self.transformer.transformer_blocks.register_forward_hook(
                    self.get_attention_mix_hook(
                        i, first_mixing_block, last_mixing_block, first_mixing_denoising_step, last_mixing_denoising_step
                    )
                ):
                    noise_pred = self.transformer(
                        latent_model_input, timestep=t, encoder_hidden_states=final_prompt_embeds
                    ).sample

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # --- START: DEN ENDELIGE KORREKSJONEN ---
                # Hent ut tensoren fra tuplen som returneres av scheduler.step
                latents = self.scheduler.step(noise_pred, t, latents)[0]
                # --- SLUTT: DEN ENDELIGE KORREKSJONEN ---
                
                progress_bar.update()

        if output_type == "latent":
            image = latents
        else:
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)

    def get_attention_mix_hook(self, step_index, first_mixing_block, last_mixing_block, first_mixing_denoising_step, last_mixing_denoising_step):
        def attention_mix_hook(module, input, output):
            if first_mixing_denoising_step <= step_index <= last_mixing_denoising_step:
                for block_index, block in enumerate(module):
                    if first_mixing_block <= block_index <= last_mixing_block:
                        block.attn.to_out[0].weight.data = (block.attn.to_out[0].weight.data[0] + block.attn.to_out[0].weight.data[1]) / 2
                        block.attn.to_out[0].weight.data[1] = block.attn.to_out[0].weight.data[0]
            return output
        return attention_mix_hook
