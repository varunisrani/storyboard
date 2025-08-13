
import argparse

import os
from pathlib import Path

import torch
from story2board_pipeline import Story2BoardPipeline
from attention_store import AttentionStore
from story2board_transformer import Story2BoardTransformer2DModel

def save_storyboard_panels(images, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    def split_image(image):
        width, height = image.size
        top_half = image.crop((0, 0, width, height // 2))
        bottom_half = image.crop((0, height // 2, width, height))
        return top_half, bottom_half

    panel_index = 0

    for i, image in enumerate(images):
        top, bottom = split_image(image)
        panels = [top, bottom] if i == 0 else [bottom]

        for panel in panels:
            filename = f"image_{panel_index}.png"
            path = os.path.join(output_dir, filename)
            panel.save(path)
            print(f"Saved panel at: {path}")
            panel_index += 1


def run_mutual_story(
    prompts, 
    seed, 
    guidance, 
    n_diff_steps,
    same_noise,
    ravm_mixing_coef,
    first_mixing_block,
    last_mixing_block,
    first_mixing_denoising_step,
    last_mixing_denoising_step,
    dtype,
    device=0,
):
    # Load custom Story2Board transformer
    base = "black-forest-labs/FLUX.1-dev"
    s2b_transformer = Story2BoardTransformer2DModel.from_pretrained(base, subfolder="transformer", torch_dtype=dtype)

    # Create custom Story2Board pipeline
    pipe = Story2BoardPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", 
        torch_dtype=dtype,
        # Custom Story2Board transformer
        transformer=s2b_transformer,
        # RAVM hyperparameters
        first_mixing_block=first_mixing_block,
        last_mixing_block=last_mixing_block,
        first_mixing_denoising_step=first_mixing_denoising_step,
        last_mixing_denoising_step=last_mixing_denoising_step,
    ).to(device)
    
    n_transformer_blocks = s2b_transformer.num_layers + s2b_transformer.num_single_layers
    n_prompt_tokens = pipe.tokenizer_2.model_max_length
    n_attn_heads = s2b_transformer.num_attention_heads
    n_image_tokens = s2b_transformer.joint_attention_dim

    attention_store = AttentionStore(
        batch_size=len(prompts),
        n_diff_steps=n_diff_steps,
        n_trans_blocks=n_transformer_blocks,
        n_image_tokens=n_image_tokens,
        n_attn_heads=n_attn_heads,
        dtype=dtype,
        device=device,
    )

    # Arguments to be passed to the attention processor
    attention_args = {
        'attn_store': attention_store,
        'n_prompt_tokens': n_prompt_tokens,
        'n_image_tokens': n_image_tokens,
        'ravm_mixing_coef': ravm_mixing_coef,
        'first_mixing_block': first_mixing_block,
        'last_mixing_block': last_mixing_block,
        'first_mixing_denoising_step': first_mixing_denoising_step,
        'last_mixing_denoising_step': last_mixing_denoising_step,
    }
    
    generator = torch.Generator("cuda").manual_seed(seed)
    
    images = pipe(
        prompts,
        guidance_scale=guidance,
        num_inference_steps=n_diff_steps,
        generator=generator,
        same_noise=same_noise,
        joint_attention_kwargs=attention_args,
    ).images
    
    return images


def parse_args():
    parser = argparse.ArgumentParser(description="Generate mutual storyboards with FluxPipeline")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--guidance', type=float, default=3.5,
                        help='Classifier-free guidance scale (higher values = closer to prompt, lower = more diverse)')
    parser.add_argument('--same_noise', type=bool, default=True,
                        help='Use the same initial noise tensor for all images in the batch')
    parser.add_argument('--ravm_mixing_coef', type=float, default=0.5,
                        help='Mixing coefficient for Reciprocal Attention Value Mixing (RAVM)')
    parser.add_argument('--first_mixing_block', type=int, default=30,
                        help='First transformer block index where RAVM mixing is applied')
    parser.add_argument('--last_mixing_block', type=int, default=57,
                        help='Last transformer block index where RAVM mixing is applied')
    parser.add_argument('--first_mixing_denoising_step', type=int, default=1,
                        help='First denoising step where RAVM mixing is applied')
    parser.add_argument('--last_mixing_denoising_step', type=int, default=21,
                        help='Last denoising step where RAVM mixing is applied')
    parser.add_argument('--n_diff_steps', type=int, default=28,
                        help='Total number of denoising steps')
    parser.add_argument('--subject', type=str, required=True,
        help='Main subject of the storyboard.')
    parser.add_argument(
        '--ref_panel_prompt', type=str, required=True,
        help='Prompt describing the reference (top) panel.')
    parser.add_argument(
        '--panel_prompts', nargs='+', type=str, default=[],
        help='List of panel prompts (space-separated).')
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Directory where generated images will be saved.')

    args = parser.parse_args()

    # Validate arguments
    if args.first_mixing_denoising_step < 1:
        parser.error("--first_mixing_denoising_step must be at least 1.")

    # Construct full prompts
    args.prompts = [
        f"A storyboard of a {args.subject} {args.ref_panel_prompt} (top) and "
        f"the exact same {args.subject} {panel_prompt} (bottom)"
        for panel_prompt in args.panel_prompts
    ]
    
    return args


if __name__ == '__main__':
    args = parse_args()
    
    print('Beginning storyboard generation...')
    print(f'{args.prompts=}')

    # Setup output directory
    outputs_dir = Path(args.output_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    storyboard_panels = run_mutual_story(
        prompts=args.prompts,
        seed=args.seed,
        guidance=args.guidance,
        n_diff_steps=args.n_diff_steps,
        same_noise=args.same_noise,
        ravm_mixing_coef=args.ravm_mixing_coef,
        first_mixing_block=args.first_mixing_block,
        last_mixing_block=args.last_mixing_block,
        first_mixing_denoising_step=args.first_mixing_denoising_step,
        last_mixing_denoising_step=args.last_mixing_denoising_step,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device=0 if torch.cuda.is_available() else 'cpu',
    )

    save_storyboard_panels(storyboard_panels, output_dir=outputs_dir)    
    print(f'Saved storyboard panels at {args.output_dir}')

