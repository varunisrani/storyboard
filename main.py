import argparse
import os
import torch
from diffusers.pipelines.flux.pipeline_flux import VaeImageProcessor
from PIL import Image
from story2board_pipeline import Story2BoardPipeline

def run_mutual_story(
    prompts: list[str],
    num_inference_steps: int = 28,
    guidance_scale: float = 7,
    first_mixing_block: int = 30,
    last_mixing_block: int = 57,
    first_mixing_denoising_step: int = 1,
    last_mixing_denoising_step: int = 21,
):
    """
    Generates a storyboard from a list of prompts.
    """
    # Denne linjen er ikke lenger nødvendig her, da offloading håndterer enheten
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Laster modellen med bfloat16 for å redusere minnebruk
    pipe = Story2BoardPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        max_sequence_length=512,
        split_embed_tokens=True,
    )

    # Aktiverer minneoptimaliseringer.
    # enable_sequential_cpu_offload vil automatisk bruke GPUen.
    pipe.enable_sequential_cpu_offload()
    pipe.enable_attention_slicing()

    # Fjernet: pipe.to(device) - DENNE LINJEN FORÅRSAKET FEILEN

    storyboard_panels = []
    for prompt in prompts:
        storyboard = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            first_mixing_block=first_mixing_block,
            last_mixing_block=last_mixing_block,
            first_mixing_denoising_step=first_mixing_denoising_step,
            last_mixing_denoising_step=last_mixing_denoising_step,
        ).images[0]
        storyboard_panels.append(storyboard)

    # Extract individual panels from the storyboard
    storyboard_width, storyboard_height = storyboard_panels[0].size
    panel_height = storyboard_height // 2
    panels = []
    for storyboard in storyboard_panels:
        panel_1 = storyboard.crop((0, 0, storyboard_width, panel_height))
        panel_2 = storyboard.crop((0, panel_height, storyboard_width, storyboard_height))
        panels.append(panel_1)
        panels.append(panel_2)

    return panels

def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate a storyboard from prompts.")
    parser.add_argument("--subject", type=str, required=True, help="The subject of the storyboard.")
    parser.add_argument("--ref_panel_prompt", type=str, required=True, help="The prompt for the reference panel.")
    parser.add_argument(
        "--panel_prompts",
        nargs="+",
        required=True,
        help="The prompts for the subsequent panels.",
    )
    parser.add_argument("--output_dir", type=str, default="outputs", help="The directory to save the output images.")
    parser.add_argument("--num_inference_steps", type=int, default=28, help="The number of denoising steps.")
    parser.add_argument("--guidance_scale", type=float, default=7, help="The guidance scale.")
    parser.add_argument("--first_mixing_block", type=int, default=30, help="The first mixing block.")
    parser.add_argument("--last_mixing_block", type=int, default=57, help="The last mixing block.")
    parser.add_argument("--first_mixing_denoising_step", type=int, default=1, help="The first mixing denoising step.")
    parser.add_argument(
        "--last_mixing_denoising_step",
        type=int,
        default=21,
        help="The last mixing denoising step.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print("Beginning storyboard generation...")

    prompts = [
        f"A storyboard of a {args.subject} {args.ref_panel_prompt} (top) and the exact same {args.subject} {panel_prompt} (bottom)"
        for panel_prompt in args.panel_prompts
    ]
    print(f"args.prompts={prompts}")

    storyboard_panels = run_mutual_story(
        prompts=prompts,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        first_mixing_block=args.first_mixing_block,
        last_mixing_block=args.last_mixing_block,
        first_mixing_denoising_step=args.first_mixing_denoising_step,
        last_mixing_denoising_step=args.last_mixing_denoising_step,
    )

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save the panels
    # Prepend the reference panel to the list of panels
    ref_panel = storyboard_panels[0]
    ref_panel.save(os.path.join(args.output_dir, "panel_0.png"))

    # Save the subsequent panels
    for i, panel in enumerate(storyboard_panels[1::2]):
        panel.save(os.path.join(args.output_dir, f"panel_{i+1}.png"))

    print(f"Storyboard panels saved to {args.output_dir}")
