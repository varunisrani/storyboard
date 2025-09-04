#!/usr/bin/env python3
"""
RunPod Serverless Handler for Story2Board
Handles requests from RunPod's serverless platform and executes Story2Board generation
"""

import os
import sys
import json
import base64
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
from PIL import Image

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Story2Board components
from story2board_pipeline import Story2BoardPipeline
from attention_store import AttentionStore
from story2board_transformer import Story2BoardTransformer2DModel

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def split_image(image: Image.Image) -> List[Image.Image]:
    """Split image into top and bottom halves"""
    width, height = image.size
    top_half = image.crop((0, 0, width, height // 2))
    bottom_half = image.crop((0, height // 2, width, height))
    return [top_half, bottom_half]

def parse_input(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Parse and validate input parameters"""
    # Required parameters
    subject = job_input.get('SUBJECT', job_input.get('subject', ''))
    ref_panel_prompt = job_input.get('REF_PANEL_PROMPT', job_input.get('ref_panel_prompt', ''))
    panel_prompts_str = job_input.get('PANEL_PROMPTS', job_input.get('panel_prompts', ''))
    
    if not subject:
        raise ValueError("SUBJECT is required")
    if not ref_panel_prompt:
        raise ValueError("REF_PANEL_PROMPT is required")
    if not panel_prompts_str:
        raise ValueError("PANEL_PROMPTS is required")
    
    # Parse panel prompts (comma-separated string to list)
    panel_prompts = [p.strip() for p in panel_prompts_str.split(',') if p.strip()]
    if not panel_prompts:
        raise ValueError("At least one panel prompt is required")
    
    # Optional parameters with defaults
    params = {
        'subject': subject,
        'ref_panel_prompt': ref_panel_prompt,
        'panel_prompts': panel_prompts,
        'seed': int(job_input.get('SEED', job_input.get('seed', 42))),
        'guidance_scale': float(job_input.get('GUIDANCE_SCALE', job_input.get('guidance_scale', 3.5))),
        'n_diff_steps': int(job_input.get('N_DIFF_STEPS', job_input.get('n_diff_steps', 28))),
        'same_noise': str(job_input.get('SAME_NOISE', job_input.get('same_noise', 'true'))).lower() == 'true',
        'ravm_mixing_coef': float(job_input.get('RAVM_MIXING_COEF', job_input.get('ravm_mixing_coef', 0.5))),
        'first_mixing_block': int(job_input.get('FIRST_MIXING_BLOCK', job_input.get('first_mixing_block', 30))),
        'last_mixing_block': int(job_input.get('LAST_MIXING_BLOCK', job_input.get('last_mixing_block', 57))),
        'first_mixing_denoising_step': int(job_input.get('FIRST_MIXING_DENOISING_STEP', job_input.get('first_mixing_denoising_step', 1))),
        'last_mixing_denoising_step': int(job_input.get('LAST_MIXING_DENOISING_STEP', job_input.get('last_mixing_denoising_step', 21))),
    }
    
    return params

def generate_storyboard(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate storyboard using Story2Board pipeline"""
    
    # Construct prompts in the format expected by Story2Board
    prompts = [
        f"A storyboard of a {params['subject']} {params['ref_panel_prompt']} (top) and "
        f"the exact same {params['subject']} {panel_prompt} (bottom)"
        for panel_prompt in params['panel_prompts']
    ]
    
    print(f"Generated prompts: {prompts}")
    
    # Determine device and dtype
    device = 0 if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    print(f"Using device: {device}, dtype: {dtype}")
    
    # Load custom Story2Board transformer
    base = "black-forest-labs/FLUX.1-dev"
    print("Loading Story2Board transformer...")
    s2b_transformer = Story2BoardTransformer2DModel.from_pretrained(
        base, 
        subfolder="transformer", 
        torch_dtype=dtype
    )
    
    # Create custom Story2Board pipeline
    print("Creating Story2Board pipeline...")
    pipe = Story2BoardPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", 
        torch_dtype=dtype,
        transformer=s2b_transformer,
        first_mixing_block=params['first_mixing_block'],
        last_mixing_block=params['last_mixing_block'],
        first_mixing_denoising_step=params['first_mixing_denoising_step'],
        last_mixing_denoising_step=params['last_mixing_denoising_step'],
    ).to(device)
    
    # Setup attention store
    n_transformer_blocks = s2b_transformer.num_layers + s2b_transformer.num_single_layers
    n_prompt_tokens = pipe.tokenizer_2.model_max_length
    n_attn_heads = s2b_transformer.num_attention_heads
    n_image_tokens = s2b_transformer.joint_attention_dim
    
    print("Setting up attention store...")
    attention_store = AttentionStore(
        batch_size=len(prompts),
        n_diff_steps=params['n_diff_steps'],
        n_trans_blocks=n_transformer_blocks,
        n_image_tokens=n_image_tokens,
        n_attn_heads=n_attn_heads,
        dtype=dtype,
        device=device,
    )
    
    # Arguments for attention processor
    attention_args = {
        'attn_store': attention_store,
        'n_prompt_tokens': n_prompt_tokens,
        'n_image_tokens': n_image_tokens,
        'ravm_mixing_coef': params['ravm_mixing_coef'],
        'first_mixing_block': params['first_mixing_block'],
        'last_mixing_block': params['last_mixing_block'],
        'first_mixing_denoising_step': params['first_mixing_denoising_step'],
        'last_mixing_denoising_step': params['last_mixing_denoising_step'],
    }
    
    # Set up generator
    generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(params['seed'])
    
    print("Starting storyboard generation...")
    # Generate images
    images = pipe(
        prompts,
        guidance_scale=params['guidance_scale'],
        num_inference_steps=params['n_diff_steps'],
        generator=generator,
        same_noise=params['same_noise'],
        joint_attention_kwargs=attention_args,
    ).images
    
    print(f"Generated {len(images)} storyboard images")
    
    # Process and split images into panels
    panels = []
    panel_info = []
    
    for i, image in enumerate(images):
        if i == 0:
            # First image: both top (reference) and bottom panels
            top, bottom = split_image(image)
            panels.extend([top, bottom])
            panel_info.extend([
                {"type": "reference", "prompt": f"{params['subject']} {params['ref_panel_prompt']}", "index": 0},
                {"type": "story", "prompt": f"{params['subject']} {params['panel_prompts'][0]}", "index": 1}
            ])
        else:
            # Subsequent images: only bottom panel
            _, bottom = split_image(image)
            panels.append(bottom)
            panel_info.append({
                "type": "story", 
                "prompt": f"{params['subject']} {params['panel_prompts'][i]}", 
                "index": len(panels) - 1
            })
    
    # Convert panels to base64
    panel_data = []
    for i, panel in enumerate(panels):
        panel_base64 = image_to_base64(panel)
        panel_data.append({
            "image": panel_base64,
            "info": panel_info[i]
        })
    
    return {
        "panels": panel_data,
        "metadata": {
            "total_panels": len(panels),
            "subject": params['subject'],
            "ref_panel_prompt": params['ref_panel_prompt'],
            "panel_prompts": params['panel_prompts'],
            "generation_params": {
                "seed": params['seed'],
                "guidance_scale": params['guidance_scale'],
                "n_diff_steps": params['n_diff_steps'],
                "ravm_mixing_coef": params['ravm_mixing_coef']
            }
        }
    }

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main handler function for RunPod serverless
    """
    try:
        print("Story2Board Handler - Starting...")
        print(f"Event: {json.dumps(event, indent=2)}")
        
        # Get input from event
        job_input = event.get('input', {})
        if not job_input:
            return {"error": "No input provided"}
        
        # Parse and validate input
        print("Parsing input parameters...")
        params = parse_input(job_input)
        print(f"Parsed parameters: {json.dumps(params, indent=2, default=str)}")
        
        # Generate storyboard
        print("Generating storyboard...")
        result = generate_storyboard(params)
        
        print(f"Story2Board generation completed successfully! Generated {len(result['panels'])} panels")
        
        return result
        
    except Exception as e:
        print(f"Error in Story2Board handler: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# For local testing
if __name__ == "__main__":
    # Test the handler locally
    test_input = {
        "input": {
            "SUBJECT": "cute robot",
            "REF_PANEL_PROMPT": "standing in a colorful garden",
            "PANEL_PROMPTS": "waving hello,picking flowers",
            "SEED": "42",
            "N_DIFF_STEPS": "20"
        }
    }
    
    result = handler(test_input)
    print("Test result:", json.dumps(result, indent=2)[:500] + "...")