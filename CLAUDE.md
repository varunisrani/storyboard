# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Story2Board is a training-free framework for expressive storyboard generation from natural language stories. It uses FLUX.1-dev as the base T2I model with custom attention mechanisms to maintain character consistency across panels while allowing for diverse layouts and backgrounds.

## Core Architecture

The codebase consists of 4 main components:

1. **Story2BoardPipeline** (`story2board_pipeline.py`) - Custom diffusion pipeline extending FluxPipeline with Story2Board-specific functionality
2. **Story2BoardTransformer2DModel** (`story2board_transformer.py`) - Modified transformer that integrates the custom attention processor
3. **Story2BoardAttentionProcessor** (`story2board_attention_processor.py`) - Implements RAVM (Reciprocal Attention Value Mixing) and attention mechanisms
4. **AttentionStore** (`attention_store.py`) - Manages attention maps across diffusion steps and transformer blocks

## Key Technologies

- **FLUX.1-dev**: Base diffusion model from Black Forest Labs
- **Diffusers**: HuggingFace diffusion library (v0.34.0)
- **PyTorch**: Deep learning framework with CUDA 12.x support
- **Transformers**: For text encoding (CLIP, T5)

## Running the Code

### Environment Setup
```bash
# Create conda environment
conda create -n story2board python=3.12
conda activate story2board

# Install dependencies (Linux/CUDA 12.x officially supported)
pip install -r requirements.txt

# For Windows/macOS (unofficial)
pip install -r requirements_all_platforms.txt
```

### Basic Usage
```bash
python main.py \
  --subject "fox with shimmering fur and glowing eyes" \
  --ref_panel_prompt "stepping onto a mossy stone path under twilight trees" \
  --panel_prompts "bounding across a fallen tree" "perched atop a broken archway" \
  --output_dir outputs
```

### Key Parameters
- `--ravm_mixing_coef`: Controls RAVM mixing strength (default: 0.5)
- `--first_mixing_block`, `--last_mixing_block`: Transformer block range for RAVM (default: 30-57)
- `--first_mixing_denoising_step`, `--last_mixing_denoising_step`: Denoising step range for RAVM (default: 1-21)
- `--same_noise`: Use same initial noise for all panels (default: True)
- `--guidance`: CFG scale (default: 3.5)
- `--n_diff_steps`: Number of denoising steps (default: 28)

## Core Algorithms

### Latent Panel Anchoring (LPA)
Preserves shared character reference across panels by reusing reference latents.

### Reciprocal Attention Value Mixing (RAVM)
Softly blends visual features between token pairs with strong reciprocal attention, controlled by:
- Mixing coefficient (strength)
- Block range (which transformer layers)
- Denoising step range (when during generation)

## Output Structure

Generated storyboards are split into individual panels:
- Reference panel becomes `image_0.png`
- Subsequent panels become `image_1.png`, `image_2.png`, etc.
- Full prompts are logged for reproducibility

## Hardware Requirements

- **GPU**: CUDA-capable GPU with sufficient VRAM for FLUX.1-dev
- **Memory**: Adequate RAM for loading transformer models
- **Storage**: Space for model weights and generated outputs

## RunPod Deployment

### Docker Setup
The repository includes Docker configuration for RunPod deployment:

```bash
# Build the Docker image
docker build -t story2board .

# Run with GPU support
docker run --gpus all -it -v $(pwd)/outputs:/app/outputs story2board
```

### RunPod Configuration
1. **Template Selection**: Use NVIDIA CUDA 12.4+ template
2. **GPU Requirements**: RTX 4090, RTX 6000 Ada, A100, or similar (minimum 16GB VRAM recommended)
3. **Storage**: At least 50GB for model weights and outputs
4. **Container Setup**:
   - Use the provided Dockerfile
   - Mount `/app/outputs` for result persistence
   - Set environment variables for CUDA

### RunPod Deployment Steps
1. Upload repository to RunPod or clone from GitHub
2. Build container: `docker build -t story2board .`
3. Run container with GPU access
4. Execute generation commands inside container
5. Download results from `/app/outputs`

### Example RunPod Command
```bash
python main.py \
  --subject "mystical dragon" \
  --ref_panel_prompt "soaring above misty mountains at dawn" \
  --panel_prompts "diving through cloud formations" "landing on ancient castle towers" \
  --output_dir /app/outputs
```

## Development Notes

- The codebase extends HuggingFace Diffusers pipeline architecture
- Custom attention processor is injected into the transformer layers
- Attention maps are accumulated across diffusion steps for consistency
- The pipeline supports batched generation with shared noise initialization