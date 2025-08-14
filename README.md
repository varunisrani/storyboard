# Story2Board: Training-Free, Consistent & Expressive Storyboard Generation

**Project page:** https://daviddinkevich.github.io/Story2Board  
**Paper (arXiv):** _coming soon_  
**Code:** this repo

---

## Abstract

We present **Story2Board**, a training-free framework for expressive storyboard generation from natural language. Existing methods narrowly focus on subject identity, overlooking key aspects of visual storytelling such as spatial composition, background evolution, and narrative pacing. To address this, we introduce a lightweight consistency framework composed of two components: **Latent Panel Anchoring**, which preserves a shared character reference across panels, and **Reciprocal Attention Value Mixing (RAVM)**, which softly blends visual features between token pairs with strong reciprocal attention. Together, these mechanisms enhance coherence without architectural changes or fine-tuning, enabling state-of-the-art diffusion models to generate visually diverse yet consistent storyboards. We convert free-form stories into grounded panel-level prompts with an off-the-shelf LLM, and evaluate on a new **Rich Storyboard Benchmark** that measures layout diversity, background-grounded storytelling, and consistency. Qualitative/quantitative results and a user study show that Story2Board produces more dynamic, coherent, and narratively engaging storyboards than existing baselines.

<p>
    <img src="docs/teaser.webp" width="800px"/>  
    <br/>
    A training-free method for storyboard generation that balances identity consistency with cinematic layout diversity.
</p>

---

## Environment Setup

We recommend a fresh Conda environment with Python 3.12.

```bash
# 1) Create and activate env
conda create -n story2board python=3.12 -y
conda activate story2board

# 2) Install dependencies
pip install -r requirements.txt
```

> Tip: If you want a specific CUDA build of PyTorch, install PyTorch first following the official instructions, then run `pip install -r requirements.txt`.

---

## Quickstart

The entry point is `main.py`. The **required** arguments are:

- `--subject` – the main subject (e.g., “smiling boy”).
- `--ref_panel_prompt` – description of the **reference (top) panel**.
- `--panel_prompts` – one or more prompts for the remaining panel(s).
- `--output_dir` – where to save results.

Minimal skeleton:

```bash
python main.py   --subject "SUBJECT_NAME"   --ref_panel_prompt "REFERENCE_PANEL_TEXT"   --panel_prompts "PANEL_1_TEXT" "PANEL_2_TEXT" ...   --output_dir path/to/out
```

### Concrete example

```bash
python main.py   --subject "smiling boy"   --ref_panel_prompt "clearly visible and centered in a sunlit train yard, warm golden-hour light"   --panel_prompts     "sitting on a broken crate, sketching the shadows of overhead wires"     "watching rats scurry past graffiti-covered pillars"     "peering through a chain-link fence at a passing freight train"   --output_dir outputs/smiling_boy_trainyard
```

This will generate a storyboard where the **top** panel is the reference, and each **bottom** panel reuses the same character identity while varying the scene/action.

> Notes:
> - By default the script sets a seed (`--seed 40`) and uses the **same initial noise** across the batch (`--same_noise`), making panel comparisons cleaner.  
> - Additional knobs (guidance, RAVM ranges, diffusion steps) exist but aren’t required for a first run.

---

## Outputs

- Generated images are written to `--output_dir`.  
- The constructed, per-panel prompts are logged for reproducibility.

---

## Method Overview (Very Brief)

- **Latent Panel Anchoring**: reuses a shared reference latent to stabilize identity across panels.  
- **RAVM (Reciprocal Attention Value Mixing)**: gently blends attention **values** between token pairs with strong reciprocal attention, preserving the model’s prior while improving cross-panel coherence.

---

## BibTeX

```bibtex
@article{dinkevich2025story2board,
  title   = {Story2Board: Training-Free, Consistent and Expressive Storyboard Generation},
  author  = {Dinkevich, David and Levy, Matan and Avrahami, Omri and Samuel, Dvir and Lischinski, Dani},
  journal = {arXiv preprint arXiv:xxxx.xxxxx},
  year    = {2025}
}
```

---

## Acknowledgements

This repository builds on the excellent open-source ecosystems of **PyTorch** and **Hugging Face Diffusers**, and uses **FLUX.1-dev** weights as the base T2I model. Our README style and repo organization take inspiration from public research codebases such as [Stable Flow](https://github.com/snap-research/stable-flow).

---

## License

See `LICENSE` in this repository.
