# BRIDGE: Background Routing and Isolated Discrete Gating for Coarse-Mask Local Editing

[![arXiv](https://img.shields.io/badge/arXiv-2605.07846-b31b1b.svg)](https://arxiv.org/abs/2605.07846)

**BRIDGE** addresses coarse-mask local image editing by separating localization support from geometry generation. It uses BridgePath (Main Path + Subject Path) and a learnable Discrete Geometric Gate for token-level positional-embedding routing.

## Overview

- **BridgePath**: Two-path generation where Main Path preserves background context and Subject Path generates editable content from independent noise
- **Discrete Geometric Gate**: Token-level PE routing that lets subject tokens borrow background-anchored coordinates near fusion regions or keep subject-centric coordinates for geometry freedom
- **Original Qwen implementation**: 13.31M GateBlock parameters (vs ~1.13B for ControlNet-style branches); this count is not the size of the FLUX full-transformer checkpoints.

## FLUX.2 Klein 9B update

BRIDGE now also includes two trained **FLUX.2 Klein 9B** variants, using the same
main/subject paths and discrete positional-encoding routing:

- **Sparse / Mask**: mask-selected sub tokens with mask-based PE exchange.
- **Dense / BBox**: a full bbox sub grid with bbox-wide PE exchange.

Both step-1400 **BF16 ScheduleFree eval** checkpoints are released alongside the
original Qwen weights on [Hugging Face](https://huggingface.co/PANDATREE/BRIDGE/tree/main/flux2-klein-9b).
See **[FLUX setup, Gradio comparison, subject inputs, sub-token control, and examples](flux2/README.md)**.
Generation uses **50 inference steps**. The FLUX weights are full transformer
states, not Qwen-compatible LoRA adapters, and carry the FLUX Non-Commercial
License. The original Qwen code and weight file remain available unchanged.

The sections below describe the original **Qwen** release.

## Download

### 1. Base Model (Required)

BRIDGE is built on top of Qwen-Image-Edit-2511. You must download it first:

```bash
# Coming soon: auto-download in inference script
# Model: https://huggingface.co/Qwen/Qwen-Image-Edit-2511
```

### 2. BRIDGE Model Weights

Pre-trained BRIDGE weights (STE GateBlocks + LoRA) are available on Hugging Face:

```
https://huggingface.co/PANDATREE/BRIDGE
```

Download `model.safetensors`:

```python
from safetensors.torch import load_file
state = load_file("model.safetensors")

# The checkpoint contains:
# - pipe.ste.* → GateBlocks (Discrete Geometric Gate)
# - lora_*     → LoRA adapters (rank 512)
```

### 3. Dataset

The BRIDGE training/evaluation dataset (with `global_caption`/`local_caption`) is available on Hugging Face:

```
https://huggingface.co/datasets/PANDATREE/BRIDGE
```

**Contents:**
- `dataset_qwen_pe_reversed.json` — 42,425 training pairs with captions
- `dataset_qwen_pe_top1000_captioned.json` — 1,000 evaluation pairs with captions
- `fixed_images/` — edited results (guided-filter blended)
- `ref_gt_fixed/` / `ref_gt_fixed_crop/` — reference ground truth
- `fixed_masks/` — editing region masks

> **Note:** `target_images/` (original edited outputs from Nano-Banana) are not included in this repo. Please obtain them from [Apple Pico-Banana-400K](https://github.com/apple/pico-banana-400k) (CC BY-NC-ND 4.0).

## Requirements

```bash
pip install torch torchvision
pip install -r requirements.txt
```

## Quick Start

### Gradio Demo

```bash
# Place model.safetensors at:
# train/Qwen-Image-Edit-2511_lora-rank512-cfg/step-28000.safetensors

python apps_demo/app_gradio_multi.py
```

### Inference

See `DiffSynth-Studio/examples/qwen_image/model_training/train.py` for training,
and the `evaluation/` scripts for metrics computation.

## Training

```bash
bash training/Qwen-Image-Edit-2511.sh
```

## Citation

```bibtex
@article{xiong2025bridge,
  title={BRIDGE: Background Routing and Isolated Discrete Gating for Coarse-Mask Local Editing},
  author={Peilin Xiong, Honghui Yuan, Junwen Chen, Keiji Yanai},
  journal={arXiv preprint arXiv:2605.07846},
  year={2025}
}
```

## License

The original code release is Apache 2.0. The newly added FLUX model weights are
subject to the upstream [FLUX Non-Commercial License](flux2/LICENSE-FLUX.md);
see [the FLUX notice](flux2/NOTICE-FLUX.md). Apache 2.0 does not relicense those weights.
