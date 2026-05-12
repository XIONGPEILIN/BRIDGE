# BRIDGE: Background Routing and Isolated Discrete Gating for Coarse-Mask Local Editing

[![arXiv](https://img.shields.io/badge/arXiv-2605.07846-b31b1b.svg)](https://arxiv.org/abs/2605.07846)

**BRIDGE** addresses coarse-mask local image editing by separating localization support from geometry generation. It uses BridgePath (Main Path + Subject Path) and a learnable Discrete Geometric Gate for token-level positional-embedding routing.

## Overview

- **BridgePath**: Two-path generation where Main Path preserves background context and Subject Path generates editable content from independent noise
- **Discrete Geometric Gate**: Token-level PE routing that lets subject tokens borrow background-anchored coordinates near fusion regions or keep subject-centric coordinates for geometry freedom
- **Lightweight**: 13.31M GateBlock parameters (vs ~1.13B for ControlNet-style branches)

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

Apache 2.0
