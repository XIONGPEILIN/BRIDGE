# BRIDGE: Background Routing and Isolated Discrete Gating for Coarse-Mask Local Editing

[![arXiv](https://img.shields.io/badge/arXiv-2605.07846-b31b1b.svg)](https://arxiv.org/abs/2605.07846)

**BRIDGE** addresses coarse-mask local image editing by separating localization support from geometry generation. It uses BridgePath (Main Path + Subject Path) and a learnable Discrete Geometric Gate for token-level positional-embedding routing.

## Overview

- **BridgePath**: Two-path generation where Main Path preserves background context and Subject Path generates editable content from independent noise
- **Discrete Geometric Gate**: Token-level PE routing that lets subject tokens borrow background-anchored coordinates near fusion regions or keep subject-centric coordinates for geometry freedom
- **Lightweight**: 13.31M GateBlock parameters (vs ~1.13B for ControlNet-style branches)

## Model Weights

Pre-trained weights are available on Hugging Face:

```
https://huggingface.co/PANDATREE/BRIDGE
```

Download `model.safetensors` and place it in the project root, then load with:

```python
from safetensors.torch import load_file
state = load_file("model.safetensors")
```

**Base model** (required): [`Qwen/Qwen-Image-Edit-2511`](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)

## Requirements

```bash
pip install torch torchvision
pip install -r requirements.txt
```

## Quick Start

### Gradio Demo

```bash
python apps_demo/app_gradio_multi.py
```

### Inference Script

See `DiffSynth-Studio/examples/qwen_image/model_training/train.py` for training,
and the `evaluation/` scripts for metrics computation.

## Training

```bash
bash training/Qwen-Image-Edit-2511.sh
```

## Dataset

The BRIDGE training/evaluation dataset is available on Hugging Face:

```
https://huggingface.co/datasets/PANDATREE/BRIDGE
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
