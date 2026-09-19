# Training BRIDGE on FLUX.2 Klein 9B

BRIDGE uses the **same method** on Qwen and FLUX: a main path, an independently
denoised sub path, and learned discrete positional-encoding routing. Qwen trains
LoRA plus gates; FLUX trains the **full transformer plus gates**. VAE and text
encoder are frozen. The paper diagram illustrates the shared method; its Qwen
backbone/LoRA labels describe the Qwen realization, not a different method.

## 1. Environment and data

From `flux2/`, create the environment described in [README.md](README.md), then:

```bash
uv pip install --python .venv/bin/python -r requirements-training.txt
.venv/bin/hf download PANDATREE/BRIDGE --type dataset --local-dir ./data/BRIDGE
```

The `subject_condition/` extension provides **27,834 training rows** and
**3,092 test rows** used by this project's FLUX pipeline. Its 30,926 subject
conditions were generated using `Qwen/Qwen-Image-Edit-2511`, with the selected
best-of-three references retained. It preserves the original split and row order.
This is not the original Qwen train/evaluation split, and the 3,092-row test set
must not be described as the old Top-1000 evaluation.

Original `target_images/target_N.png` files remain separately obtained from
the original Pico-Banana preparation, as in the existing dataset release. They
are not in the new archives. Use the matching original target files; do not
substitute `fixed_images` or subject crops as targets. The preparation command
checks every required path and fails on missing files.

```bash
.venv/bin/python prepare_dataset.py \
  --dataset-root ./data/BRIDGE \
  --target-images /your/original/target_images \
  --output-dir ./data/local_manifests --extract
```

This verifies archive/member hashes, safely extracts only listed regular files,
and writes local absolute-path train/test manifests without overwriting existing
ones. Downloaded portable manifests remain unchanged. Allow disk space for both
approximately 31.4 GB of archives and their extracted assets, plus existing data.

### Exact cached-training field roles

| Field | Consumed as |
|---|---|
| `image` | Main target / ground truth |
| `edit_image` | Background image condition |
| `generated_subject_image` | Qwen-generated subject image condition |
| `sub` | Subject crop target for the generated sub branch |
| `back_mask` | Region selection and bbox geometry |
| `prompt`, `long_prompt` | Short/long text training conditions |

The new subject condition is **not** the sub target. Follow the cache pipeline
below: the legacy raw loader has a different data schema and is not the entry
point for these exported manifests.

## 2. Build the training cache

```bash
.venv/bin/python cache_flux2_full_dataset.py \
  --dataset-json ./data/local_manifests/train.json \
  --output-dir ./cache_train \
  --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-base-9B \
  --device cuda:0 --dtype bf16 --resolution 1024 --no-local-files-only

.venv/bin/python precompute_null_prompt.py --cache_dir ./cache_train \
  --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-base-9B \
  --device cuda --dtype bf16

.venv/bin/python precompute_bucket_order.py \
  --dataset ./data/local_manifests/train.json --cache_dir ./cache_train \
  --train_batch_size 2 --seed 0 --output ./bucket_order_bs2.json
```

Cache production VAE latents and Qwen3 layers 9/18/27 features once. Images use
native-aspect alignment to 16-pixel cells with a 1024-squared area cap. The cache
must contain `subcrop_sparse_*` from **subject-crop** tokens, not background slices.
The `_ids20` training loader remaps cached positions to main=0, sub=20 and
conditions=40/60. Mixed sub lengths are padded, with masks excluding padding
from attention, STE and sub loss. Batches must share main resolution.

Use a fresh cache for this manifest; do not reuse arbitrary shards keyed by
unrelated row indices. Do not train on test rows. Cached embeddings/latents and
optimizer states are intentionally not shipped as dataset assets.

## 3. Full-transformer training

The included ZeRO-1 config defaults to eight GPUs. Adjust its `num_processes`
and `CUDA_VISIBLE_DEVICES` together for your hardware; 9B full fine-tuning is
memory-intensive. The published sparse run's conversion manifest records six
ranks; the dense run records eight. Commands below are eight-GPU launch examples,
not a claim to reproduce every historical hardware detail bit-for-bit.

```bash
# Activate only this project's venv.
source .venv/bin/activate
export DATASET="$PWD/data/local_manifests/train.json"
export CACHE_DIR="$PWD/cache_train"
export BUCKET_ORDER_FILE="$PWD/bucket_order_bs2.json"
export ACCELERATE_CONFIG="$PWD/accelerate_config_zero1_inline.yaml"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export EXPERIMENT_MODE=hard_exchange
export TRAIN_BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=8
export GRADIENT_CHECKPOINTING=1 MIXED_PRECISION=bf16
export OPTIMIZER=prodigyplus_schedulefree LEARNING_RATE=1.0
export SUBJECT_DROP_PROB=0.5 PROMPT_DROP_PROB=0.1 LONG_PROMPT_PROB=0.5
export CHECKPOINTING_STEPS=200 REPORT_TO=none

# Choose ONE variant per run; keep separate output names.
SUB_BRANCH_LAYOUT=sparse_mask PE_EXCHANGE_REGION=mask \
  RUN_NAME=bridge_flux_sparse bash run_train_ablation.sh

# Run separately when resources are available:
SUB_BRANCH_LAYOUT=dense_bbox PE_EXCHANGE_REGION=bbox \
  RUN_NAME=bridge_flux_bbox bash run_train_ablation.sh
```

`MAX_TRAIN_STEPS`, `NUM_TRAIN_EPOCHS` (default 10), `OUTPUT_ROOT` and
`RESUME_FROM_CHECKPOINT` can be overridden. Resuming requires a compatible
training checkpoint with optimizer state, not a released inference-only export.
Rebuild batch ordering if batch size, data or resolution changes. The launcher
rejects dense bbox PE with sparse training layout and refuses missing caches.

Subject dropout removes only the subject **condition**, retaining sub and PE
exchange. Text dropout uses the encoded empty prompt, not zero embeddings.
Main and sub flow-matching losses are combined with default sub weight 1.0.
`transformer.requires_grad_(True)` enables full transformer/gate optimization;
there is no LoRA-only training in this FLUX path.

Other experiment modes are `no_sub`, `sub_no_exchange`, and `soft_exchange`.
The current soft batched RoPE path has a known batch-size>1 limitation; these
releases and the commands above use hard exchange.

## 4. Export inference checkpoints

ScheduleFree train/eval views differ. Historical DeepSpeed wrapper checkpoints
require reconstruction using the optimizer's master and `z` tensors; reading
the module's BF16 snapshot alone is insufficient. The converter validates
parameter fragments and uses `torch.lerp(train, z, 1 - 1/beta1)` when required.

```bash
.venv/bin/python consolidate_run_checkpoints_bf16.py \
  --run_dir ./runs/bridge_flux_bbox --experiment_mode hard_exchange \
  --pretrained black-forest-labs/FLUX.2-klein-base-9B \
  --checkpoints checkpoint-1400 --out_root ./inference_exports/bridge_flux_bbox \
  --keep_ds --no_local_files_only
```

**Always use `--keep_ds`** to preserve optimizer/resume data. Review converter
`--help` before use; its legacy default otherwise removes heavy training state.
Point the Gradio backend at the exported checkpoint directory containing
`transformer/`. Inference/evaluation must use **50 steps**, not reduced-step
quality tests. The public weights are already BF16 eval exports.

## Verification scope

Release checks cover CLI parsing, CPU unit tests, portable manifest preparation,
asset integrity and source provenance. No new full training run is launched
as part of documentation/publication. See [VALIDATION.md](VALIDATION.md).
