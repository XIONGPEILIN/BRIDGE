#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Unified ablation launcher. One script runs ALL FOUR experiment modes through
# the SAME cache-backed pipeline (train_flux2_klein_subject_img2img_sub_ids20.py
# + cache_full/ + torch.compile max-autotune + suppress_errors).
#
#   EXPERIMENT_MODE=no_sub|sub_no_exchange|hard_exchange|soft_exchange  (required)
#
# Example (single 4-GPU group with compile):
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_CONFIG=$PWD/accelerate_config_4gpu_a.yaml \
#   EXPERIMENT_MODE=hard_exchange NUM_TRAIN_EPOCHS=10 bash run_train_ablation.sh
#
# Smoke test (3 steps, one big GPU):
#   CUDA_VISIBLE_DEVICES=4 ACCELERATE_CONFIG=$PWD/accelerate_config_1gpu_compile_p1.yaml \
#   EXPERIMENT_MODE=soft_exchange MAX_TRAIN_STEPS=3 MAX_TRAIN_SAMPLES=64 \
#   REPORT_TO=tensorboard bash run_train_ablation.sh
# ============================================================================

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/.venv/bin/activate"

EXPERIMENT_MODE="${EXPERIMENT_MODE:?set EXPERIMENT_MODE=no_sub|sub_no_exchange|hard_exchange|soft_exchange}"
case "$EXPERIMENT_MODE" in
  no_sub|sub_no_exchange|hard_exchange|soft_exchange) ;;
  *) echo "[run] bad EXPERIMENT_MODE='$EXPERIMENT_MODE'" >&2; exit 1 ;;
esac

# Move compile/autotune caches off NFS and reduce fragmentation. Per-mode dirs so two
# concurrent ablations on one box never clobber each other's inductor/triton cache.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/dev/shm/abl-$EXPERIMENT_MODE-tri}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/dev/shm/abl-$EXPERIMENT_MODE-ind}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.2-klein-base-9B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/runs}"
DATASET="${DATASET:-$ROOT/dataset_qwen_subject_driven_best_of_three_mapped_abs_train.json}"
CACHE_DIR="${CACHE_DIR:-$ROOT/cache_full}"
RESOLUTION="${RESOLUTION:-1024}"
DOWNSAMPLE_FACTOR="${DOWNSAMPLE_FACTOR:-16}"
SEED="${SEED:-0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
SUB_BRANCH_LAYOUT="${SUB_BRANCH_LAYOUT:-sparse_mask}"
PE_EXCHANGE_REGION="${PE_EXCHANGE_REGION:-mask}"
case "$SUB_BRANCH_LAYOUT" in
  sparse_mask|dense_bbox) ;;
  *) echo "[run] bad SUB_BRANCH_LAYOUT='$SUB_BRANCH_LAYOUT' (expected sparse_mask|dense_bbox)" >&2; exit 1 ;;
esac
case "$PE_EXCHANGE_REGION" in
  mask|bbox) ;;
  *) echo "[run] bad PE_EXCHANGE_REGION='$PE_EXCHANGE_REGION' (expected mask|bbox)" >&2; exit 1 ;;
esac
if [[ "$SUB_BRANCH_LAYOUT" == "sparse_mask" && "$PE_EXCHANGE_REGION" == "bbox" ]]; then
  echo "[run] PE_EXCHANGE_REGION=bbox requires SUB_BRANCH_LAYOUT=dense_bbox." >&2
  exit 1
fi
# Keep incompatible training protocols in different output directories by default. An
# explicitly supplied RUN_NAME still takes precedence, which is useful for named reruns.
RUN_NAME="${RUN_NAME:-flux2_klein_ablation_${EXPERIMENT_MODE}_${SUB_BRANCH_LAYOUT}_pe_${PE_EXCHANGE_REGION}}"
# Deterministic, main-resolution-grouped batch order from precompute_bucket_order.py. REQUIRED
# for TRAIN_BATCH_SIZE>1 on native-aspect data (collate hard-stacks packed_target, so each batch
# must share one main resolution). Auto-detected under bucket_orders/ by the four key dims.
DATASET_STEM="$(basename "${DATASET%.json}")"
BUCKET_ORDER_DIR="${BUCKET_ORDER_DIR:-$ROOT/bucket_orders}"
BUCKET_ORDER_FILE="${BUCKET_ORDER_FILE:-$BUCKET_ORDER_DIR/${DATASET_STEM}_bucket_order_res${RESOLUTION}_ds${DOWNSAMPLE_FACTOR}_bs${TRAIN_BATCH_SIZE}_seed${SEED}.json}"
LONG_PROMPT_PROB="${LONG_PROMPT_PROB:-0.5}"   # random long/short prompt per sample (cache pads both to 512)
SUBJECT_DROP_PROB="${SUBJECT_DROP_PROB:-0.5}" # subject-IMAGE dropout (drops packed_subject cond; sub+PE kept)
PROMPT_DROP_PROB="${PROMPT_DROP_PROB:-0.1}"    # CFG text dropout: replace prompt with encoded empty "" null
# Gradient checkpointing toggle. Default ON (memory-saving). Set GRADIENT_CHECKPOINTING=0 to
# train without it (faster, more activation memory) — used by the no-GC memory probe.
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"

# Accelerate config: defaults to the whole-model compile config (max-autotune-no-cudagraphs).
# suppress_errors=True is baked into the training script, so the GC+compile dynamo bug
# falls back to eager per-subgraph instead of crashing. Override with a *_nocompile.yaml
# config (no dynamo_config block) to run pure eager.
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$ROOT/accelerate_config_4gpu_a.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

# ---- Per-mode flags ---------------------------------------------------------
# Sub layout and PE exchange region are independent. Defaults preserve the historical
# sparse-mask input + mask-only PE exchange used by checkpoint-1400.
# zero_cond_t (t=0 AdaLN for cond tokens) is only wired for hard_exchange — the base and
# soft transformers never receive num_cond_tokens, so it is a no-op elsewhere; pass
# --no-zero_cond_t there to keep the plain transformers from seeing the flag at all.
MODE_FLAGS=()
case "$EXPERIMENT_MODE" in
  no_sub)
    MODE_FLAGS+=(--no-zero_cond_t)
    SUBJECT_DROP_PROB=0   # no subject to drop; drop_subject is auto-disabled for no_sub anyway
    ;;
  sub_no_exchange)
    MODE_FLAGS+=(--no-zero_cond_t)
    ;;
  hard_exchange)
    MODE_FLAGS+=(--zero_cond_t)
    ;;
  soft_exchange)
    MODE_FLAGS+=(--no-zero_cond_t)
    ;;
esac
if [[ "$EXPERIMENT_MODE" != "no_sub" && "$SUB_BRANCH_LAYOUT" == "sparse_mask" ]]; then
  MODE_FLAGS+=(--use_sparse_sub_branch)
fi

CMD=(accelerate launch)
if [[ -f "$ACCELERATE_CONFIG" ]]; then
  CMD+=(--config_file "$ACCELERATE_CONFIG")
  echo "[run] accelerate config: $ACCELERATE_CONFIG"
else
  CMD+=(--num_processes "$NUM_PROCESSES")
  echo "[run] accelerate config not found at $ACCELERATE_CONFIG; using --num_processes $NUM_PROCESSES"
fi

CMD+=(
  "$ROOT/train_flux2_klein_subject_img2img_sub_ids20.py"
  --pretrained_model_name_or_path "$MODEL_ID"
  --output_dir "$OUTPUT_ROOT/$RUN_NAME"
  --dataset "$DATASET"
  --experiment_mode "$EXPERIMENT_MODE"
  --sub_region_mode "${SUB_REGION_MODE:-mask}"
  --pe_exchange_region "$PE_EXCHANGE_REGION"
  --resolution "$RESOLUTION"
  --train_batch_size "${TRAIN_BATCH_SIZE:-1}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-0}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-10}"
  --checkpointing_steps "${CHECKPOINTING_STEPS:-200}"
  --checkpoints_total_limit "${CHECKPOINTS_TOTAL_LIMIT:-5}"
  --mixed_precision "${MIXED_PRECISION:-bf16}"
  --optimizer "${OPTIMIZER:-prodigyplus_schedulefree}"
  --learning_rate "${LEARNING_RATE:-1.0}"
  --sub_loss_weight "${SUB_LOSS_WEIGHT:-1.0}"
  --seed "$SEED"
  --report_to "${REPORT_TO:-wandb}"
  --no-local_files_only
  --long_prompt_prob "$LONG_PROMPT_PROB"
  --subject_drop_prob "$SUBJECT_DROP_PROB"
  --prompt_drop_prob "$PROMPT_DROP_PROB"
  "${MODE_FLAGS[@]}"
)

# Gradient checkpointing (default on). GRADIENT_CHECKPOINTING=0 disables it (no-GC probe).
[[ "$GRADIENT_CHECKPOINTING" == "1" ]] && CMD+=(--gradient_checkpointing)

# Optional smoke-test caps.
[[ -n "${MAX_TRAIN_STEPS:-}" ]]   && CMD+=(--max_train_steps "$MAX_TRAIN_STEPS")
[[ -n "${MAX_TRAIN_SAMPLES:-}" ]] && CMD+=(--max_train_samples "$MAX_TRAIN_SAMPLES")

# Explicit DeepSpeed torch.compile (max-autotune): accelerate's auto dynamo path doesn't fire
# under DeepSpeed, so the script calls engine.compile() itself. Set DEEPSPEED_COMPILE=1 when
# using a ZeRO config (the dynamo_config is intentionally removed from those yamls).
[[ "${DEEPSPEED_COMPILE:-}" == "1" ]] && CMD+=(--deepspeed_compile)
# Optional bf16 gradient accumulation (drops DeepSpeed's ~9GB/GPU fp32 grad-accum buffer).
[[ -n "${GRAD_ACCUM_DTYPE:-}" ]] && CMD+=(--grad_accum_dtype "$GRAD_ACCUM_DTYPE")

if [[ -d "$CACHE_DIR" ]]; then
  CMD+=(--cache_dir "$CACHE_DIR")
  echo "[run] cache: $CACHE_DIR"
else
  echo "[run] ERROR: cache dir not found at $CACHE_DIR — the dataset's image paths are not on this machine." >&2
  exit 1
fi

# Bucket order (main-resolution-grouped). REQUIRED for TRAIN_BATCH_SIZE>1 in ANY mode (incl. no_sub),
# because collate_cached_examples hard-stacks packed_target/packed_main/cond_image_ids, so every
# sample in a batch must share one main resolution. no_sub at bs>1 replays the same order (the sub
# secondary sort is a harmless no-op there); at bs=1 the order is unused (plain shuffle / dynamic sampler).
if [[ -f "$BUCKET_ORDER_FILE" ]]; then
  CMD+=(--bucket_order_file "$BUCKET_ORDER_FILE")
  echo "[run] bucket order: $BUCKET_ORDER_FILE"
elif [[ "$TRAIN_BATCH_SIZE" -gt 1 ]]; then
  echo "[run] ERROR: TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE but no bucket order file at $BUCKET_ORDER_FILE." >&2
  echo "[run]   On native-aspect data, bs>1 needs a main-resolution-grouped order or collate torch.stack crashes." >&2
  echo "[run]   Generate it: python precompute_bucket_order.py --dataset \"$DATASET\" --cache_dir \"$CACHE_DIR\" \\" >&2
  echo "[run]                  --train_batch_size $TRAIN_BATCH_SIZE --seed $SEED --output \"$BUCKET_ORDER_FILE\"" >&2
  exit 1
else
  echo "[run] no bucket order file (bs=1) — plain shuffle / dynamic sampler."
fi

if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  CMD+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

echo "[run] EXPERIMENT_MODE=$EXPERIMENT_MODE  sub_layout=$SUB_BRANCH_LAYOUT  pe_exchange_region=$PE_EXCHANGE_REGION"
echo "[run] output_dir=$OUTPUT_ROOT/$RUN_NAME"
echo "[run] long_prompt_prob=$LONG_PROMPT_PROB  subject_drop_prob=$SUBJECT_DROP_PROB"
echo "[run] launching:"; printf '  %q' "${CMD[@]}"; echo
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[run] DRY_RUN=1; prerequisites and command construction passed."
  exit 0
fi
"${CMD[@]}"
