#!/bin/bash
export WANDB_PROJECT="${WANDB_PROJECT:-qwen-image}"
export WANDB_NAME="${WANDB_NAME:-Qwen-Image-Edit-LBM}"
export WANDB_MODE=online

# Use existing cache
CACHE_PATH="./data/Qwen-Image-Edit-2511_lora-rank512-split-cache"
OUTPUT_PATH="./train/Qwen-Image-Edit-LBM_lora-rank512"

# Phase 2: Training with LBM task
CUDA_VISIBLE_DEVICES=7  accelerate launch  DiffSynth-Studio/examples/qwen_image/model_training/train.py \
  --dataset_base_path "$CACHE_PATH" \
  --data_file_keys "image,edit_image,ref_gt,back_mask" \
  --extra_inputs "edit_image,ref_gt,back_mask" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit-2511:transformer/diffusion_pytorch_model*.safetensors" \
  --learning_rate 1 \
  --num_epochs 100 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$OUTPUT_PATH" \
  --lora_base_model "dit" \
  --lora_checkpoint "train/Qwen-Image-Edit-2511_lora-rank512-cfg/step-28000.safetensors" \
  --trainable_models "ste" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.0.proj,img_mlp.net.2,img_mod.1,txt_mlp.net.0.proj,txt_mlp.net.2,txt_mod.1,img_in,txt_in,proj_out" \
  --lora_rank 512 \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --task "lbm:train" \
  --save_steps 2000 \
  --cfg_drop_prob 0.1 \
  --disable_epoch_resume \
  --gradient_accumulation_steps 32 \
  --zero_cond_t \
  --use_gradient_checkpointing \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_name "$WANDB_NAME"
