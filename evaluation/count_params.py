import torch
from safetensors.torch import load_file
from pathlib import Path

ckpt_path = "train/Qwen-Image-Edit-2511_lora-rank512-cfg/step-28000.safetensors"
print(f"Loading metadata from {ckpt_path}...")

# Use safe_open to avoid loading all tensors into memory at once
from safetensors import safe_open

ste_params = 0
lora_params = 0
total_params = 0

with safe_open(ckpt_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        tensor_slice = f.get_slice(key)
        shape = tensor_slice.get_shape()
        numel = 1
        for dim in shape:
            numel *= dim
        
        total_params += numel
        
        if key.startswith("pipe.ste."):
            ste_params += numel
            if ste_params <= numel * 5: # Print first few
                print(f"  STE key example: {key}, shape: {shape}")
        elif "lora_" in key:
            lora_params += numel
            if lora_params <= numel * 5: # Print first few
                print(f"  LoRA key example: {key}, shape: {shape}")

print(f"STE parameters: {ste_params:,} ({ste_params/1e6:.2f} M)")
print(f"LoRA parameters: {lora_params:,} ({lora_params/1e6:.2f} M)")
print(f"Total parameters in file: {total_params:,} ({total_params/1e6:.2f} M)")
