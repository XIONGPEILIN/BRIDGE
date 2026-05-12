import os
import torch
import argparse
import torch.multiprocessing as mp
from tqdm import tqdm
from diffsynth.models.qwen_image_vae import QwenImageVAE
from diffsynth.core import ModelConfig, load_state_dict
import math

def worker(gpu_id, all_chunks):
    # Fix: Retrieve the specific chunk for this GPU
    if gpu_id >= len(all_chunks):
        return
    file_subset = all_chunks[gpu_id]
    
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading VAE...")
    
    try:
        # VAE config path is assumed to be ready
        vae_config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors")
        vae_config.download_if_necessary() # Fix: Ensure path is resolved in worker process
        state_dict = load_state_dict(vae_config.path)
        vae = QwenImageVAE().to(device=device, dtype=torch.bfloat16)
        vae.load_state_dict(state_dict)
        vae.eval()
    except Exception as e:
        print(f"[GPU {gpu_id}] Failed to load VAE: {e}")
        return

    print(f"[GPU {gpu_id}] Processing {len(file_subset)} files...")
    
    # Use position to avoid progress bar overlapping
    for file_path in tqdm(file_subset, position=gpu_id, desc=f"GPU {gpu_id}", leave=False):
        try:
            # Load data
            data = torch.load(file_path, map_location="cpu", weights_only=False) # Fix: Allow PIL images
            inputs_shared, inputs_posi, inputs_nega = data
            
            # Skip update if 'sub_noise' is already present and valid? 
            # Uncomment below if you want to resume or skip updated files
            # if "sub_noise" in inputs_shared: continue

            if "sub_input_latents" in inputs_shared and inputs_shared["sub_input_latents"] is not None:
                sub_latents = inputs_shared["sub_input_latents"]
                b, c, h, w = sub_latents.shape
                
                # Qwen VAE downsamples by 8
                pixel_h = h * 8
                pixel_w = w * 8
                
                # Create white image [B, 3, H, W]
                white_image = torch.ones((b, 3, pixel_h, pixel_w), device=device, dtype=torch.bfloat16)
                
                with torch.no_grad():
                    # tiled=False is usually fine for sub-crops, but if crop is huge it might OOM.
                    # Given it's "sub", it should be small.
                    sub_noise = vae.encode(white_image, tiled=False)
                
                inputs_shared["sub_noise"] = sub_noise.cpu()
                
                # Atomic save
                tmp_path = file_path + ".tmp"
                torch.save((inputs_shared, inputs_posi, inputs_nega), tmp_path)
                os.replace(tmp_path, file_path)
                
        except Exception as e:
            print(f"[GPU {gpu_id}] Error processing {file_path}: {e}")

    print(f"[GPU {gpu_id}] Finished.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="./data/Qwen-Image-Edit-2511_lora-rank512-split-cache")
    parser.add_argument("--num_gpus", type=int, default=8)
    args = parser.parse_args()
    
    # Pre-check VAE download in main process to avoid race conditions
    print("Checking VAE model files...")
    vae_config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors")
    vae_config.download_if_necessary()
    
    # 1. Collect all files
    print(f"Scanning cache directory: {args.cache_dir}")
    file_list = []
    for root, dirs, files in os.walk(args.cache_dir):
        for file in files:
            if file.endswith(".pth"):
                file_list.append(os.path.join(root, file))
    
    total_files = len(file_list)
    print(f"Found {total_files} cache files.")
    if total_files == 0:
        return

    # 2. Split work
    # Handle edge case where total_files < num_gpus
    num_chunks = min(args.num_gpus, total_files)
    chunk_size = math.ceil(total_files / num_chunks)
    chunks = [file_list[i:i + chunk_size] for i in range(0, total_files, chunk_size)]
    
    print(f"Spawning {len(chunks)} workers processing {total_files} files...")
    
    # Pass 'chunks' as a tuple argument. 
    # worker will be called as worker(gpu_id, chunks)
    mp.spawn(worker, args=(chunks,), nprocs=len(chunks), join=True)

if __name__ == "__main__":
    main()
