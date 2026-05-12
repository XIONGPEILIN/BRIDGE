#!/usr/bin/env python
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.multiprocessing as mp
from queue import Empty

"""
Multi-GPU Image Evaluation Script.
Refactored to use a standard PyTorch Dataset.
"""

# -----------------------------------------------------------------------------
# Dataset Definition
# -----------------------------------------------------------------------------

class EvalDataset(Dataset):
    def __init__(self, json_path, pred_dir, gt_base_dir):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.pred_dir = pred_dir
        self.gt_base_dir = gt_base_dir
        self.valid_indices = self._filter_valid()

    def _filter_valid(self):
        valid = []
        for i, entry in enumerate(self.data):
            if not entry.get('edit_image'): continue
            fname = os.path.basename(entry['edit_image'][0])
            p_path = os.path.join(self.pred_dir, fname)
            g_path = os.path.join(self.gt_base_dir, entry['image'])
            
            # Check if both pred and gt exist
            if os.path.exists(p_path) and os.path.exists(g_path):
                valid.append(i)
        return valid

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        entry = self.data[real_idx]
        
        fname = os.path.basename(entry['edit_image'][0])
        p_path = os.path.join(self.pred_dir, fname)
        g_path = os.path.join(self.gt_base_dir, entry['image'])
        m_path = os.path.join(self.gt_base_dir, entry['back_mask']) if entry.get('back_mask') else None
        
        # Load Images
        img_p = Image.open(p_path).convert("RGB")
        img_g = Image.open(g_path).convert("RGB")
        
        # Align sizes
        if img_p.size != img_g.size:
            img_p = img_p.resize(img_g.size, Image.LANCZOS)
            
        mask_p = None
        if m_path and os.path.exists(m_path):
            mask_p = Image.open(m_path).convert("L").resize(img_p.size, Image.NEAREST)
            
        return {
            "filename": fname,
            "img_p": img_p,
            "img_g": img_g,
            "mask_p": mask_p,
            "prompt": entry.get('prompt', ""),
            "global_caption": entry.get("global_caption", ""),
            "local_caption": entry.get("local_caption", "")
        }

# -----------------------------------------------------------------------------
# Worker Function
# -----------------------------------------------------------------------------

def gpu_worker(gpu_id, task_indices, result_queue, args, dataset):
    import torch
    from torchvision import transforms
    import torch.nn.functional as F
    from transformers import AutoModel, AutoProcessor, pipeline
    from dreamsim import dreamsim as load_dreamsim_lib

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0" 

    print(f"[GPU {gpu_id}] Loading models (SigLIP2, DINOv3, DreamSim)...")
    try:
        # SigLIP2
        siglip_ckpt = "google/siglip2-large-patch16-512"
        siglip_model = AutoModel.from_pretrained(siglip_ckpt, torch_dtype=torch.bfloat16).to(device).eval()
        siglip_processor = AutoProcessor.from_pretrained(siglip_ckpt)

        # DINOv3
        dinov3_pipe = pipeline(
            model="facebook/dinov3-vit7b16-pretrain-lvd1689m",
            task="image-feature-extraction",
            device=0, 
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.bfloat16}
        )

        # DreamSim
        ds_res = load_dreamsim_lib(pretrained=True)
        ds_model = ds_res[0] if isinstance(ds_res, tuple) else ds_res
        ds_preprocess = ds_res[1] if isinstance(ds_res, tuple) else ds_model.preprocess
        ds_model = ds_model.to(device).eval()
    except Exception as e:
        print(f"[GPU {gpu_id}] Model Init Failed: {e}")
        return

    pixel_transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])

    # Pre-extract scale and bias
    logit_scale = siglip_model.logit_scale.exp().item()
    logit_bias = siglip_model.logit_bias.item()

    def compute_siglip2(img, gt, prompt=None):
        inputs_img = siglip_processor(images=img, return_tensors="pt").to(device)
        inputs_gt = siglip_processor(images=gt, return_tensors="pt").to(device)
        if siglip_model.dtype == torch.bfloat16:
            inputs_img["pixel_values"] = inputs_img["pixel_values"].to(torch.bfloat16)
            inputs_gt["pixel_values"] = inputs_gt["pixel_values"].to(torch.bfloat16)
        
        with torch.inference_mode():
            emb_img = siglip_model.get_image_features(**inputs_img)
            emb_gt = siglip_model.get_image_features(**inputs_gt)
            emb_img = emb_img / emb_img.norm(dim=-1, keepdim=True)
            emb_gt = emb_gt / emb_gt.norm(dim=-1, keepdim=True)
            sim_i = torch.sum(emb_img.float() * emb_gt.float(), dim=-1).item()
            
            sim_t = np.nan
            if prompt:
                # Use max_length=64 as per standard SigLIP config
                inputs_text = siglip_processor(text=prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=64).to(device)
                emb_text = siglip_model.get_text_features(**inputs_text)
                emb_text = emb_text / emb_text.norm(dim=-1, keepdim=True)
                
                # Apply Sigmoid(Logits) transform
                cosine_sim = torch.sum(emb_img.float() * emb_text.float(), dim=-1).item()
                logits = cosine_sim * logit_scale + logit_bias
                sim_t = torch.sigmoid(torch.tensor(logits)).item()
        return sim_i, sim_t

    def compute_dino(img, gt):
        with torch.inference_mode():
            feat_img = dinov3_pipe(img)
            feat_gt = dinov3_pipe(gt)
            t_img = torch.tensor(feat_img[0]).mean(dim=0 if torch.tensor(feat_img[0]).ndim==2 else 1, keepdim=True)
            t_gt = torch.tensor(feat_gt[0]).mean(dim=0 if torch.tensor(feat_gt[0]).ndim==2 else 1, keepdim=True)
            return F.cosine_similarity(t_img.float(), t_gt.float(), dim=-1).item()

    def compute_ds(img, gt):
        with torch.inference_mode():
            t_p = ds_preprocess(img).to(device)
            t_g = ds_preprocess(gt).to(device)
            return ds_model(t_p, t_g).item()

    def extract_bbox(mask_pil):
        mask_np = np.array(mask_pil)
        rows = np.any(mask_np > 128, axis=1); cols = np.any(mask_np > 128, axis=0)
        if not np.any(rows) or not np.any(cols): return None
        rmin, rmax = np.where(rows)[0][[0, -1]]; cmin, cmax = np.where(cols)[0][[0, -1]]
        return (cmin, rmin, cmax + 1, rmax + 1)

    for idx in task_indices:
        try:
            item = dataset[idx]
            filename, img_p, img_g, mask_p, prompt = item["filename"], item["img_p"], item["img_g"], item["mask_p"], item["prompt"]
            global_cap = item.get("global_caption", "")
            local_cap = item.get("local_caption", "")
            
            row = {"filename": filename, "prompt": prompt, "global_caption": global_cap, "local_caption": local_cap}
            
            # Use global_caption if available, else fallback to prompt
            text_for_global = global_cap if global_cap else prompt
            
            # Metrics calculation logic
            row["siglip2_i"], row["siglip2_t"] = compute_siglip2(img_p, img_g, text_for_global)
            # Alias for clarity
            row["siglip2_t_global"] = row["siglip2_t"]
            
            row["dino"] = compute_dino(img_p, img_g)
            row["dreamsim"] = compute_ds(img_p, img_g)
            
            if mask_p:
                bbox = extract_bbox(mask_p)
                if bbox:
                    crop_p = img_p.crop(bbox)
                    crop_g = img_g.crop(bbox)
                    
                    row["siglip2_i_bbox"], _ = compute_siglip2(crop_p, crop_g)
                    
                    # Compute SigLIP2_T_Local if local_caption exists
                    if local_cap:
                        # Original: _, row["siglip2_t_local"] = compute_siglip2(crop_p, crop_g, local_cap)
                        # We need manual calculation here to reuse crop logic but apply new formula
                        # Actually compute_siglip2 already does the transform now!
                        # But wait, compute_siglip2 takes (img, gt, prompt).
                        # We can just reuse it:
                        _, row["siglip2_t_local"] = compute_siglip2(crop_p, crop_g, local_cap)
                    else:
                        row["siglip2_t_local"] = 0.0

                    row["dino_bbox"] = compute_dino(crop_p, crop_g)
                    row["dreamsim_bbox"] = compute_ds(crop_p, crop_g)
                
                black = Image.new("RGB", img_p.size, (0, 0, 0))
                row["siglip2_i_mask"], _ = compute_siglip2(Image.composite(img_p, black, mask_p), Image.composite(img_g, black, mask_p))
                row["dino_mask"] = compute_dino(Image.composite(img_p, black, mask_p), Image.composite(img_g, black, mask_p))
                row["dreamsim_mask"] = compute_ds(Image.composite(img_p, black, mask_p), Image.composite(img_g, black, mask_p))

            # L1/L2 Calculation at Full Resolution
            with torch.inference_mode():
                # Convert PIL images directly to tensor (no resize)
                # Ensure values are [0, 1] for L1/L2 consistency
                to_tensor = transforms.ToTensor()
                t_p = to_tensor(img_p).to(device)
                t_g = to_tensor(img_g).to(device)
                
                diff_sq = (t_p - t_g) ** 2
                diff_abs = torch.abs(t_p - t_g)
                
                row["l2"] = torch.mean(diff_sq).item()
                row["l1"] = torch.mean(diff_abs).item()

                if mask_p:
                    # Use original mask (aligned with image size in __getitem__)
                    t_m = to_tensor(mask_p).to(device) # (1, H, W)
                    
                    # Inside Mask
                    m_sum = t_m.sum()
                    if m_sum > 0:
                        row["l2_in_mask"] = (diff_sq * t_m).sum().item() / (m_sum.item() * 3)
                        row["l1_in_mask"] = (diff_abs * t_m).sum().item() / (m_sum.item() * 3)
                    
                    # Outside Mask
                    t_m_inv = 1.0 - t_m
                    m_inv_sum = t_m_inv.sum()
                    if m_inv_sum > 0:
                        row["l2_out_mask"] = (diff_sq * t_m_inv).sum().item() / (m_inv_sum.item() * 3)
                        row["l1_out_mask"] = (diff_abs * t_m_inv).sum().item() / (m_inv_sum.item() * 3)

                    # Inside/Outside BBox
                    bbox = extract_bbox(mask_p) # Uses original PIL mask
                    if bbox:
                        cmin, rmin, cmax, rmax = bbox
                        t_b = torch.zeros_like(t_m)
                        t_b[:, rmin:rmax, cmin:cmax] = 1.0
                        
                        b_sum = t_b.sum()
                        if b_sum > 0:
                            row["l2_in_bbox"] = (diff_sq * t_b).sum().item() / (b_sum.item() * 3)
                            row["l1_in_bbox"] = (diff_abs * t_b).sum().item() / (b_sum.item() * 3)
                        
                        t_b_inv = 1.0 - t_b
                        b_inv_sum = t_b_inv.sum()
                        if b_inv_sum > 0:
                            row["l2_out_bbox"] = (diff_sq * t_b_inv).sum().item() / (b_inv_sum.item() * 3)
                            row["l1_out_bbox"] = (diff_abs * t_b_inv).sum().item() / (b_inv_sum.item() * 3)

            result_queue.put(row)
        except Exception as e:
            print(f"[GPU {gpu_id}] Error {idx}: {e}")
            result_queue.put({"filename": str(idx), "error": str(e)})

    print(f"[GPU {gpu_id}] Finished.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gt_base_dir", required=True)
    parser.add_argument("--output_csv", default="evaluation_results.csv")
    parser.add_argument("--gpu_ids", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()

    # Init Dataset
    dataset = EvalDataset(args.json_path, args.pred_dir, args.gt_base_dir)
    num_tasks = len(dataset)
    print(f"Total tasks in dataset: {num_tasks}")
    if num_tasks == 0: return

    # Task distribution
    gpu_list = [int(x.strip()) for x in args.gpu_ids.split(",")]
    num_gpus = len(gpu_list)
    indices = np.array_split(range(num_tasks), num_gpus)

    ctx = mp.get_context('spawn')
    r_q = ctx.Queue()
    processes = []
    for i, gid in enumerate(gpu_list):
        p = ctx.Process(target=gpu_worker, args=(gid, indices[i].tolist(), r_q, args, dataset))
        p.start(); processes.append(p)

    results = []
    with tqdm(total=num_tasks, desc="Evaluating") as pbar:
        while len(results) < num_tasks:
            try:
                res = r_q.get(timeout=5)
                if "error" not in res: results.append(res)
                else: results.append(None)
                pbar.update(1)
            except Empty:
                if not any(p.is_alive() for p in processes): break

    [p.join() for p in processes]
    valid = [r for r in results if r is not None]
    if valid:
        df = pd.DataFrame(valid); df.to_csv(args.output_csv, index=False)
        m_map = {"siglip2_i": "SigLIP2_I", "siglip2_t": "SigLIP2_T", "siglip2_t_local": "SigLIP2_T_Local", "dino": "DINO", "dreamsim": "DS",
                 "siglip2_i_bbox": "SigLIP2_I_BBox", "dino_bbox": "DINO_BBox", "dreamsim_bbox": "DS_BBox",
                 "siglip2_i_mask": "SigLIP2_I_Mask", "dino_mask": "DINO_Mask", "dreamsim_mask": "DS_Mask",
                 "l2": "MSE", "l1": "MAE",
                 "l2_in_mask": "MSE_InMask", "l1_in_mask": "MAE_InMask",
                 "l2_out_mask": "MSE_OutMask", "l1_out_mask": "MAE_OutMask",
                 "l2_in_bbox": "MSE_InBBox", "l1_in_bbox": "MAE_InBBox",
                 "l2_out_bbox": "MSE_OutBBox", "l1_out_bbox": "MAE_OutBBox"}
        print("\n" + "="*40 + "\nSUMMARY\n" + "="*40)
        for k, v in m_map.items():
            if k in df.columns: print(f"{v}: {df[k].mean():.4f}")
        with open(os.path.splitext(args.output_csv)[0] + "_summary.json", "w") as f:
            json.dump({m_map[k]: float(df[k].mean()) for k, v in m_map.items() if k in df.columns}, f, indent=4)

if __name__ == "__main__":
    main()
