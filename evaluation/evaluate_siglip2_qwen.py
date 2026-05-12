import os
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

class SigLIPQwenDataset(Dataset):
    def __init__(self, json_path, pred_dir, gt_base_dir):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.pred_dir = pred_dir
        self.gt_base_dir = gt_base_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        
        # 1. Get Prediction Image (from specific model directory)
        # We take the basename of the original edit_image path
        p_rel_orig = entry['edit_image'][0] if isinstance(entry['edit_image'], list) else entry['edit_image']
        fname = os.path.basename(p_rel_orig)
        p_path = os.path.join(self.pred_dir, fname)
        
        # 2. Get Ground Truth and Mask
        # USER CORRECTION: 'image' is the actual GT image.
        g_rel = entry['image']
        m_rel = entry.get('back_mask')
        g_path = os.path.join(self.gt_base_dir, g_rel)
        
        if not os.path.exists(p_path):
            # Try recursive search or skip
            return None
            
        try:
            img_p = Image.open(p_path).convert("RGB")
            img_g = Image.open(g_path).convert("RGB")
            
            # Align sizes (important for I-I and Masking)
            if img_p.size != img_g.size:
                img_p = img_p.resize(img_g.size, Image.LANCZOS)
                
            mask_p = None
            if m_rel:
                m_path = os.path.join(self.gt_base_dir, m_rel)
                if os.path.exists(m_path):
                    mask_p = Image.open(m_path).convert("L").resize(img_p.size, Image.NEAREST)
            
            return {
                "filename": fname,
                "img_p": img_p,
                "img_g": img_g,
                "mask_p": mask_p,
                "global_caption": entry.get("global_caption", ""),
                "local_caption": entry.get("local_caption", "")
            }
        except Exception:
            return None

def gpu_worker(gpu_id, task_indices, result_queue, dataset):
    from transformers import AutoModel, AutoProcessor
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"

    siglip_ckpt = "google/siglip2-large-patch16-512"
    model = AutoModel.from_pretrained(siglip_ckpt, torch_dtype=torch.bfloat16).to(device).eval()
    processor = AutoProcessor.from_pretrained(siglip_ckpt)

    # Pre-extract scale and bias for manual calculation
    # SigLIP stores scale as learnable parameter, usually it's log(scale)
    # Actually HuggingFace SiglipModel implementation:
    # logits = (image_embeds @ text_embeds.T) * torch.exp(self.logit_scale) + self.logit_bias
    logit_scale = model.logit_scale.exp().item()
    logit_bias = model.logit_bias.item()

    def get_emb(pixel_values=None, input_ids=None):
        with torch.inference_mode():
            if pixel_values is not None:
                e = model.get_image_features(pixel_values.to(torch.bfloat16))
            else:
                e = model.get_text_features(input_ids)
            return e / e.norm(dim=-1, keepdim=True)

    def extract_bbox(mask_pil):
        mask_np = np.array(mask_pil)
        rows = np.any(mask_np > 128, axis=1); cols = np.any(mask_np > 128, axis=0)
        if not np.any(rows) or not np.any(cols): return None
        rmin, rmax = np.where(rows)[0][[0, -1]]; cmin, cmax = np.where(cols)[0][[0, -1]]
        return (cmin, rmin, cmax + 1, rmax + 1)

    for idx in task_indices:
        item = dataset[idx]
        if item is None:
            result_queue.put({"filename": "not_found_or_error"})
            continue
        
        res = {"filename": item["filename"], "siglip2_i": 0, "siglip2_t_global": 0, "siglip2_t_local": 0}
        img_p, img_g, mask_p = item["img_p"], item["img_g"], item["mask_p"]
        g_cap, l_cap = item["global_caption"], item["local_caption"]

        # 1. Image-Image Score (Pred vs GT) - Keep Cosine for I-I consistency
        in_p = processor(images=img_p, return_tensors="pt").to(device)
        in_g = processor(images=img_g, return_tensors="pt").to(device)
        e_p = get_emb(pixel_values=in_p.pixel_values)
        e_g = get_emb(pixel_values=in_g.pixel_values)
        res["siglip2_i"] = torch.sum(e_p * e_g).item()

        # 2. Global Text Score (Pred vs Global Caption) - Use Sigmoid(Logits)
        if g_cap:
            in_t = processor(text=g_cap, return_tensors="pt", padding="max_length", truncation=True, max_length=64).to(device)
            e_t = get_emb(input_ids=in_t.input_ids)
            # Calculate Cosine
            cosine_sim = torch.sum(e_p * e_t).item()
            # Apply SigLIP transform
            logits = cosine_sim * logit_scale + logit_bias
            res["siglip2_t_global"] = torch.sigmoid(torch.tensor(logits)).item()

        # 3. Local Text Score (Pred Crop vs Local Caption) - Use Sigmoid(Logits)
        if l_cap and mask_p:
            bbox = extract_bbox(mask_p)
            if bbox:
                crop_p = img_p.crop(bbox)
                in_p_l = processor(images=crop_p, return_tensors="pt").to(device)
                e_p_l = get_emb(pixel_values=in_p_l.pixel_values)
                
                in_t_l = processor(text=l_cap, return_tensors="pt", padding="max_length", truncation=True, max_length=64).to(device)
                e_t_l = get_emb(input_ids=in_t_l.input_ids)
                
                cosine_sim_l = torch.sum(e_p_l * e_t_l).item()
                logits_l = cosine_sim_l * logit_scale + logit_bias
                res["siglip2_t_local"] = torch.sigmoid(torch.tensor(logits_l)).item()

        result_queue.put(res)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", default="dataset_qwen_pe_top1000_captioned.json")
    parser.add_argument("--pred_dir", required=True, help="Directory containing model generated images")
    parser.add_argument("--gt_base_dir", default="/host/ssd2/xiong-p/qwenpe/pico-banana-400k-subject_driven/openimages")
    parser.add_argument("--output_csv", help="Output CSV path. If not set, defaults to pred_dir/siglip2_qwen_eval.csv")
    parser.add_argument("--gpu_ids", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()

    if not args.output_csv:
        args.output_csv = os.path.join(args.pred_dir, "siglip2_qwen_eval.csv")

    dataset = SigLIPQwenDataset(args.json_path, args.pred_dir, args.gt_base_dir)
    gpu_list = [int(x.strip()) for x in args.gpu_ids.split(",")]
    indices = np.array_split(range(len(dataset)), len(gpu_list))

    ctx = mp.get_context('spawn')
    r_q = ctx.Queue()
    processes = []
    for i, gid in enumerate(gpu_list):
        p = ctx.Process(target=gpu_worker, args=(gid, indices[i].tolist(), r_q, dataset))
        p.start(); processes.append(p)

    results = []
    for _ in tqdm(range(len(dataset)), desc=f"Eval {os.path.basename(args.pred_dir)}"):
        results.append(r_q.get())

    [p.join() for p in processes]
    valid = [r for r in results if r.get("filename") != "not_found_or_error"]
    
    if valid:
        df = pd.DataFrame(valid)
        df.to_csv(args.output_csv, index=False)
        print(f"\n--- Results for {args.pred_dir} ---")
        print(f"Mean SigLIP2_I: {df['siglip2_i'].mean():.4f}")
        print(f"Mean SigLIP2_T_Global: {df['siglip2_t_global'].mean():.4f}")
        print(f"Mean SigLIP2_T_Local: {df['siglip2_t_local'].mean():.4f}")
    else:
        print(f"No valid images found in {args.pred_dir}")

if __name__ == "__main__":
    main()