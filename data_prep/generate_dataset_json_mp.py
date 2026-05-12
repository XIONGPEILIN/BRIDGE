import os
import json
import glob
import numpy as np
import torch
import torch.nn.functional as F
import shutil
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
from torch.multiprocessing import Pool
from multiprocessing import cpu_count
from torchvision.transforms import InterpolationMode

# 配置路径
BASE_DIR = "pico-banana-400k-subject_driven"
AUDIT_DIR = os.path.join(BASE_DIR, "openimages/dino_mask_audit_1")
OUTPUT_JSON = "dataset_qwen_pe_full.json"
REF_GT_DIR = os.path.join(BASE_DIR, "openimages/ref_gt_generated_full")
EDIT_ALIGNED_DIR = os.path.join(BASE_DIR, "openimages/edit_aligned_all")
OLD_MASK_DIR = os.path.join(BASE_DIR, "openimages/ref_gt_generated_all")

# 确保输出目录存在
os.makedirs(REF_GT_DIR, exist_ok=True)

# 建议使用 CPU 以避免多进程 CUDA 问题，除非显存非常充足
# 如果遇到 RuntimeError: CUDA error，请改为 "cpu"
DEVICE = "cpu" 
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_PIXELS = 1048576  # 与训练脚本默认 max_pixels 对齐

def _compute_target_size(width, height, max_pixels=MAX_PIXELS, div=16):
    # 复用训练时的动态分辨率逻辑：先控制总像素，再向下取整到 16 倍数
    if width * height > max_pixels:
        scale = (width * height / max_pixels) ** 0.5
        height = int(height / scale)
        width = int(width / scale)
    height = (height // div) * div
    width = (width // div) * div
    return width, height

def _resize_pil_like_operator(img, target_w, target_h):
    # 与 ImageCropAndResize 的 resize+center_crop 行为一致（双线性）
    scale = max(target_w / img.size[0], target_h / img.size[1])
    new_h, new_w = round(img.size[1] * scale), round(img.size[0] * scale)
    img = TF.resize(img, (new_h, new_w), interpolation=InterpolationMode.BILINEAR )
    img = TF.center_crop(img, (target_h, target_w))
    return img

def _resize_pil_lanczos(img, target_w, target_h):
    # LANCZOS 版本，几何逻辑同上
    scale = max(target_w / img.size[0], target_h / img.size[1])
    new_h, new_w = round(img.size[1] * scale), round(img.size[0] * scale)
    img = TF.resize(img, (new_h, new_w), interpolation=InterpolationMode.LANCZOS)
    img = TF.center_crop(img, (target_h, target_w))
    return img

def _resize_mask_like_operator(mask, target_w, target_h):
    # 与上面一致的几何变换，但掩码用最近邻
    _, _, h, w = mask.shape
    scale = max(target_w / w, target_h / h)
    new_h, new_w = round(h * scale), round(w * scale)
    resized = F.interpolate(mask, size=(new_h, new_w), mode="nearest")
    top = max((new_h - target_h) // 2, 0)
    left = max((new_w - target_w) // 2, 0)
    resized = resized[:, :, top:top + target_h, left:left + target_w]
    return resized

def _load_mask_tensor(path, size=None):
    try:
        m = Image.open(path).convert("L")
        transform = transforms.ToTensor()
        t = transform(m)
        t = t.unsqueeze(0)
        t = t.to(DEVICE)
        if size is not None:
            target_h, target_w = size[1], size[0]
            if t.shape[-2:] != (target_h, target_w):
                t = F.interpolate(t, size=(target_h, target_w), mode='nearest')
        return t
    except:
        return None

def process_one_file(audit_file):
    REL_ROOT = os.path.join(BASE_DIR, "openimages")
    try:
        with open(audit_file, 'r') as f:
            audit_data = json.load(f)
        
        # Load Log
        log_rel_path = audit_data.get("log_path")
        if not log_rel_path: return None
        log_full_path = os.path.join(BASE_DIR, log_rel_path)
        if not os.path.exists(log_full_path): return None

        with open(log_full_path, 'r') as f:
            log_data = json.load(f)

        item_idx = audit_data.get("item_idx")
        bg_rel_path = log_data.get("original_item", {}).get("local_input_image")
        edit_rel_path = log_data.get("original_item", {}).get("output_image")
        if not bg_rel_path or not edit_rel_path: return None
        
        edit_full = os.path.join(BASE_DIR, edit_rel_path)
        if not os.path.exists(edit_full): return None
        bg_full = os.path.join(BASE_DIR, bg_rel_path)
        if not os.path.exists(bg_full): return None

        # Try to load existing mask from OLD_MASK_DIR
        old_mask_filename = f"mask_combined_{item_idx}.png"
        old_mask_path = os.path.join(OLD_MASK_DIR, old_mask_filename)
        
        if not os.path.exists(old_mask_path):
            return None # Skip if no pre-generated mask

        # Load Full Precise Mask
        full_mask_pil = Image.open(old_mask_path).convert("L")
        tgt_w, tgt_h = full_mask_pil.size
        
        # 5. Generate Ref GT, and aligned edit_image
        ref_gt_filename = f"ref_gt_{item_idx}.png"
        ref_gt_full_path = os.path.join(REF_GT_DIR, ref_gt_filename)
        aligned_edit_filename = f"edit_aligned_{item_idx}.png"
        aligned_edit_full_path = os.path.join(EDIT_ALIGNED_DIR, aligned_edit_filename)

        # Save aligned edit_image (source) using LANCZOS (Reuse if exists, else regenerate)
        # Note: If reusing, we assume it matches tgt_w, tgt_h.
        if not os.path.exists(aligned_edit_full_path):
             with Image.open(bg_full).convert("RGB") as bg_img:
                bg_resized = _resize_pil_lanczos(bg_img, tgt_w, tgt_h)
                bg_resized.save(aligned_edit_full_path)
        
        if not os.path.exists(ref_gt_full_path):
            edit_img = Image.open(edit_full).convert("RGB")
            # Resize edit_image to match mask size
            edit_img = _resize_pil_like_operator(edit_img, tgt_w, tgt_h)
            
            # Apply Mask to FULL image
            # Use white background so transparent areas are white instead of black
            ref_gt_img = Image.new("RGB", edit_img.size, (255, 255, 255))
            ref_gt_img.paste(edit_img, (0, 0), mask=full_mask_pil)
            ref_gt_img.save(ref_gt_full_path)

        # 6. Construct Entry
        original_text = log_data.get("original_item", {}).get("text", "")
        if not original_text:
            original_text = log_data.get("instruction", "")
        
        prompt = f"Picture 1 is the image to modify. {original_text}"
        
        bg_rel_path = os.path.relpath(bg_full, REL_ROOT)
        edit_rel_path = os.path.relpath(edit_full, REL_ROOT)
        aligned_edit_rel_path = os.path.relpath(aligned_edit_full_path, REL_ROOT)
        ref_gt_rel_path = os.path.relpath(ref_gt_full_path, REL_ROOT)
        # final_mask_rel_path = os.path.relpath(new_mask_full_path, REL_ROOT)

        return {
            "prompt": prompt,
            "image": edit_rel_path,        # GT
            "edit_image": [aligned_edit_rel_path],   # Source (aligned to target resolution)
            "ref_gt": ref_gt_rel_path,
            # "back_mask": final_mask_rel_path
        }

    except Exception as e:
        return None

def main():
    audit_files = glob.glob(os.path.join(AUDIT_DIR, "*_dino_audit.json"))
    # audit_files = audit_files[:1000]  # limit for quick test run
    print(f"Found {len(audit_files)} audit files. Processing with {cpu_count()} CPUs...")

    # 确保输出目录存在
    os.makedirs(EDIT_ALIGNED_DIR, exist_ok=True)

    # 清空并重建 ref_gt 输出目录，确保重新生成全图尺寸
    if os.path.exists(REF_GT_DIR):
        shutil.rmtree(REF_GT_DIR)
    os.makedirs(REF_GT_DIR, exist_ok=True)
    
    # Use CPU by default for stability with multiprocessing
    with Pool(processes=8) as pool:
        results = list(tqdm(pool.imap(process_one_file, audit_files), total=len(audit_files)))
    
    dataset = [r for r in results if r is not None]

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated dataset with {len(dataset)} entries saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
