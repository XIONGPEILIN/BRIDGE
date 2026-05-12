import os
import json
import glob
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- Configuration ---
BASE_DIR = "pico-banana-400k-subject_driven"
AUDIT_DIR = os.path.join(BASE_DIR, "openimages/dreamsim_mask_audit_mask")
OUTPUT_JSON = "dataset_qwen_pe_fixed.json"

# Output Directories
FIXED_IMG_DIR = os.path.join(BASE_DIR, "openimages/fixed_images")
REF_GT_DIR = os.path.join(BASE_DIR, "openimages/ref_gt_fixed")
REF_GT_CROP_DIR = os.path.join(BASE_DIR, "openimages/ref_gt_fixed_crop")
MASK_DIR = os.path.join(BASE_DIR, "openimages/fixed_masks") 
TARGET_DIR = os.path.join(BASE_DIR, "openimages/target_images")

MAX_PIXELS = 1048576 

# Ensure directories exist
os.makedirs(FIXED_IMG_DIR, exist_ok=True)
os.makedirs(REF_GT_DIR, exist_ok=True)
os.makedirs(REF_GT_CROP_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(TARGET_DIR, exist_ok=True)

def guided_filter(I, p, r, eps):
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    q = mean_a * I + mean_b
    return q

def _compute_target_size(width, height, max_pixels=MAX_PIXELS, div=16):
    if width * height > max_pixels:
        scale = (width * height / max_pixels) ** 0.5
        height = int(height / scale)
        width = int(width / scale)
    height = (height // div) * div
    width = (width // div) * div
    return width, height

def process_one_file(json_path):
    REL_ROOT = os.path.join(BASE_DIR, "openimages")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        item_idx = data.get("item_idx", "unknown")
        before_rel = data.get("before_image")
        after_rel = data.get("after_image")
        log_rel = data.get("log_path")

        if not before_rel or not after_rel: return None

        before_path = os.path.join(BASE_DIR, before_rel)
        after_path = os.path.join(BASE_DIR, after_rel)
        mask_path = json_path.replace("_dreamsim_audit.json", "_bg_check_mask.png")
        log_path = os.path.join(BASE_DIR, log_rel) if log_rel else None

        # 1. Fetch short prompt
        text_prompt = ""
        if log_path and os.path.exists(log_path):
            with open(log_path, 'r') as f_log:
                log_data = json.load(f_log)
            text_prompt = log_data.get("original_item", {}).get("summarized_text", "")
            if not text_prompt: text_prompt = log_data.get("instruction", "")

        # 2. Load & Resize Images
        img_before = Image.open(before_path).convert("RGB")
        img_after = Image.open(after_path).convert("RGB")
        img_mask = Image.open(mask_path).convert("L")

        target_w, target_h = _compute_target_size(img_after.size[0], img_after.size[1])
        img_before = img_before.resize((target_w, target_h), Image.Resampling.LANCZOS)
        img_after = img_after.resize((target_w, target_h), Image.Resampling.LANCZOS)
        img_mask = img_mask.resize((target_w, target_h), Image.Resampling.NEAREST)

        # 3. Guided Filter for Background Fixing
        # In 'p', 1.0 is Background, 0.0 is Object (Guided Filter logic)
        input_mask = np.array(img_mask).astype(np.float64) / 255.0 
        kernel = np.ones((5, 5), np.uint8)
        mask_inner_u8 = cv2.dilate((input_mask * 255).astype(np.uint8), kernel, iterations=9)
        mask_outer_u8 = cv2.erode((input_mask * 255).astype(np.uint8), kernel, iterations=9)
        
        p = input_mask.copy()
        p[mask_outer_u8 > 127] = 1.0 # Background
        p[mask_inner_u8 < 127] = 0.0 # Object

        guide_before = np.array(img_before.convert("L")).astype(np.float64) / 255.0
        guide_after = np.array(img_after.convert("L")).astype(np.float64) / 255.0
        refined_mask = np.clip(np.maximum(guided_filter(guide_before, p, 16, 1e-3), guided_filter(guide_after, p, 16, 1e-3)), 0, 1)

        # 4. Generate Fixed Image (Mask*Before + (1-Mask)*After)
        arr_before, arr_after = np.array(img_before).astype(float), np.array(img_after).astype(float)
        arr_bg_mask = np.expand_dims(refined_mask, axis=2)
        img_fixed = Image.fromarray(np.uint8(np.clip(arr_bg_mask * arr_before + (1.0 - arr_bg_mask) * arr_after, 0, 255)))

        # 5. Generate Ref GT (Object on White)
        # Note: obj_mask is 1.0 for Object, 0.0 for Background
        obj_mask = refined_mask
        arr_obj_mask = np.expand_dims(obj_mask, axis=2)
        white_bg = np.ones_like(arr_after) * 255.0
        img_ref_gt = Image.fromarray(np.uint8(np.clip(arr_after * arr_obj_mask + white_bg * (1.0 - arr_obj_mask), 0, 255)))

        # 6. Crop Ref GT
        bbox_mask = Image.fromarray((obj_mask * 255).astype(np.uint8))
        bbox = bbox_mask.getbbox()
        if bbox:
            l, u, r, b = bbox
            nl, nu, nr, nb = (l//16)*16, (u//16)*16, ((r+15)//16)*16, ((b+15)//16)*16
            img_ref_gt_crop = img_ref_gt.crop((max(0, nl), max(0, nu), min(target_w, nr), min(target_h, nb)))
        else:
            img_ref_gt_crop = img_ref_gt

        # 7. Save
        paths = {
            "fixed": os.path.join(FIXED_IMG_DIR, f"fixed_{item_idx}.png"),
            "ref_gt": os.path.join(REF_GT_DIR, f"ref_gt_{item_idx}.png"),
            "ref_gt_crop": os.path.join(REF_GT_CROP_DIR, f"ref_gt_crop_{item_idx}.png"),
            "mask": os.path.join(MASK_DIR, f"mask_{item_idx}.png"),
            "target": os.path.join(TARGET_DIR, f"target_{item_idx}.png")
        }
        img_fixed.save(paths["fixed"])
        img_ref_gt.save(paths["ref_gt"])
        img_ref_gt_crop.save(paths["ref_gt_crop"])
        Image.fromarray((obj_mask * 255).astype(np.uint8)).save(paths["mask"])
        img_after.save(paths["target"])

        return {
            "prompt": f"Picture 1 is the image to modify. {text_prompt}",
            "image": os.path.relpath(paths["target"], REL_ROOT),
            "edit_image": [os.path.relpath(paths["fixed"], REL_ROOT)],
            "ref_gt": os.path.relpath(paths["ref_gt"], REL_ROOT),
            "ref_gt_crop": os.path.relpath(paths["ref_gt_crop"], REL_ROOT),
            "back_mask": os.path.relpath(paths["mask"], REL_ROOT)
        }
    except Exception: return None

def main():
    json_files = glob.glob(os.path.join(AUDIT_DIR, "*.json"))
    print(f"Processing {len(json_files)} files...")
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_one_file, json_files), total=len(json_files)))
    dataset = [r for r in results if r is not None]
    with open(OUTPUT_JSON, 'w') as f: json.dump(dataset, f, indent=2)
    print(f"Done. Saved to {OUTPUT_JSON}")

if __name__ == "__main__": 
    main()