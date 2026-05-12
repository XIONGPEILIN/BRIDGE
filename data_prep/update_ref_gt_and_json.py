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
OUTPUT_JSON = "dataset_qwen_pe_fixed_updated.json"

# Output Directories (Targets to update)
REF_GT_DIR = os.path.join(BASE_DIR, "openimages/ref_gt_fixed")
REF_GT_CROP_DIR = os.path.join(BASE_DIR, "openimages/ref_gt_fixed_crop")

# Input Directories (Already generated/Existing)
MASK_DIR = os.path.join(BASE_DIR, "openimages/fixed_masks") 
FIXED_IMG_DIR = os.path.join(BASE_DIR, "openimages/fixed_images")
TARGET_DIR = os.path.join(BASE_DIR, "openimages/target_images")

# Ensure Output directories exist
os.makedirs(REF_GT_DIR, exist_ok=True)
os.makedirs(REF_GT_CROP_DIR, exist_ok=True)

def process_one_update(json_path):
    REL_ROOT = os.path.join(BASE_DIR, "openimages")
    try:
        # 1. Load Metadata
        with open(json_path, 'r') as f:
            audit_data = json.load(f)

        item_idx = audit_data.get("item_idx", "unknown")
        after_rel = audit_data.get("after_image")
        
        # Fetch prompt from log_path
        text_prompt = ""
        log_rel_path = audit_data.get("log_path")
        if log_rel_path:
            log_full_path = os.path.join(BASE_DIR, log_rel_path)
            if os.path.exists(log_full_path):
                with open(log_full_path, 'r') as f_log:
                    log_data = json.load(f_log)
                text_prompt = log_data.get("original_item", {}).get("summarized_text", "")
                if not text_prompt:
                    text_prompt = log_data.get("instruction", "")

        if not after_rel:
            return None
        
        after_path = os.path.join(BASE_DIR, after_rel)

        # 2. Define paths for EXISTING generated files
        mask_name = f"mask_{item_idx}.png"
        mask_full_path = os.path.join(MASK_DIR, mask_name)
        
        fixed_name = f"fixed_{item_idx}.png"
        fixed_full_path = os.path.join(FIXED_IMG_DIR, fixed_name)
        
        target_name = f"target_{item_idx}.png"
        target_full_path = os.path.join(TARGET_DIR, target_name)

        # Verify dependencies exist
        if not (os.path.exists(mask_full_path) and os.path.exists(after_path)):
            return None

        # 3. Load Images
        img_after = Image.open(after_path).convert("RGB")
        img_mask = Image.open(mask_full_path).convert("L")

        # 4. Resize After Image to match Mask (Model target resolution)
        target_w, target_h = img_mask.size
        if img_after.size != (target_w, target_h):
            img_after = img_after.resize((target_w, target_h), Image.Resampling.LANCZOS)

        # 5. Generate Ref GT (Object on White)
        # CONFIRMED: 0 is Background, 255 is Object.
        arr_after = np.array(img_after).astype(float)
        arr_mask = np.array(img_mask).astype(float) / 255.0
        arr_mask_3d = np.expand_dims(arr_mask, axis=2) # (H, W, 1)

        white_bg = np.ones_like(arr_after) * 255.0
        
        # Formula: After_Pixel (Object) + White_Background
        # Object is where mask is 1 -> arr_mask_3d
        # Background is where mask is 0 -> (1.0 - arr_mask_3d)
        arr_ref_gt = arr_after * arr_mask_3d + white_bg * (1.0 - arr_mask_3d)
        img_ref_gt = Image.fromarray(np.uint8(np.clip(arr_ref_gt, 0, 255)))

        # 6. Crop Ref GT
        # Thresholding: 0=Background, 255=Object.
        # We define Object as any pixel where mask > 127 (mostly white).
        is_object_mask = (np.array(img_mask) > 127).astype(np.uint8) * 255
        bbox_check_img = Image.fromarray(is_object_mask)
        bbox = bbox_check_img.getbbox() 

        if bbox is None:
            left, upper, right, lower = 0, 0, target_w, target_h
        else:
            left, upper, right, lower = bbox
            
        # Align to 16 (Safe Outward Rounding)
        new_left = (left // 16) * 16
        new_upper = (upper // 16) * 16
        new_right = ((right + 15) // 16) * 16
        new_lower = ((lower + 15) // 16) * 16
        
        new_left = max(0, new_left)
        new_upper = max(0, new_upper)
        new_right = min(target_w, new_right)
        new_lower = min(target_h, new_lower)
        
        # Crop
        img_ref_gt_crop = img_ref_gt.crop((new_left, new_upper, new_right, new_lower))

        # 7. Save Updated Images
        ref_gt_name = f"ref_gt_{item_idx}.png"
        ref_gt_full_path = os.path.join(REF_GT_DIR, ref_gt_name)
        img_ref_gt.save(ref_gt_full_path)
        
        ref_gt_crop_name = f"ref_gt_crop_{item_idx}.png"
        ref_gt_crop_full_path = os.path.join(REF_GT_CROP_DIR, ref_gt_crop_name)
        img_ref_gt_crop.save(ref_gt_crop_full_path)

        # 8. Construct Output Dict (NO subyx)
        final_prompt = f"Picture 1 is the image to modify. {text_prompt}"

        return {
            "prompt": final_prompt,
            "image": os.path.relpath(target_full_path, REL_ROOT),
            "edit_image": [os.path.relpath(fixed_full_path, REL_ROOT)],
            "ref_gt": os.path.relpath(ref_gt_full_path, REL_ROOT),
            "ref_gt_crop": os.path.relpath(ref_gt_crop_full_path, REL_ROOT),
            "back_mask": os.path.relpath(mask_full_path, REL_ROOT)
        }

    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return None

def main():
    json_files = glob.glob(os.path.join(AUDIT_DIR, "*.json"))
    if not json_files:
        print(f"No audit files found in {AUDIT_DIR}")
        return

    print(f"Found {len(json_files)} audit files. Updating RefGT and JSON...")
    
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_one_update, json_files), total=len(json_files)))
    
    dataset = [r for r in results if r is not None]

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Updated dataset with {len(dataset)} entries saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
