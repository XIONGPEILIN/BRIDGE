import os
import json
import shutil
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(".")
INPUT_IMAGE_ROOT = BASE_DIR / "pico-banana-400k-subject_driven/openimages"
JSON_PATH = BASE_DIR / "dataset_qwen_pe_top1000_captioned.json"
OUTPUT_ROOT = BASE_DIR / "final_model_galleries"

# Items to skip
SKIP_IDS = {
    "107", "42050", "17681", "23165", "28166", "45397", "46690", 
    "30732", "30549", "504", "15136", "37437", "10739", "36079", 
    "12108", "28925", "6083", "17669", "31032", "19463", "33196", "30735"
}

# Model Definitions
MODELS = {
    "ace_adaptive": {
        "name": "Ace Plus Adaptive",
        "csv": BASE_DIR / "final_comparison_results/ace_plus_top1000_adaptive.csv",
        "dir": BASE_DIR / "pico_test/ace_plus_results_top1000_adaptive",
        "label": "Ace+ Adaptive"
    },
    "flux": {
        "name": "Flux",
        "csv": BASE_DIR / "final_comparison_results/flux_full.csv",
        "dir": BASE_DIR / "pico_test/flux_results_top1000",
        "label": "Flux"
    },
    "noste": {
        "name": "Qwen NoSTE",
        "csv": BASE_DIR / "final_comparison_results/qwen_noste_30k_full.csv",
        "dir": BASE_DIR / "pico_test/qwen_results_noste_30k_top1000",
        "label": "NoSTE"
    },
    "ste": {
        "name": "Qwen w/ STE",
        "csv": BASE_DIR / "final_comparison_results/qwen_w_ste_full.csv",
        "dir": BASE_DIR / "pico_test/qwen_results_top1000",
        "label": "Qwen STE (Ours)"
    },
    "nosub": {
        "name": "Qwen w/o Sub",
        "csv": BASE_DIR / "final_comparison_results/qwen_wo_ste-sub_full.csv",
        "dir": BASE_DIR / "pico_test/qwen_results_wosub_top1000_pixelmask_cfg4",
        "label": "NoSub"
    }
}

# Group Members (Corrected Order)
# SOTA: Input | Ace+ | Flux | STE
GROUP_SOTA = ["ace_adaptive", "flux", "ste"]
# Ablation: Input | NoSub | NoSTE | STE
GROUP_ABLATION = ["nosub", "noste", "ste"]

TOP_K_PER_TYPE = 5
SAFE_THRESHOLD = 16
COLOR_THRESHOLD = 10
GAP_SIZE = 20

EDIT_TYPE_MAP = {
    "Replace one object category with another": "replace",
    "Remove an existing object": "remove",
    "Add a new object to the scene": "add",
    "Clothing edit (change color/outfit)": "clothing",
    "Add/Remove/Replace Accessories (glasses, hats, jewelry, masks)": "accessories"
}

def load_all_scores():
    scores = {}
    for key, cfg in MODELS.items():
        if not cfg.get("csv"): continue
        try:
            df = pd.read_csv(cfg["csv"])
            cols = {c.lower(): c for c in df.columns}
            target_col = cols.get("siglip2_t_local")
            if target_col:
                for _, row in df.iterrows():
                    f = row['filename']
                    if f not in scores: scores[f] = {}
                    scores[f][key] = row[target_col]
        except Exception as e:
            print(f"Error loading CSV for {key}: {e}")
    return scores

def load_dataset_map():
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    path_map, meta_map, mask_map = {}, {}, {}
    for entry in data:
        if not entry.get("edit_image"): continue
        out_name = os.path.basename(entry["edit_image"][0])
        path_map[out_name] = entry["edit_image"][0] 
        meta_map[out_name] = entry
        if entry.get("back_mask"):
            mask_map[out_name] = entry["back_mask"]
    return path_map, meta_map, mask_map

def rigorous_crop(img, return_margins=False):
    arr = np.array(img)
    h, w = arr.shape[:2]
    w_crop, h_crop = 0, 0
    for i in range(w - 1, w - 1 - SAFE_THRESHOLD, -1):
        if i < 0: break
        col = arr[:, i] if arr.ndim == 2 else arr[:, i, :]
        if np.mean(col) < COLOR_THRESHOLD: w_crop += 1
        else: break
    for i in range(h - 1, h - 1 - SAFE_THRESHOLD, -1):
        if i < 0: break
        row = arr[i, :] if arr.ndim == 2 else arr[i, :, :]
        if np.mean(row) < COLOR_THRESHOLD: h_crop += 1
        else: break
    cropped = img.crop((0, 0, w - w_crop, h - h_crop))
    return (cropped, w_crop, h_crop) if return_margins else cropped

def create_single_group_image(filename, path_map, mask_map, output_path, group_members, all_scores):
    in_rel_path = path_map.get(filename)
    if not in_rel_path: return
    in_full_path = INPUT_IMAGE_ROOT / in_rel_path
    if not in_full_path.exists(): return
    img_input = Image.open(in_full_path).convert("RGB")
    
    target_size, wc, hc = img_input.size, 0, 0
    ste_path = MODELS["ste"]["dir"] / filename
    if ste_path.exists():
        ste_img = Image.open(ste_path).convert("RGB")
        _, wc, hc = rigorous_crop(ste_img, return_margins=True)
        target_size = (ste_img.size[0] - wc, ste_img.size[1] - hc)

    if img_input.size != target_size:
        img_input = img_input.resize(target_size, Image.Resampling.LANCZOS)
    
    # Prepare BBox
    bbox = None
    mask_rel = mask_map.get(filename)
    if mask_rel:
        full_m = INPUT_IMAGE_ROOT / mask_rel
        if full_m.exists():
            mask = Image.open(full_m).convert("L")
            if wc > 0 or hc > 0:
                mask = mask.crop((0, 0, mask.size[0] - wc, mask.size[1] - hc))
            if mask.size != target_size:
                mask = mask.resize(target_size, Image.NEAREST)
            bbox = mask.getbbox()

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except:
        font = ImageFont.load_default()

    def draw_bbox_and_score(img, model_key=None):
        if not bbox: return
        draw = ImageDraw.Draw(img)
        draw.rectangle(bbox, outline="red", width=3)
        
        if model_key:
            score = all_scores.get(filename, {}).get(model_key)
            if score is not None:
                text = f"{score:.3f}"
                # Position: Outside top-right corner, right-aligned with bbox
                x1, y0 = bbox[2], bbox[1]
                
                # Determine position
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
                
                # Right align with bbox right edge
                pos_x = x1 - text_w
                pos_y = y0 - text_h - 5
                
                # If too high, put it below or inside?
                if pos_y < 0:
                    pos_y = y0 + 5 # Fallback to inside/below line if clipped top
                
                draw.text((pos_x, pos_y), text, fill="red", font=font)

    # Process Input Image
    draw_bbox_and_score(img_input, model_key=None)

    def load_model_img_and_process(model_key):
        m = MODELS[model_key]
        p = m["dir"] / filename
        img = None
        if p.exists():
            img = Image.open(p).convert("RGB")
            img = rigorous_crop(img)
            if img.size != target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
        else:
            img = Image.new("RGB", target_size, (255, 255, 255))
        
        draw_bbox_and_score(img, model_key)
        return img

    imgs = [img_input] + [load_model_img_and_process(k) for k in group_members]
    
    total_w = sum(im.width for im in imgs) + (GAP_SIZE * (len(imgs) - 1))
    total_h = max(im.height for im in imgs)
    canvas = Image.new("RGB", (total_w, total_h), "white")
    x_cursor = 0
    for im in imgs:
        canvas.paste(im, (x_cursor, 0))
        x_cursor += im.width + GAP_SIZE
    canvas.save(output_path, quality=95)

def process_model_per_type(path_map, meta_map, mask_map):
    all_scores = load_all_scores()
    model = MODELS["ste"]
    base_out_dir = OUTPUT_ROOT / "Qwen_w-_STE"
    df = pd.read_csv(model['csv'])
    
    col_l, col_g = "SigLIP2_T_Local", "SigLIP2_T_Global"
    cols_lower = {c.lower(): c for c in df.columns}
    if col_l.lower() in cols_lower: col_l = cols_lower[col_l.lower()]
    if col_g.lower() in cols_lower: col_g = cols_lower[col_g.lower()]
    
    def get_type(fname):
        if fname in meta_map:
            return EDIT_TYPE_MAP.get(meta_map[fname].get("edit_type"), "other")
        return "unknown"
    df['short_type'] = df['filename'].apply(get_type)
    
    def should_skip(fname):
        try: return fname.split('_')[1].split('.')[0] in SKIP_IDS
        except: return False
    df = df[~df['filename'].apply(should_skip)]
    
    for etype in df['short_type'].unique():
        if etype in ["unknown", "other"]: continue
        type_dir = base_out_dir / etype
        type_dir.mkdir(parents=True, exist_ok=True)
        df_type = df[df['short_type'] == etype]
        
        for metric_col, metric_label in [(col_l, "siglip_local"), (col_g, "siglip_global")]:
            df_top = df_type.sort_values(by=metric_col, ascending=False).head(TOP_K_PER_TYPE)
            for rank, row in enumerate(df_top.itertuples(), 1):
                fname = row.filename
                prefix = f"{etype}_rank{rank:02d}_{metric_label}"
                if fname in meta_map:
                    with open(type_dir / f"{prefix}_{fname}.json", "w") as f:
                        json.dump(meta_map[fname], f, indent=4)
                create_single_group_image(fname, path_map, mask_map, type_dir / f"{prefix}_{fname}_sota.jpg", GROUP_SOTA, all_scores)
                create_single_group_image(fname, path_map, mask_map, type_dir / f"{prefix}_{fname}_ablation.jpg", GROUP_ABLATION, all_scores)

def main():
    if OUTPUT_ROOT.exists(): shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir()
    path_map, meta_map, mask_map = load_dataset_map()
    process_model_per_type(path_map, meta_map, mask_map)
    print(f"Done. Results in {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()