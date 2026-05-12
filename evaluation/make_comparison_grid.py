import os
import json
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Configuration
BASE_DIR = Path("/host/ssd2/xiong-p/qwenpe")
INPUT_IMAGE_ROOT = Path("/host/ssd2/xiong-p/pico-banana-400k-subject_driven/openimages")
JSON_PATH = BASE_DIR / "dataset_qwen_pe_top1000_captioned.json"

MODELS = [
    {
        "name": "ACE Plus",
        "csv": BASE_DIR / "final_comparison_results/ace_plus_full.csv",
        "dir": BASE_DIR / "pico_test/ace_plus_results_top1000"
    },
    {
        "name": "ACE Plus Adaptive",
        "csv": BASE_DIR / "final_comparison_results/ace_plus_top1000_adaptive.csv",
        "dir": BASE_DIR / "pico_test/ace_plus_results_top1000_adaptive"
    },
    {
        "name": "Flux",
        "csv": BASE_DIR / "final_comparison_results/flux_full.csv",
        "dir": BASE_DIR / "pico_test/flux_results_top1000"
    },
    {
        "name": "MagicBrush",
        "csv": BASE_DIR / "final_comparison_results/magicbrush_full.csv",
        "dir": BASE_DIR / "pico_test/magicbrush_results_top1000"
    },
    {
        "name": "Qwen NoSTE",
        "csv": BASE_DIR / "final_comparison_results/qwen_noste_30k_full.csv",
        "dir": BASE_DIR / "pico_test/qwen_results_noste_30k_top1000"
    },
    {
        "name": "Qwen w/ STE",
        "csv": BASE_DIR / "final_comparison_results/qwen_w_ste_full.csv",
        "dir": BASE_DIR / "pico_test/qwen_results_top1000"
    },
    {
        "name": "Qwen w/o STE-Sub",
        "csv": BASE_DIR / "final_comparison_results/qwen_wo_ste-sub_full.csv",
        "dir": BASE_DIR / "pico_test/qwen_results_wosub_top1000_pixelmask_cfg4"
    }
]

METRICS = [
    {"col": "SigLIP2_T_Local", "ascending": False, "label": "SigLIP T Local (High)"},
    {"col": "dreamsim", "ascending": True, "label": "DreamSim (Low)"}
]

TOP_K = 10
CELL_SIZE = 256
FONT_SIZE = 20

def load_dataset_map():
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    # Map filename (fixed_XXXX.png) to input path
    # The JSON has "edit_image": ["fixed_images/fixed_10004.png"]
    # We need to map "fixed_10004.png" -> "target_images/target_10004.png"
    mapping = {}
    for entry in data:
        if not entry.get("edit_image"): continue
        out_name = os.path.basename(entry["edit_image"][0])
        in_path = entry["image"] # Relative to root
        mapping[out_name] = in_path
    return mapping

def add_text(img, text, color="white", bg="black"):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE)
    except:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    
    # Bottom left
    x, y = 5, img.height - h - 5
    draw.rectangle([x-2, y-2, x+w+2, y+h+2], fill=bg)
    draw.text((x, y), text, fill=color, font=font)
    return img

def main():
    print("Loading dataset mapping...")
    file_map = load_dataset_map()
    
    # 1. Collect Sections
    sections = [] # List of (Section Title, List of Filenames)
    
    for metric in METRICS:
        for model in MODELS:
            print(f"Processing {model['name']} for {metric['label']}...")
            df = pd.read_csv(model['csv'])
            
            # Ensure column exists
            if metric['col'] not in df.columns:
                # Try fallback for "dreamsim" (sometimes lowercase) or "SigLIP2_T_Local"
                found = False
                for c in df.columns:
                    if c.lower() == metric['col'].lower():
                        metric['col'] = c
                        found = True
                        break
                if not found:
                    print(f"  WARNING: Column {metric['col']} not found in {model['name']}")
                    continue

            # Sort
            df_sorted = df.sort_values(by=metric['col'], ascending=metric['ascending'])
            top_filenames = df_sorted['filename'].head(TOP_K).tolist()
            
            title = f"Top {TOP_K} {metric['label']} by {model['name']}"
            sections.append((title, top_filenames))

    # 2. Create Image
    # Total Rows = sum(len(s[1]) for s in sections) (should be 7 * 2 * 10 = 140)
    # Total Cols = 1 (Input) + len(MODELS)
    
    n_rows = sum(len(s[1]) for s in sections)
    n_cols = 1 + len(MODELS)
    
    # Headers
    header_h = 50
    section_h = 40
    
    total_width = n_cols * CELL_SIZE
    total_height = (n_rows * CELL_SIZE) + (len(sections) * section_h) + header_h
    
    print(f"Creating grid: {total_width}x{total_height} (Rows: {n_rows})")
    
    # Limit check
    if total_height > 60000:
        print("Warning: Image height is very large. Splitting into two images by metric.")
        # Split logic could be added here, but for now let's try to save it or split manually
        # Let's split into two images: one for SigLIP, one for DreamSim
        process_split(sections, file_map, n_cols, header_h, section_h)
        return

    # If small enough (unlikely for 140 rows @ 256 = 35840, actually fits in PNG usually)
    # PIL limit is often related to memory, but coordinates are 32-bit.
    # 65535 is a limit for some formats/viewers. 35k is fine.
    
    create_grid_image(sections, file_map, n_cols, header_h, section_h, "compare_top_samples.jpg")

def process_split(all_sections, file_map, n_cols, header_h, section_h):
    # Split by metric type
    siglip_sections = [s for s in all_sections if "SigLIP" in s[0]]
    dreamsim_sections = [s for s in all_sections if "DreamSim" in s[0]]
    
    create_grid_image(siglip_sections, file_map, n_cols, header_h, section_h, "compare_top_samples_siglip.jpg")
    create_grid_image(dreamsim_sections, file_map, n_cols, header_h, section_h, "compare_top_samples_dreamsim.jpg")

def create_grid_image(sections, file_map, n_cols, header_h, section_h, output_filename):
    n_rows = sum(len(s[1]) for s in sections)
    total_width = n_cols * CELL_SIZE
    total_height = (n_rows * CELL_SIZE) + (len(sections) * section_h) + header_h
    
    canvas = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(canvas)
    
    try:
        font_header = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_sec = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font_header = ImageFont.load_default()
        font_sec = ImageFont.load_default()

    # Draw Column Headers
    y_cursor = 0
    headers = ["Input"] + [m["name"] for m in MODELS]
    for i, h in enumerate(headers):
        draw.text((i * CELL_SIZE + 10, 10), h, fill="black", font=font_header)
    y_cursor += header_h
    
    for title, filenames in sections:
        # Draw Section Title
        draw.rectangle([0, y_cursor, total_width, y_cursor + section_h], fill="#ddd")
        draw.text((10, y_cursor + 5), title, fill="black", font=font_sec)
        y_cursor += section_h
        
        for fname in filenames:
            # 1. Draw Input
            in_rel_path = file_map.get(fname)
            if in_rel_path:
                in_full_path = INPUT_IMAGE_ROOT / in_rel_path
                if in_full_path.exists():
                    img = Image.open(in_full_path).convert("RGB").resize((CELL_SIZE, CELL_SIZE))
                    canvas.paste(img, (0, y_cursor))
                else:
                    print(f"Input missing: {in_full_path}")
            
            # 2. Draw Models
            for i, model in enumerate(MODELS):
                img_path = model["dir"] / fname
                col_idx = i + 1
                if img_path.exists():
                    try:
                        img = Image.open(img_path).convert("RGB").resize((CELL_SIZE, CELL_SIZE))
                        canvas.paste(img, (col_idx * CELL_SIZE, y_cursor))
                    except Exception as e:
                        print(f"Error reading {img_path}: {e}")
                else:
                    # Draw placeholder
                    # print(f"Output missing for {model['name']}: {fname}")
                    pass
            
            y_cursor += CELL_SIZE
            
    canvas.save(output_filename, quality=85)
    print(f"Saved {output_filename}")

if __name__ == "__main__":
    main()
