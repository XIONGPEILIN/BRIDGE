import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def add_label(image, text):
    draw = ImageDraw.Draw(image)
    # 尝试加载字体，如果失败使用默认
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    # 绘制文字背景
    bbox = draw.textbbox((10, 10), text, font=font)
    draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill="black")
    draw.text((10, 10), text, fill="white", font=font)
    return image

def main():
    base_dir = Path("compare/grid_search_ablation")
    output_dir = Path("compare/ablation_grids")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 定义实验配置前缀和变体名称
    # 注意：这里假设只有一个 ckpt/cfg/alpha 组合。如果有多个，需要额外循环。
    # 根据 ls 结果：2011-ste-28000_cfg2_alpha0.1
    prefix = "2011-ste-28000_cfg2_alpha0.1"
    
    variants = {
        "Baseline": f"{prefix}_baseline",
        "Gate 0 (Force Zero)": f"{prefix}_gate0",
        "Gate 1 (Force One)": f"{prefix}_gate1",
        "No Sub Noise": f"{prefix}_no_sub_noise"
    }
    
    mask_variants = {
        "Baseline+Mask": f"{prefix}_baseline_mask",
        "Gate 0+Mask": f"{prefix}_gate0_mask",
        "Gate 1+Mask": f"{prefix}_gate1_mask",
        "No Sub+Mask": f"{prefix}_no_sub_noise_mask"
    }

    mask_self_variants = {
        "Baseline+MaskSelf": f"{prefix}_baseline_mask_self",
        "Gate 0+MaskSelf": f"{prefix}_gate0_mask_self",
        "Gate 1+MaskSelf": f"{prefix}_gate1_mask_self",
    }

    # 找到所有样本 ID
    # 扫描 baseline 目录下的所有文件夹
    baseline_dir = base_dir / variants["Baseline"]
    if not baseline_dir.exists():
        print(f"Base directory not found: {baseline_dir}")
        return

    sample_ids = [d.name for d in baseline_dir.iterdir() if d.is_dir()]
    sample_ids.sort(key=lambda x: int(x) if x.isdigit() else x)

    print(f"Found {len(sample_ids)} samples.")

    for sample_id in sample_ids:
        print(f"Processing sample {sample_id}...")
        
        # 尝试获取 BBox
        bbox = None
        # 遍历所有可能的文件夹寻找 back_mask.png
        all_folders = list(variants.values()) + list(mask_variants.values()) + list(mask_self_variants.values())
        for folder_name in all_folders:
            mask_path = base_dir / folder_name / sample_id / "back_mask.png"
            if mask_path.exists():
                try:
                    mask_img = Image.open(mask_path).convert("L")
                    bbox = mask_img.getbbox()
                    if bbox:
                        # 稍微扩大一点 bbox，避免贴边
                        w_img, h_img = mask_img.size
                        x1, y1, x2, y2 = bbox
                        pad = 20
                        x1 = max(0, x1 - pad)
                        y1 = max(0, y1 - pad)
                        x2 = min(w_img, x2 + pad)
                        y2 = min(h_img, y2 + pad)
                        bbox = (x1, y1, x2, y2)
                        print(f"  Found bbox: {bbox}")
                        break
                except Exception as e:
                    print(f"  Error reading mask {mask_path}: {e}")
        
        images = {}
        # target_size will be the size *after* cropping
        target_size = None

        # Helper to load images
        def load_variant_images(variant_dict):
            nonlocal target_size
            for label, folder_name in variant_dict.items():
                # Load Main Output
                img_path = base_dir / folder_name / sample_id / "output.png"
                sub_img_path = base_dir / folder_name / sample_id / "output_sub.png"
                
                main_img = None
                sub_img = None

                if img_path.exists():
                    main_img = Image.open(img_path).convert("RGB")
                    
                    # Crop if bbox exists
                    if bbox:
                        main_img = main_img.crop(bbox)
                    
                    if target_size is None:
                        target_size = main_img.size
                
                if sub_img_path.exists():
                    sub_img = Image.open(sub_img_path).convert("RGB")
                
                # Handle Main Image
                if main_img:
                    main_img = add_label(main_img, label)
                    images[label] = main_img
                else:
                    print(f"  Missing {label} for {sample_id}")
                    images[label] = None # Placeholder handled later

                # Handle Sub Image
                if sub_img:
                    # Resize sub image to match main image (target_size)
                    # If main image is missing but target_size is set, resize to target_size
                    # If both missing/target_size not set, we can't resize yet, handle later
                    if target_size:
                        sub_img = sub_img.resize(target_size)
                    sub_img = add_label(sub_img, f"{label} Sub")
                    images[f"{label}_sub"] = sub_img
                else:
                    images[f"{label}_sub"] = None


        load_variant_images(variants)
        load_variant_images(mask_variants)
        load_variant_images(mask_self_variants)

        if target_size is None:
            # 如果所有图都没找到，可能这里需要根据 bbox 确定一个 target_size 或者跳过
            # 但既然没图，也无法确定 bbox (除非 mask 单独存在)
            # 如果 mask 存在但 output 不存在 (failed case)，我们需要一个默认尺寸
            if bbox:
                target_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
            else:
                print(f"  No images found for {sample_id}, skipping.")
                continue

        # 确保所有图片尺寸一致 (Handle missing images)
        all_labels = list(variants.keys()) + list(mask_variants.keys()) + list(mask_self_variants.keys())
        # Expand labels to include sub labels
        expanded_labels = []
        for label in all_labels:
            expanded_labels.append(label)
            expanded_labels.append(f"{label}_sub")

        for label in expanded_labels:
            if images.get(label) is None:
                images[label] = Image.new("RGB", target_size, color="black")
                images[label] = add_label(images[label], label) # Add label to black placeholder
            elif images[label].size != target_size:
                images[label] = images[label].resize(target_size)

        # 拼接 6x4 网格 (3 columns of pairs)
        w, h = target_size
        grid_img = Image.new("RGB", (w * 6, h * 4))
        
        # 布局：
        # Baseline | BaseSub | Mask | MaskSub | MaskSelf | MaskSelfSub
        
        def paste_pair(base_label, row_idx, col_group_idx):
            # col_group_idx: 0, 1, 2
            x_base = col_group_idx * 2 * w
            y_base = row_idx * h
            
            grid_img.paste(images[base_label], (x_base, y_base))
            grid_img.paste(images[f"{base_label}_sub"], (x_base + w, y_base))

        # Row 0: Baseline Group
        paste_pair("Baseline", 0, 0)
        paste_pair("Baseline+Mask", 0, 1)
        paste_pair("Baseline+MaskSelf", 0, 2)
        
        # Row 1: Gate 0
        paste_pair("Gate 0 (Force Zero)", 1, 0)
        paste_pair("Gate 0+Mask", 1, 1)
        paste_pair("Gate 0+MaskSelf", 1, 2)
        
        # Row 2: Gate 1
        paste_pair("Gate 1 (Force One)", 2, 0)
        paste_pair("Gate 1+Mask", 2, 1)
        paste_pair("Gate 1+MaskSelf", 2, 2)
        
        # Row 3: No Sub
        paste_pair("No Sub Noise", 3, 0)
        paste_pair("No Sub+Mask", 3, 1)
        # Empty last pair (or black placeholders if loaded as None)
        # images dict handles placeholders automatically if keys don't exist? 
        # Actually variants dicts don't have "No Sub+MaskSelf".
        # We need to manually handle the empty spot if we want to fill it black explicitly, 
        # but Image.new initialized with black, so just leaving it alone is fine.
        # But we added black placeholders for known labels. "No Sub+MaskSelf" isn't a known label.
        # So it will remain black (background).

        # 保存
        out_path = output_dir / f"{sample_id}_ablation.jpg"
        grid_img.save(out_path, quality=90)
        print(f"  Saved to {out_path}")

if __name__ == "__main__":
    main()
