import os
import json
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- Configuration ---
JSON_PATH = "dataset_qwen_pe_fixed.json"
BASE_DIR = "pico-banana-400k-subject_driven/openimages"

def check_one_entry(entry):
    """
    Checks one dataset entry for file existence, size, corruption, and dimension alignment.
    """
    errors = []
    try:
        # 1. Paths to check
        keys_to_check = ["image", "ref_gt", "ref_gt_crop", "back_mask"]
        # edit_image is a list
        files = [entry[k] for k in keys_to_check] + [entry["edit_image"][0]]
        
        dims = {}
        for rel_path in files:
            full_path = os.path.join(BASE_DIR, rel_path)
            
            # A. Check existence
            if not os.path.exists(full_path):
                errors.append(f"Missing: {rel_path}")
                continue
            
            # B. Check for zero size
            if os.path.getsize(full_path) == 0:
                errors.append(f"Zero size: {rel_path}")
                continue
            
            # C. Verify image integrity
            try:
                with Image.open(full_path) as img:
                    img.verify() # Fast check for file structure
                
                # Re-open to get dimensions (required after verify())
                with Image.open(full_path) as img:
                    dims[rel_path] = img.size
                    
                    # D. Check for empty masks
                    if "fixed_masks" in rel_path:
                        extrema = img.getextrema()
                        if extrema == (0, 0) or extrema == (255, 255):
                            errors.append(f"Empty Mask: {rel_path} (extrema {extrema})")
            except Exception as e:
                errors.append(f"Corrupted: {rel_path} ({str(e)})")

        # 2. Check dimension alignment (image, edit_image, back_mask, ref_gt must match)
        main_files = [entry["image"], entry["edit_image"][0], entry["back_mask"], entry["ref_gt"]]
        main_dims = [dims[f] for f in main_files if f in dims]
        
        if len(set(main_dims)) > 1:
            errors.append(f"Dimension Mismatch: {main_dims}")

    except Exception as e:
        errors.append(f"Unexpected error processing entry: {str(e)}")

    return errors if errors else None

def main():
    if not os.path.exists(JSON_PATH):
        print(f"Error: {JSON_PATH} not found.")
        return

    with open(JSON_PATH, 'r') as f:
        dataset = json.load(f)

    print(f"Starting multi-threaded check for {len(dataset)} entries using {cpu_count()} cores...")
    
    # Use process pool for CPU-bound image checking
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(check_one_entry, dataset), total=len(dataset)))

    # Aggregate results
    bad_indices = [i for i, res in enumerate(results) if res is not None]
    
    print("\n" + "="*50)
    print(f"Check Complete.")
    print(f"Total entries: {len(dataset)}")
    print(f"Healthy entries: {len(dataset) - len(bad_indices)}")
    print(f"Corrupted entries: {len(bad_indices)}")
    print("="*50 + "\n")

    if bad_indices:
        print("Summary of first 10 errors:")
        for idx in bad_indices[:10]:
            print(f"--- Entry Index {idx} ---")
            for error_msg in results[idx]:
                print(f"  [!] {error_msg}")
        
        # Save bad entries for further analysis
        corrupted_data = [dataset[i] for i in bad_indices]
        with open("corrupted_dataset_report.json", "w") as f:
            json.dump(corrupted_data, f, indent=2)
        print(f"\nReport saved to corrupted_dataset_report.json")
    else:
        print("No errors found. Dataset is consistent and ready for training.")

if __name__ == "__main__":
    main()
