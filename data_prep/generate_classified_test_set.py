import json
import os
import glob
import random
from tqdm import tqdm

# Configuration
AUDIT_DIR = 'pico-banana-400k-subject_driven/openimages/dreamsim_mask_audit_mask'
METADATA_JSONL = 'pico-banana-400k-subject_driven/openimages/jsonl/sft_with_local_source_image_path.jsonl'
DATASET_JSON = 'dataset_qwen_pe_fixed.json'
OUTPUT_JSON = 'dataset_qwen_pe_test_classified.json'

TARGET_TYPES = {
    "Add a new object to the scene",
    "Add/Remove/Replace Accessories (glasses, hats, jewelry, masks)",
    "Clothing edit (change color/outfit)",
    "Remove an existing object",
    "Replace one object category with another"
}

SAMPLES_PER_CLASS = 500
BG_DIST_THRESHOLD = 0.4

def main():
    # 1. Load Edit Type Mapping from JSONL
    print("Loading edit type mapping...")
    id_to_type = {}
    with open(METADATA_JSONL, 'r') as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            # The i-th line in jsonl corresponds to item_idx i in audit files
            id_to_type[i] = item.get('edit_type')

    # 2. Collect Audit Scores (bg_dist_max)
    print("Collecting audit scores...")
    audit_files = glob.glob(os.path.join(AUDIT_DIR, "item_*_dreamsim_audit.json"))
    
    type_to_items = {t: [] for t in TARGET_TYPES}
    
    for fpath in tqdm(audit_files):
        try:
            with open(fpath, 'r') as f:
                audit_data = json.load(f)
            
            idx = audit_data.get('item_idx')
            bg_dist_max = audit_data.get('results', {}).get('global', {}).get('bg_dist_max')
            
            if idx is None or bg_dist_max is None:
                continue
                
            edit_type = id_to_type.get(idx)
            if edit_type in TARGET_TYPES:
                type_to_items[edit_type].append({
                    'idx': idx,
                    'bg_dist_max': bg_dist_max
                })
        except Exception as e:
            continue

    # 3. Filter and Random Select
    selected_indices = set()
    print(f"\nSelection Statistics (Threshold < {BG_DIST_THRESHOLD}):")
    random.seed(42) # Ensure reproducibility
    
    for etype, items in type_to_items.items():
        # Filter items by threshold
        valid_items = [x for x in items if x['bg_dist_max'] < BG_DIST_THRESHOLD]
        
        # Shuffle randomly
        random.shuffle(valid_items)
        
        # Select top N
        top_n = valid_items[:SAMPLES_PER_CLASS]
        
        for item in top_n:
            selected_indices.add(item['idx'])
        print(f" - {etype}: Total Found {len(items)}, Valid (<{BG_DIST_THRESHOLD}) {len(valid_items)}, Selected {len(top_n)}")

    # 4. Filter original dataset for selected indices
    print(f"\nFiltering dataset: {DATASET_JSON}")
    with open(DATASET_JSON, 'r') as f:
        full_dataset = json.load(f)
    
    test_set = []
    train_set = []
    
    for item in full_dataset:
        image_path = item.get('image', '')
        if 'target_images/target_' in image_path:
            try:
                idx_str = image_path.split('target_')[-1].split('.png')[0]
                idx = int(idx_str)
                if idx in selected_indices:
                    # Add metadata for convenience
                    item['edit_type'] = id_to_type.get(idx)
                    test_set.append(item)
                else:
                    train_set.append(item)
            except:
                train_set.append(item)
        else:
            train_set.append(item)

    # 5. Save outputs
    TEST_OUTPUT = 'dataset_qwen_pe_test.json'
    TRAIN_OUTPUT = 'dataset_qwen_pe_train.json'
    
    print(f"Saving {len(test_set)} items to {TEST_OUTPUT}")
    with open(TEST_OUTPUT, 'w') as f:
        json.dump(test_set, f, indent=2)
        
    print(f"Saving {len(train_set)} items to {TRAIN_OUTPUT}")
    with open(TRAIN_OUTPUT, 'w') as f:
        json.dump(train_set, f, indent=2)
    
    print("\nFinal counts:")
    print(f"Total Source: {len(full_dataset)}")
    print(f"Test Set: {len(test_set)}")
    print(f"Train Set: {len(train_set)}")
    print("Done.")

if __name__ == "__main__":
    main()
