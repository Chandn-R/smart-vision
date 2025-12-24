import fiftyone as fo
import fiftyone.zoo as foz
import os

# --- CONFIGURATION ---
EXPORT_DIR = os.path.expanduser("../data/yolo_balanced_coco")

# 1. The Exact Quotas we calculated for your dataset balance
# This ensures we don't just get 5000 persons and 0 backpacks.
QUOTAS = {
    "backpack": 1500,  # We need these to learn "Backpack vs Gun"
    "handbag": 1500,   # We need these to learn "Bag vs Gun"
    "knife": 500,      # Just a few real knives to augment your set
    "person": 2000     # Pure "Normal" people (walking, sitting, etc.)
}

# 2. The Final List of Classes to Keep in the Labels
# (We want to keep ALL these labels, even if we downloaded the image looking for a backpack)
FINAL_CLASSES = ["person", "backpack", "handbag", "knife"]

def download_and_merge():
    print("🚀 Initializing Smart Balanced Download...")
    
    # Create a temporary internal dataset to hold our merged collection
    combined_dataset = fo.Dataset("smartvision_coco_mix")
    
    # --- PHASE 1: DOWNLOAD BY QUOTA ---
    for class_name, count in QUOTAS.items():
        print(f"\n📥 Hunting for {count} images containing '{class_name}'...")
        
        try:
            # Download specific subset
            subset = foz.load_zoo_dataset(
                "coco-2017",
                split="validation", # Use 'train' if you need more data, 'validation' is cleaner
                label_types=["detections"],
                classes=[class_name], # Only download if it has THIS specific class
                max_samples=count,
                shuffle=True, # Randomize so we don't get 1500 frames of the same video
            )
            
            # Merge into our main bucket (FiftyOne handles de-duplication automatically)
            combined_dataset.merge_samples(subset)
            
        except Exception as e:
            print(f"⚠️ Warning: Could not download full quota for {class_name}. {e}")

    # --- PHASE 2: EXPORT TO YOLO FORMAT ---
    print(f"\n💾 Exporting {len(combined_dataset)} unique images to {EXPORT_DIR}...")
    
    combined_dataset.export(
        export_dir=EXPORT_DIR,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        classes=FINAL_CLASSES, # CRITICAL: This filters out 'giraffe', 'umbrella', etc.
    )
    
    print("\n✅ SUCCESS! Data is ready.")
    print(f"   Folder: {EXPORT_DIR}")
    print("   Next Step: Merge this folder with your 'smartvision_master' using the YAML file.")

if __name__ == "__main__":
    # Ensure fiftyone is installed
    try:
        import fiftyone
        download_and_merge()
    except ImportError:
        print("❌ Error: FiftyOne is not installed.")
        print("   Run: pip install fiftyone")