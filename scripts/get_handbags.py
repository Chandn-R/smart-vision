import fiftyone as fo
import fiftyone.zoo as foz
import os

# Destination
EXPORT_DIR = os.path.expanduser("../data/yolo_handbags-2")

def get_handbags():
    print("👜 Downloading 1,500 Handbags...")
    
    # Download strictly handbags
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="train", 
        label_types=["detections"],
        classes=["handbag"], 
        max_samples=1500,
        shuffle=True
    )

    print(f"💾 Exporting to {EXPORT_DIR}...")
    
    dataset.export(
        export_dir=EXPORT_DIR,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        classes=["handbag"], # Only keep the handbag label
    )
    
    print("✅ Done! Upload this folder to Roboflow.")

if __name__ == "__main__":
    get_handbags()