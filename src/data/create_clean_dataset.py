import cv2
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil

def mask_to_yolo_bbox(mask_path, class_id, img_width=512, img_height=512):
    """Convert mask to YOLO format"""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 10 or h < 10:  # Skip tiny contours
            continue
        
        center_x = (x + w/2) / img_width
        center_y = (y + h/2) / img_height
        norm_width = w / img_width
        norm_height = h / img_height
        
        bboxes.append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}")
    
    return bboxes

print("Creating CLEAN dataset with ONLY complete disc+cup pairs...")

# Set paths
FUNDUS_DIR = Path("raw_data/full-fundus")
DISC_DIR = Path("raw_data/optic-disc")
CUP_DIR = Path("raw_data/optic-cup")
OUTPUT_DIR = Path("yolo_dataset_clean")

# Clean up any existing output directory
if OUTPUT_DIR.exists():
    print(f"Removing existing {OUTPUT_DIR}...")
    shutil.rmtree(OUTPUT_DIR)

# Create output directories
(OUTPUT_DIR / 'images' / 'train').mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'images' / 'val').mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

# Get disc and cup masks
disc_masks = {f.stem: f for f in DISC_DIR.glob("*.png")}
cup_masks = {f.stem: f for f in CUP_DIR.glob("*.png")}

# Find ONLY images with BOTH disc AND cup masks
paired_images = set(disc_masks.keys()) & set(cup_masks.keys())
print(f"Found {len(paired_images)} images with BOTH disc AND cup masks")

# Get fundus images that have both masks
fundus_images = list(FUNDUS_DIR.glob("*.png")) + list(FUNDUS_DIR.glob("*.jpg"))
fundus_dict = {f.stem: f for f in fundus_images}

# Process ONLY paired images
valid_images = []
missing_fundus = []

for base_name in tqdm(paired_images, desc="Processing paired annotations"):
    # Check if we have the fundus image
    if base_name not in fundus_dict:
        missing_fundus.append(base_name)
        continue
    
    fundus_path = fundus_dict[base_name]
    
    # Get both disc and cup annotations
    disc_bboxes = mask_to_yolo_bbox(disc_masks[base_name], class_id=0)
    cup_bboxes = mask_to_yolo_bbox(cup_masks[base_name], class_id=1)
    
    # Only add if both annotations are valid
    if disc_bboxes and cup_bboxes:
        annotations = disc_bboxes + cup_bboxes
        valid_images.append({
            'image_name': fundus_path.name,
            'image_path': fundus_path,
            'annotations': annotations,
            'base_name': base_name
        })

print(f"\n📊 Dataset Statistics:")
print(f"  Images with both masks: {len(paired_images)}")
print(f"  Images with valid annotations: {len(valid_images)}")
print(f"  Missing fundus images: {len(missing_fundus)}")

if len(valid_images) == 0:
    print("\n No valid paired images found!")
    exit(1)

# Split into train/val
train_data, val_data = train_test_split(valid_images, test_size=0.2, random_state=42)
print(f"\n📊 Split:")
print(f"  Training: {len(train_data)} images")
print(f"  Validation: {len(val_data)} images")

# Copy images and create labels
for split_name, split_data in [('train', train_data), ('val', val_data)]:
    print(f"\n📝 Creating {split_name} set...")
    for item in tqdm(split_data, desc=f"Processing {split_name}"):
        # Copy image
        img = cv2.imread(str(item['image_path']))
        if img is None:
            print(f"Warning: Could not read {item['image_path']}")
            continue
            
        img_dst = OUTPUT_DIR / 'images' / split_name / item['image_name']
        cv2.imwrite(str(img_dst), img)
        
        # Create label file
        label_name = item['image_name'].replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = OUTPUT_DIR / 'labels' / split_name / label_name
        
        with open(label_path, 'w') as f:
            for ann in item['annotations']:
                f.write(ann + '\n')

# Create data.yaml
yaml_content = """# Clean Glaucoma Detection Dataset (Paired Only)
# This dataset contains ONLY images with both optic disc AND cup annotations
# Perfect for CDR (Cup-to-Disc Ratio) calculation

path: yolo_dataset_clean
train: images/train
val: images/val

# Classes
names:
  0: optic_disc
  1: optic_cup

nc: 2  # number of classes

# Dataset info
# All images have both disc and cup annotations for proper CDR calculation
"""

with open(OUTPUT_DIR / 'data.yaml', 'w') as f:
    f.write(yaml_content)

