import json
import random
import shutil
import os
from pathlib import Path
from tqdm import tqdm

# Paths (adjust to your system)
coco_root = './datasets'  # e.g., /home/user/datasets/coco
annotations_path = f'{coco_root}/annotations/stuff_train2017.json'
images_dir = f'{coco_root}/train2017'

# Output folders
output_root = './datasets'
os.makedirs(output_root, exist_ok=True)
os.makedirs(f'{output_root}/images/train', exist_ok=True)
os.makedirs(f'{output_root}/images/val', exist_ok=True)
os.makedirs(f'{output_root}/images/test', exist_ok=True)
os.makedirs(f'{output_root}/labels/train', exist_ok=True)
os.makedirs(f'{output_root}/labels/val', exist_ok=True)
os.makedirs(f'{output_root}/labels/test', exist_ok=True)

# Small object categories
small_objects = [
    "toothbrush", "bottle", "wine glass", "apple", "orange", "banana",
    "fork", "spoon", "knife", "remote", "mouse", "tie", "handbag", "fire hydrant",
    "parking meter", "bird", "frisbee", "kite", "baseball glove", "sports ball",
    "bat", "cup", "bowl", "sandwich", "broccoli", "hot dog", "carrot", "cake", "donut",
    "potted plant", "cell phone", "toaster", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier"
]

# Load annotations
with open(annotations_path, 'r') as f:
    coco = json.load(f)

print(coco)

categories = coco['categories']
annotations = coco['annotations']
images = coco['images']

# Map category names to COCO IDs
category_name_to_id = {cat['name']: cat['id'] for cat in categories}

# Map COCO IDs to new compact IDs (0 to 10)
small_category_name_to_newid = {name: idx for idx, name in enumerate(small_objects)}
small_coco_id_to_newid = {category_name_to_id[name]: new_id for name, new_id in small_category_name_to_newid.items() if name in category_name_to_id}

# Map image_id -> list of small object annotations
image_id_to_small_objs = {}

for ann in annotations:
    if ann['category_id'] in small_coco_id_to_newid:
        img_id = ann['image_id']
        if img_id not in image_id_to_small_objs:
            image_id_to_small_objs[img_id] = []
        image_id_to_small_objs[img_id].append(ann)

# Get all eligible image_ids
eligible_image_ids = list(image_id_to_small_objs.keys())
print(f"Found {len(eligible_image_ids)} eligible images with small objects.")

# Random sampling
random.seed(42)
random.shuffle(eligible_image_ids)

train_ids = eligible_image_ids[:5000]
val_ids = eligible_image_ids[5000:5500]
test_ids = eligible_image_ids[5500:6500]

# Helper: image id -> image info
image_id_to_info = {img['id']: img for img in images}

# Copy images and create YOLO labels
def create_yolo_labels(image_ids, split):
    for img_id in tqdm(image_ids, desc=f'Processing {split}'):
        img_info = image_id_to_info[img_id]
        file_name = img_info['file_name']

        # Copy image
        src_img = os.path.join(images_dir, file_name)
        dst_img = os.path.join(output_root, 'images', split, file_name)
        shutil.copy(src_img, dst_img)

        # Create label file
        anns = image_id_to_small_objs[img_id]
        label_lines = []
        for ann in anns:
            coco_category_id = ann['category_id']
            if coco_category_id not in small_coco_id_to_newid:
                continue  # skip if somehow not in mapping

            new_class_id = small_coco_id_to_newid[coco_category_id]

            bbox = ann['bbox']  # [x_min, y_min, width, height]
            x_min, y_min, w, h = bbox
            x_center = x_min + w / 2
            y_center = y_min + h / 2
            img_w = img_info['width']
            img_h = img_info['height']

            x_center /= img_w
            y_center /= img_h
            w /= img_w
            h /= img_h

            label_line = f"{new_class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            label_lines.append(label_line)

        # Save label
        base_name = Path(file_name).stem
        label_path = os.path.join(output_root, 'labels', split, f"{base_name}.txt")
        with open(label_path, 'w') as f:
            f.write('\n'.join(label_lines))

# Run
create_yolo_labels(train_ids, 'train')
create_yolo_labels(val_ids, 'val')
create_yolo_labels(test_ids, 'test')

# Save small object class names
with open(os.path.join(output_root, 'small_objects.names'), 'w') as f:
    for name in small_objects:
        f.write(name + '\n')
