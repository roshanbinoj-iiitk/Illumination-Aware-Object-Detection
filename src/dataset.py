"""
ExDark Dataset Loader and Illumination Stratifier
Pairs raw images from ExDark with YOLO annotations and metadata.
"""

import os
import glob
import re
import shutil
from pathlib import Path
from PIL import Image

LIGHT_CONDITIONS = {
    1: 'Low',
    2: 'Ambient',
    3: 'Object',
    4: 'Single',
    5: 'Weak',
    6: 'Strong',
    7: 'Screen',
    8: 'Window',
    9: 'Shadow',
    10: 'Twilight'
}

CLASSES = [
    'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat',
    'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table'
]

CLASS_MAP = {i + 1: c for i, c in enumerate(CLASSES)}


def load_imageclasslist(metadata_path):
    """Load imageclasslist.txt metadata."""
    metadata = {}
    with open(metadata_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('Name'):
                continue
            parts = line.split()
            if len(parts) >= 5:
                fname = parts[0]
                prefix = fname.split('.')[0]
                metadata[prefix] = {
                    'filename': fname,
                    'class_id': int(parts[1]),
                    'class_name': CLASS_MAP.get(int(parts[1]), 'Unknown'),
                    'light_id': int(parts[2]),
                    'light_name': LIGHT_CONDITIONS.get(int(parts[2]), 'Unknown'),
                    'inout_id': int(parts[3]),
                    'inout': 'Indoor' if int(parts[3]) == 1 else 'Outdoor',
                    'split_id': int(parts[4])
                }
    return metadata


def build_image_index(exdark_root):
    """Index all images in the ExDark directory by prefix."""
    index = {}
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
        for img_path in glob.glob(os.path.join(exdark_root, '**', ext), recursive=True):
            fname = os.path.basename(img_path)
            prefix = fname.split('.')[0]
            index[prefix] = img_path
    return index


def prepare_yolo_dataset(exdark_root, metadata_path, cache_dir, output_dir):
    """
    Prepare structured dataset for YOLO training and evaluation.
    Preserves original files by creating symlinks or copying into output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata = load_imageclasslist(metadata_path)
    img_index = build_image_index(exdark_root)

    # Find all cached label files
    label_files = glob.glob(os.path.join(cache_dir, '**', '*.txt'), recursive=True)
    label_files = [f for f in label_files if not f.endswith('README.dataset.txt') and not f.endswith('README.roboflow.txt')]

    train_img_dir = os.path.join(output_dir, 'images', 'train')
    train_lbl_dir = os.path.join(output_dir, 'labels', 'train')
    val_img_dir = os.path.join(output_dir, 'images', 'val')
    val_lbl_dir = os.path.join(output_dir, 'labels', 'val')

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    train_count = 0
    val_count = 0
    dataset_records = []

    for lf in label_files:
        basename = os.path.basename(lf)
        m = re.search(r'(2015_\d{5})', basename)
        if not m:
            continue
        prefix = m.group(1)
        if prefix not in img_index or prefix not in metadata:
            continue

        src_img = img_index[prefix]
        meta = metadata[prefix]

        # Determine split based on path in cache (test/valid)
        is_val = 'test' in lf
        target_img_dir = val_img_dir if is_val else train_img_dir
        target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

        dst_img_name = f"{prefix}_{meta['light_name']}.jpg"
        dst_img = os.path.join(target_img_dir, dst_img_name)
        dst_lbl = os.path.join(target_lbl_dir, f"{prefix}_{meta['light_name']}.txt")

        # Symlink or copy
        if not os.path.exists(dst_img):
            try:
                os.symlink(os.path.abspath(src_img), dst_img)
            except OSError:
                shutil.copy2(src_img, dst_img)

        if not os.path.exists(dst_lbl):
            shutil.copy2(lf, dst_lbl)

        record = {
            'prefix': prefix,
            'image_path': dst_img,
            'label_path': dst_lbl,
            'class_id': meta['class_id'],
            'class_name': meta['class_name'],
            'light_id': meta['light_id'],
            'light_name': meta['light_name'],
            'inout': meta['inout'],
            'split': 'val' if is_val else 'train'
        }
        dataset_records.append(record)

        if is_val:
            val_count += 1
        else:
            train_count += 1

    # Create dataset YAML for YOLO
    yaml_path = os.path.join(output_dir, 'exdark.yaml')
    yaml_content = f"""path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

names:
  0: Bicycle
  1: Boat
  2: Bottle
  3: Bus
  4: Car
  5: Cat
  6: Chair
  7: Cup
  8: Dog
  9: Motorbike
  10: People
  11: Table
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"Prepared YOLO dataset at {output_dir}:")
    print(f"  Training images: {train_count}")
    print(f"  Validation/Test images: {val_count}")
    print(f"  Dataset YAML: {yaml_path}")

    return dataset_records, yaml_path


if __name__ == '__main__':
    exdark_root = '/home/roshanbinoj/Documents/BTP/ExDark'
    metadata_path = '/home/roshanbinoj/.gemini/antigravity-ide/brain/4bbf8a75-473c-481c-965e-8a486f09dc02/scratch/exdark_repo/Groundtruth/imageclasslist.txt'
    cache_dir = '/home/roshanbinoj/.cache/huggingface/hub/datasets--dronefreak--ExDark/snapshots'
    output_dir = '/home/roshanbinoj/Documents/BTP/data_processed'
    prepare_yolo_dataset(exdark_root, metadata_path, cache_dir, output_dir)
