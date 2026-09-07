"""
ExDark Dataset Loader and Illumination Stratifier
Pairs raw images from ExDark with official ExDark_Annno ground truth annotations
and formats them for YOLO training, validation, and testing.
"""

import os
import glob
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
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


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
                    'split_id': int(parts[4])  # 1: Train, 2: Val, 3: Test
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


def build_anno_index(anno_root):
    """Index all annotation files in ExDark_Annno by prefix."""
    index = {}
    for anno_path in glob.glob(os.path.join(anno_root, '**', '*.txt'), recursive=True):
        fname = os.path.basename(anno_path)
        prefix = fname.split('.')[0]
        index[prefix] = anno_path
    return index


def parse_exdark_annotation(anno_path, img_w, img_h):
    """
    Parse ExDark annotation file (% bbGt version=3 format) and convert to YOLO format.
    Format: [class_name] [l] [t] [w] [h] ...
    Returns list of formatted strings: "class_id xc yc w h"
    """
    yolo_lines = []
    with open(anno_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            parts = line.split()
            cname = parts[0]
            if cname not in CLASS_TO_IDX:
                continue
            try:
                l, t, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except (ValueError, IndexError):
                continue

            # Clip coordinates to image boundary
            x1 = max(0.0, min(float(img_w), l))
            y1 = max(0.0, min(float(img_h), t))
            x2 = max(0.0, min(float(img_w), l + w))
            y2 = max(0.0, min(float(img_h), t + h))

            box_w = x2 - x1
            box_h = y2 - y1
            if box_w <= 1.0 or box_h <= 1.0:
                continue

            # Compute normalized center and dimensions
            xc = (x1 + box_w / 2.0) / img_w
            yc = (y1 + box_h / 2.0) / img_h
            norm_w = box_w / img_w
            norm_h = box_h / img_h

            cls_idx = CLASS_TO_IDX[cname]
            yolo_lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {norm_w:.6f} {norm_h:.6f}")
    return yolo_lines


def prepare_yolo_dataset(exdark_root, anno_root, metadata_path, output_dir):
    """
    Prepare structured dataset for YOLO training, validation, and testing.
    Uses official ExDark_Annno ground truth and imageclasslist.txt splits.
    Split 1: Train (3,000 images)
    Split 2: Val (1,800 images)
    Split 3: Test (2,563 images)
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata = load_imageclasslist(metadata_path)
    img_index = build_image_index(exdark_root)
    anno_index = build_anno_index(anno_root)

    splits = {
        1: ('train', os.path.join(output_dir, 'images', 'train'), os.path.join(output_dir, 'labels', 'train')),
        2: ('val', os.path.join(output_dir, 'images', 'val'), os.path.join(output_dir, 'labels', 'val')),
        3: ('test', os.path.join(output_dir, 'images', 'test'), os.path.join(output_dir, 'labels', 'test'))
    }

    for _, (_, img_d, lbl_d) in splits.items():
        os.makedirs(img_d, exist_ok=True)
        os.makedirs(lbl_d, exist_ok=True)

    split_counts = {'train': 0, 'val': 0, 'test': 0}
    dataset_records = []

    for prefix, meta in metadata.items():
        if prefix not in img_index or prefix not in anno_index:
            continue

        src_img = img_index[prefix]
        src_anno = anno_index[prefix]
        split_id = meta['split_id']
        split_name, target_img_dir, target_lbl_dir = splits[split_id]

        dst_img_name = f"{prefix}_{meta['light_name']}.jpg"
        dst_img = os.path.join(target_img_dir, dst_img_name)
        dst_lbl = os.path.join(target_lbl_dir, f"{prefix}_{meta['light_name']}.txt")

        # Symlink or copy image
        if not os.path.exists(dst_img):
            try:
                os.symlink(os.path.abspath(src_img), dst_img)
            except OSError:
                shutil.copy2(src_img, dst_img)

        # Generate YOLO label from ExDark_Annno if not exists
        if not os.path.exists(dst_lbl):
            with Image.open(src_img) as im:
                img_w, img_h = im.size
            yolo_boxes = parse_exdark_annotation(src_anno, img_w, img_h)
            with open(dst_lbl, 'w') as f:
                f.write('\n'.join(yolo_boxes) + '\n')

        split_counts[split_name] += 1
        record = {
            'prefix': prefix,
            'image_path': dst_img,
            'label_path': dst_lbl,
            'class_id': meta['class_id'],
            'class_name': meta['class_name'],
            'light_id': meta['light_id'],
            'light_name': meta['light_name'],
            'inout': meta['inout'],
            'split': split_name
        }
        dataset_records.append(record)

    # Create dataset YAML for YOLO
    yaml_path = os.path.join(output_dir, 'exdark.yaml')
    yaml_content = f"""path: {os.path.abspath(output_dir)}
train: images/train
val: images/val
test: images/test

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

    print(f"Prepared full ExDark YOLO dataset at {output_dir}:")
    print(f"  Training images:   {split_counts['train']}")
    print(f"  Validation images: {split_counts['val']}")
    print(f"  Testing images:    {split_counts['test']}")
    print(f"  Total images:      {sum(split_counts.values())}")
    print(f"  Dataset YAML:      {yaml_path}")

    return dataset_records, yaml_path


if __name__ == '__main__':
    exdark_root = '/home/roshanbinoj/Documents/BTP/ExDark'
    anno_root = '/home/roshanbinoj/Documents/BTP/ExDark_Annno'
    metadata_path = '/home/roshanbinoj/Documents/BTP/imageclasslist.txt'
    output_dir = '/home/roshanbinoj/Documents/BTP/data_processed'
    prepare_yolo_dataset(exdark_root, anno_root, metadata_path, output_dir)
