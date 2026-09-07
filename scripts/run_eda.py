"""
Exploratory Data Analysis (EDA) on ExDark Dataset
Generates publication-quality charts:
1. Illumination Condition Breakdown (10 types)
2. Class Distribution Across Splits
3. Per-Condition Luminance Histogram
4. Representative Image Grid showing all 10 low-light conditions
"""

import os
import glob
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

LIGHT_MAP = {
    1: 'Low', 2: 'Ambient', 3: 'Object', 4: 'Single', 5: 'Weak',
    6: 'Strong', 7: 'Screen', 8: 'Window', 9: 'Shadow', 10: 'Twilight'
}

CLASSES = [
    'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat',
    'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table'
]

CLASS_MAP = {i + 1: c for i, c in enumerate(CLASSES)}


def parse_metadata(meta_path):
    records = []
    with open(meta_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5 and not parts[0].startswith('Name'):
                records.append({
                    'name': parts[0],
                    'prefix': parts[0].split('.')[0],
                    'class_id': int(parts[1]),
                    'class_name': CLASS_MAP.get(int(parts[1]), 'Unknown'),
                    'light_id': int(parts[2]),
                    'light_name': LIGHT_MAP.get(int(parts[2]), 'Unknown'),
                    'inout': 'Indoor' if int(parts[3]) == 1 else 'Outdoor',
                    'split': int(parts[4])
                })
    return records


def generate_eda_figures(meta_path, exdark_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    records = parse_metadata(meta_path)
    total_imgs = len(records)
    print(f"Total images in ExDark official metadata: {total_imgs}")

    # 1. Illumination Type Distribution Bar & Pie Chart
    light_counts = {}
    for r in records:
        light_counts[r['light_name']] = light_counts.get(r['light_name'], 0) + 1

    sorted_lights = sorted(light_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [k for k, v in sorted_lights]
    counts = [v for k, v in sorted_lights]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    palette = sns.color_palette("mako", len(labels))
    bars = ax.barh(labels, counts, color=palette, edgecolor='black', alpha=0.85)
    ax.invert_yaxis()
    ax.set_title("ExDark Dataset: Distribution of 10 Illumination Conditions (N=7,363)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Number of Images", fontsize=11, fontweight='bold')
    ax.set_ylabel("Illumination Category", fontsize=11, fontweight='bold')

    for bar, count in zip(bars, counts):
        pct = (count / total_imgs) * 100
        ax.text(bar.get_width() + 25, bar.get_y() + bar.get_height() / 2,
                f"{count:,} ({pct:.1f}%)", va='center', ha='left', fontsize=9.5, fontweight='semibold')

    ax.set_xlim(0, max(counts) * 1.18)
    plt.tight_layout()
    p1 = os.path.join(out_dir, "illumination_distribution.png")
    fig.savefig(p1, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {p1}")

    # 2. Class Distribution
    class_counts = {}
    for r in records:
        class_counts[r['class_name']] = class_counts.get(r['class_name'], 0) + 1

    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    c_labels = [k for k, v in sorted_classes]
    c_counts = [v for k, v in sorted_classes]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    c_palette = sns.color_palette("viridis", len(c_labels))
    bars = ax.bar(c_labels, c_counts, color=c_palette, edgecolor='black', alpha=0.85)
    ax.set_title("ExDark Dataset: Distribution Across 12 Target Classes", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Number of Annotated Images", fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(c_labels)))
    ax.set_xticklabels(c_labels, rotation=35, ha='right', fontsize=10, fontweight='bold')

    for bar, count in zip(bars, c_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f"{count}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylim(0, max(c_counts) * 1.15)
    plt.tight_layout()
    p2 = os.path.join(out_dir, "class_distribution.png")
    fig.savefig(p2, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {p2}")

    # 3. Luminance Distribution Across Illumination Types
    print("Computing RGB luminance histograms across sample images...")
    # Find image paths
    img_paths = {}
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
        for p in glob.glob(os.path.join(exdark_root, '**', ext), recursive=True):
            img_paths[os.path.basename(p).split('.')[0]] = p

    # Sample up to 50 images per illumination category for luminance profile
    lum_by_type = {k: [] for k in LIGHT_MAP.values()}
    for r in records:
        prefix = r['prefix']
        if prefix in img_paths and len(lum_by_type[r['light_name']]) < 40:
            try:
                im = cv2.imread(img_paths[prefix])
                if im is not None:
                    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    mean_lum = float(np.mean(gray))
                    lum_by_type[r['light_name']].append(mean_lum)
            except Exception:
                pass

    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    ordered_types = ['Low', 'Weak', 'Shadow', 'Single', 'Ambient', 'Object', 'Screen', 'Window', 'Twilight', 'Strong']
    plot_data = [lum_by_type[t] for t in ordered_types if len(lum_by_type[t]) > 0]
    plot_labels = [t for t in ordered_types if len(lum_by_type[t]) > 0]

    box = ax.boxplot(plot_data, tick_labels=plot_labels, patch_artist=True,
                     medianprops=dict(color='black', linewidth=1.5),
                     whiskerprops=dict(color='gray', linewidth=1.2))
    colors = sns.color_palette("Spectral", len(plot_labels))
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_title("Luminance Distribution Across 10 Illumination Conditions in ExDark", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Mean Grayscale Luminance (0–255)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Illumination Condition (Ordered by Mean Luminance)", fontsize=11, fontweight='bold')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Severe Low-Light Threshold (L < 50)')
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    p3 = os.path.join(out_dir, "luminance_by_condition.png")
    fig.savefig(p3, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {p3}")

    # 4. Multi-panel Representative Image Grid
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.5), dpi=300)
    axes = axes.flatten()

    chosen_samples = {}
    for r in records:
        lname = r['light_name']
        if lname not in chosen_samples and r['prefix'] in img_paths:
            chosen_samples[lname] = (img_paths[r['prefix']], r['class_name'])
        if len(chosen_samples) == 10:
            break

    for idx, lname in enumerate(ordered_types):
        ax = axes[idx]
        if lname in chosen_samples:
            img_p, cname = chosen_samples[lname]
            img = Image.open(img_p).convert('RGB')
            # Calculate luminance
            gray = np.array(img.convert('L'))
            avg_l = float(np.mean(gray))
            ax.imshow(img)
            ax.set_title(f"Type: {lname}\nClass: {cname} | Mean Lum: {avg_l:.1f}", fontsize=9.5, fontweight='bold')
        ax.axis('off')

    plt.suptitle("Representative ExDark Samples Across 10 Environmental Illumination Regimes", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    p4 = os.path.join(out_dir, "representative_samples_grid.png")
    fig.savefig(p4, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {p4}")

    return {
        'illumination_distribution': p1,
        'class_distribution': p2,
        'luminance_by_condition': p3,
        'representative_samples_grid': p4
    }


if __name__ == '__main__':
    meta_path = '/home/roshanbinoj/Documents/BTP/imageclasslist.txt'
    exdark_root = '/home/roshanbinoj/Documents/BTP/ExDark'
    out_dir = '/home/roshanbinoj/Documents/BTP/outputs/figures'
    generate_eda_figures(meta_path, exdark_root, out_dir)
