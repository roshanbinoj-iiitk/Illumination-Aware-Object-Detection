"""
High-Resolution Architecture Diagram Generator
Generates clean, publication-ready schematic for the Proposed:
Illumination-Aware Feature Modulation Network (IA-FMN).
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle


def draw_box(ax, x, y, w, h, text, color='#2b5c8f', edge='#1a365d', text_color='white', fontsize=10, fontweight='bold', alpha=0.92, subtext=None):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.04",
                         facecolor=color, edgecolor=edge, linewidth=1.8, alpha=alpha)
    ax.add_patch(box)
    if subtext:
        ax.text(x + w / 2, y + h * 0.62, text, color=text_color, fontsize=fontsize,
                fontweight=fontweight, ha='center', va='center')
        ax.text(x + w / 2, y + h * 0.28, subtext, color=text_color, fontsize=fontsize * 0.8,
                ha='center', va='center', style='italic')
    else:
        ax.text(x + w / 2, y + h / 2, text, color=text_color, fontsize=fontsize,
                fontweight=fontweight, ha='center', va='center')
    return box


def draw_arrow(ax, start, end, color='#2d3748', lw=2.0, style='-|>', rad=0.0):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                shrinkA=3, shrinkB=3,
                                connectionstyle=f"arc3,rad={rad}"))


def generate_architecture_diagram(output_path):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Background subtle panel
    bg = patches.Rectangle((0.1, 0.1), 15.8, 8.8, facecolor='#f8fafc', edgecolor='#cbd5e1', linewidth=1.5, zorder=0)
    ax.add_patch(bg)

    # Title Header
    ax.text(8.0, 8.45, "Proposed System Architecture: Illumination-Aware Feature Modulation Network (IA-FMN)",
            ha='center', va='center', fontsize=15, fontweight='bold', color='#0f172a')
    ax.text(8.0, 8.12, "End-to-End Object Detection with Dynamic Illumination Guidance and Feature Recalibration",
            ha='center', va='center', fontsize=11, color='#475569', style='italic')

    # Color definitions
    c_input = '#334155'     # Slate
    c_ieb = '#0284c7'       # Sky blue
    c_backbone = '#2563eb'  # Blue
    c_igfm = '#7c3aed'      # Purple (Proposed Core)
    c_neck = '#0d9488'      # Teal
    c_head = '#ea580c'      # Orange

    # 1. INPUT IMAGE
    draw_box(ax, 0.6, 3.8, 1.8, 1.8, "Input Image", color=c_input, subtext=r"$\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$" + "\n(ExDark Low-Light)", fontsize=10.5)

    # 2. ILLUMINATION ESTIMATION BRANCH (IEB) [Top Branch]
    ieb_bg = FancyBboxPatch((3.0, 5.7), 4.6, 2.0, boxstyle="round,pad=0.04", facecolor='#f0f9ff', edgecolor='#38bdf8', linewidth=1.5, linestyle='--')
    ax.add_patch(ieb_bg)
    ax.text(5.3, 7.45, "Illumination Estimation Branch (IEB)", fontsize=11, fontweight='bold', color='#0369a1', ha='center')

    draw_box(ax, 3.2, 6.0, 1.8, 1.1, "Luminance Extractor", color=c_ieb, subtext="Conv3x3 + GELU", fontsize=9.5)
    draw_box(ax, 5.5, 6.6, 1.9, 0.8, "Dual-Pool Descriptor", color='#0369a1', subtext=r"$\mathbf{v}_{illum} \in \mathbb{R}^{d}$ (Avg+Max)", fontsize=8.5)
    draw_box(ax, 5.5, 5.7, 1.9, 0.8, "Spatial Attention", color='#0284c7', subtext=r"$\mathbf{M}_{illum} \in [0,1]^{H/4 \times W/4}$", fontsize=8.5)

    # 3. MULTI-SCALE BACKBONE [Bottom Branch]
    bb_bg = FancyBboxPatch((3.0, 1.4), 4.6, 3.8, boxstyle="round,pad=0.04", facecolor='#eff6ff', edgecolor='#93c5fd', linewidth=1.5, linestyle='--')
    ax.add_patch(bb_bg)
    ax.text(5.3, 4.95, "Multi-Scale Backbone (CNN / Pyramidal)", fontsize=11, fontweight='bold', color='#1d4ed8', ha='center')

    draw_box(ax, 3.4, 3.8, 1.7, 0.85, "Stage 3 (Stride 8)", color=c_backbone, subtext=r"$C_3 \in \mathbb{R}^{C \times H/8 \times W/8}$", fontsize=8.5)
    draw_box(ax, 3.4, 2.7, 1.7, 0.85, "Stage 4 (Stride 16)", color=c_backbone, subtext=r"$C_4 \in \mathbb{R}^{2C \times H/16 \times W/16}$", fontsize=8.5)
    draw_box(ax, 3.4, 1.6, 1.7, 0.85, "Stage 5 (Stride 32)", color=c_backbone, subtext=r"$C_5 \in \mathbb{R}^{4C \times H/32 \times W/32}$", fontsize=8.5)

    # 4. ILLUMINATION-GUIDED FEATURE MODULATION (IGFM) BLOCKS [Core Novelty]
    igfm_bg = FancyBboxPatch((8.2, 1.4), 2.8, 5.6, boxstyle="round,pad=0.04", facecolor='#faf5ff', edgecolor='#c084fc', linewidth=2.0)
    ax.add_patch(igfm_bg)
    ax.text(9.6, 6.75, "IGFM Modules (Novelty)", fontsize=11.5, fontweight='bold', color='#6b21a8', ha='center')
    ax.text(9.6, 6.42, r"$\widetilde{\mathbf{F}} = \gamma \odot \mathbf{F} + \beta + \mathbf{M} \otimes \mathbf{F}$", fontsize=9.5, fontweight='bold', color='#581c87', ha='center')

    draw_box(ax, 8.5, 4.8, 2.2, 1.0, "IGFM Block 3", color=c_igfm, subtext="Affine Scale + Attn (P3)", fontsize=9.5)
    draw_box(ax, 8.5, 3.3, 2.2, 1.0, "IGFM Block 4", color=c_igfm, subtext="Affine Scale + Attn (P4)", fontsize=9.5)
    draw_box(ax, 8.5, 1.8, 2.2, 1.0, "IGFM Block 5", color=c_igfm, subtext="Affine Scale + Attn (P5)", fontsize=9.5)

    # 5. FEATURE PYRAMID NETWORK / NECK
    draw_box(ax, 11.5, 2.6, 1.6, 3.2, "Feature Neck\n(PANet / FPN)", color=c_neck, subtext="Top-Down +\nBottom-Up Fusion", fontsize=10)

    # 6. MULTI-SCALE DETECTION HEADS
    head_bg = FancyBboxPatch((13.6, 1.8), 2.1, 4.8, boxstyle="round,pad=0.04", facecolor='#fff7ed', edgecolor='#fdba74', linewidth=1.5, linestyle='--')
    ax.add_patch(head_bg)
    ax.text(14.65, 6.35, "Decoupled Heads", fontsize=10.5, fontweight='bold', color='#c2410c', ha='center')

    draw_box(ax, 13.8, 4.7, 1.7, 1.1, "Small Objects", color=c_head, subtext="Boxes + 12 Classes\n(Stride 8)", fontsize=8.5)
    draw_box(ax, 13.8, 3.3, 1.7, 1.1, "Medium Objects", color=c_head, subtext="Boxes + 12 Classes\n(Stride 16)", fontsize=8.5)
    draw_box(ax, 13.8, 1.9, 1.7, 1.1, "Large Objects", color=c_head, subtext="Boxes + 12 Classes\n(Stride 32)", fontsize=8.5)

    # CONNECTING ARROWS & DATAFLOWS
    # Input -> IEB & Input -> Backbone
    draw_arrow(ax, (2.4, 4.7), (3.2, 6.4), color='#0284c7', rad=0.15)
    draw_arrow(ax, (2.4, 4.7), (3.4, 4.2), color='#2563eb', rad=-0.1)

    # IEB internal
    draw_arrow(ax, (5.0, 6.6), (5.5, 6.8), color='#0284c7')
    draw_arrow(ax, (5.0, 6.3), (5.5, 6.1), color='#0284c7')

    # Backbone stages
    draw_arrow(ax, (4.25, 3.8), (4.25, 3.55), color='#2563eb')
    draw_arrow(ax, (4.25, 2.7), (4.25, 2.45), color='#2563eb')

    # Backbone -> IGFM
    draw_arrow(ax, (5.1, 4.2), (8.5, 5.2), color='#2563eb')
    draw_arrow(ax, (5.1, 3.1), (8.5, 3.8), color='#2563eb')
    draw_arrow(ax, (5.1, 2.0), (8.5, 2.3), color='#2563eb')

    # IEB -> IGFM (Illumination guidance lines)
    draw_arrow(ax, (7.4, 6.9), (8.5, 5.5), color='#7c3aed', rad=0.1, style='-|>')
    draw_arrow(ax, (7.4, 6.1), (8.5, 4.1), color='#7c3aed', rad=0.08, style='-|>')
    draw_arrow(ax, (7.4, 5.9), (8.5, 2.6), color='#7c3aed', rad=-0.15, style='-|>')

    # IGFM -> Neck
    draw_arrow(ax, (10.7, 5.3), (11.5, 4.8), color='#7c3aed')
    draw_arrow(ax, (10.7, 3.8), (11.5, 4.2), color='#7c3aed')
    draw_arrow(ax, (10.7, 2.3), (11.5, 3.4), color='#7c3aed')

    # Neck -> Detection Heads
    draw_arrow(ax, (13.1, 5.0), (13.8, 5.2), color='#ea580c')
    draw_arrow(ax, (13.1, 4.2), (13.8, 3.8), color='#ea580c')
    draw_arrow(ax, (13.1, 3.2), (13.8, 2.4), color='#ea580c')

    # Legend / Key Insights footer
    ax.text(0.8, 0.75, "Key Architecture Insights:", fontsize=10, fontweight='bold', color='#1e293b')
    ax.text(0.8, 0.42, "• Zero enhancement latency: Eliminates external enhancement stages (e.g., RetinexNet/Zero-DCE) by learning illumination modulation internally.", fontsize=8.8, color='#334155')
    ax.text(0.8, 0.18, "• Illumination-Adaptive: Dual global vector (ambient/glare) and spatial attention map dynamically scale object features and suppress dark noise.", fontsize=8.8, color='#334155')

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Generated architecture diagram: {output_path}")


if __name__ == '__main__':
    out_path = '/home/roshanbinoj/Documents/BTP/outputs/architecture_diagram.png'
    generate_architecture_diagram(out_path)
