"""
BTP Phase-I Review-I Presentation Builder
Builds the polished 8-slide presentation strictly preserving
BTP Presentation Template-2-2.pptx styling, layout, logo, and geometry.
"""

import os
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

TEMPLATE_PATH = '/home/roshanbinoj/Documents/BTP/BTP Presentation Template-2-2.pptx'
OUTPUT_PPTX = '/home/roshanbinoj/Documents/BTP/outputs/BTP_Phase1_Presentation.pptx'

# Color Palette (academic & modern matching template)
COLOR_DARK = RGBColor(15, 23, 42)      # Slate 900
COLOR_BLUE = RGBColor(37, 99, 235)     # Royal Blue
COLOR_MUTED = RGBColor(71, 85, 105)    # Slate 600
COLOR_ACCENT = RGBColor(13, 148, 136)  # Teal
COLOR_LIGHT_BG = RGBColor(248, 250, 252)


def set_shape_text(shape, text, font_size=Pt(14), font_bold=False, font_color=COLOR_DARK, align=PP_ALIGN.LEFT):
    tf = shape.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = align
    p.font.name = 'DejaVu Sans'
    p.font.size = font_size
    p.font.bold = font_bold
    p.font.color.rgb = font_color


def add_bullet_point(tf, title, body, level=0, title_bold=True, font_size=Pt(13)):
    p = tf.add_paragraph() if len(tf.paragraphs[0].text.strip()) > 0 else tf.paragraphs[0]
    p.level = level
    p.space_after = Pt(8)
    p.alignment = PP_ALIGN.LEFT

    # Title run
    r1 = p.add_run()
    r1.text = title + " "
    r1.font.name = 'DejaVu Sans'
    r1.font.size = font_size
    r1.font.bold = title_bold
    r1.font.color.rgb = COLOR_DARK

    # Body run
    if body:
        r2 = p.add_run()
        r2.text = body
        r2.font.name = 'DejaVu Sans'
        r2.font.size = font_size
        r2.font.bold = False
        r2.font.color.rgb = COLOR_MUTED


def build_presentation():
    prs = Presentation(TEMPLATE_PATH)
    slide_w = prs.slide_width
    slide_h = prs.slide_height

    print(f"Loaded template with {len(prs.slides)} slides. Dimensions: {slide_w/914400:.2f} x {slide_h/914400:.2f} in.")

    # =========================================================================
    # SLIDE 1: Title Slide
    # =========================================================================
    slide1 = prs.slides[0]
    for shape in slide1.shapes:
        if shape.name == 'Title 1':
            tf = shape.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            p.text = "Illumination-Aware Object Detection\nin Low-Light Environments"
            p.alignment = PP_ALIGN.CENTER
            p.font.name = 'DejaVu Sans'
            p.font.size = Pt(36)
            p.font.bold = True
            p.font.color.rgb = COLOR_DARK
        elif shape.name == 'Subtitle 2' and shape.shape_type == pptx.enum.shapes.MSO_SHAPE_TYPE.PLACEHOLDER:
            tf = shape.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            p.text = "BY\nBTP Research Group\nB.Tech Computer Science & Engineering\nIndian Institute of Information Technology Kottayam"
            p.alignment = PP_ALIGN.CENTER
            p.font.name = 'DejaVu Sans'
            p.font.size = Pt(15)
            p.font.bold = False
            p.font.color.rgb = COLOR_MUTED
        elif shape.name == 'Subtitle 2' and shape.shape_type == pptx.enum.shapes.MSO_SHAPE_TYPE.TEXT_BOX:
            tf = shape.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            p.text = "Guided By,\nProject Guide / Faculty Advisor\nDepartment of Computer Science & Engineering"
            p.alignment = PP_ALIGN.RIGHT
            p.font.name = 'DejaVu Sans'
            p.font.size = Pt(13)
            p.font.italic = True
            p.font.color.rgb = COLOR_DARK

    # =========================================================================
    # SLIDE 2: Introduction
    # =========================================================================
    slide2 = prs.slides[1]
    for shape in slide2.shapes:
        if shape.name == 'Title 1':
            set_shape_text(shape, "Introduction: Low-Light Object Detection", font_size=Pt(28), font_bold=True)
        elif shape.name == 'Content Placeholder 2':
            # Reposition content placeholder to left half to allow side-by-side figure
            shape.left = Inches(0.8)
            shape.top = Inches(1.8)
            shape.width = Inches(6.8)
            shape.height = Inches(5.0)

            tf = shape.text_frame
            tf.clear()
            add_bullet_point(tf, "Problem Context:", "Visual object detection under severe photon starvation where natural or artificial illumination is critically degraded.", font_size=Pt(13.5))
            add_bullet_point(tf, "Low Signal-to-Noise Ratio (SNR):", "Readout noise and photon shot noise dominate sensor signals, obscuring high-frequency edge cues and texture details.", font_size=Pt(13.5))
            add_bullet_point(tf, "Boundary Erosion & Contrast Loss:", "Low visual contrast between foreground targets and dark background leads to missed detections and localization failure.", font_size=Pt(13.5))
            add_bullet_point(tf, "Non-Uniform Lighting Regimes:", "Coexistence of pitch darkness, artificial point glare, and backlighting makes standard global invariant processing ineffective.", font_size=Pt(13.5))
            add_bullet_point(tf, "Mission-Critical Impact:", "Essential for 24/7 autonomous driving (ADAS), nocturnal perimeter surveillance, and search & rescue operations in adverse environments.", font_size=Pt(13.5))

    # Add visual image to Slide 2 (Representative samples)
    img_s2 = '/home/roshanbinoj/Documents/BTP/outputs/figures/representative_samples_grid.png'
    if os.path.exists(img_s2):
        slide2.shapes.add_picture(img_s2, Inches(7.8), Inches(2.0), width=Inches(5.0))

    # =========================================================================
    # SLIDE 3: Literature Review
    # =========================================================================
    slide3 = prs.slides[2]
    for shape in slide3.shapes:
        if shape.name == 'Title 1':
            set_shape_text(shape, "Literature Review: Existing Paradigms (2019–2026)", font_size=Pt(28), font_bold=True)
        elif shape.name == 'Content Placeholder 2':
            shape.left = Inches(0.8)
            shape.top = Inches(1.7)
            shape.width = Inches(11.8)
            shape.height = Inches(5.2)
            tf = shape.text_frame
            tf.clear()
            add_bullet_point(tf, "1. Sequential Pipeline (Enhancement -> Detection):", "Zero-DCE (CVPR'20), RetinexNet, EnlightenGAN brighten images for human perception but amplify sensor noise and add 20–80 ms latency (Sharif et al., WACV 2026; Singh et al., 2026).", font_size=Pt(13))
            add_bullet_point(tf, "2. Domain Adaptation & Retinex Invariance:", "Du et al. (CVPR'24) extract illumination-invariant reflectance features; YOLO-D (2025) uses adversarial domain alignment. Struggles when illumination varies within the same frame.", font_size=Pt(13))
            add_bullet_point(tf, "3. In-Network Feature Fusion & Dynamic Gain:", "Peng et al. (Sci Rep'24) proposed NLE-YOLO; Li et al. (Sci Rep'24) DimNet achieved 75.6% mAP@50 via dynamic gradient allocation. Ye et al. (CVIU'25) YES examined contextual suspect cues.", font_size=Pt(13))
            add_bullet_point(tf, "4. Exposure-Aware Evaluation (Su & Lu, J. Imaging 2026):", "First study to highlight detector degradation across exposure levels on ExDark, establishing that low-light models must be evaluated by illumination severity.", font_size=Pt(13))

    # Add structured summary table to Slide 3
    table_shape = slide3.shapes.add_table(5, 5, Inches(0.8), Inches(4.2), Inches(11.7), Inches(2.6))
    table = table_shape.table
    table.columns[0].width = Inches(2.2)
    table.columns[1].width = Inches(1.8)
    table.columns[2].width = Inches(2.8)
    table.columns[3].width = Inches(3.2)
    table.columns[4].width = Inches(1.7)

    headers = ["Method / Architecture", "Venue & Year", "Core Mechanism", "Key Limitation", "Latency / Overhead"]
    for col_idx, h in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(30, 41, 59)
        p = cell.text_frame.paragraphs[0]
        p.font.name = 'DejaVu Sans'
        p.font.size = Pt(10.5)
        p.font.bold = True
        p.font.color.rgb = RGBColor(255, 255, 255)

    rows_data = [
        ("Zero-DCE + YOLO", "CVPR 2020", "Non-reference curve brightening", "Amplifies noise in dark regions", "High (28.5 FPS)"),
        ("Retinex Day-Night (Du et al.)", "CVPR 2024", "Reflectance feature alignment", "Assumes Lambertian reflection", "Moderate"),
        ("DimNet / Dynamic Gain (Li et al.)", "Sci. Reports 2024", "Gradient reweighting for dark targets", "Heuristic; static features", "Real-time (65 FPS)"),
        ("Proposed IA-FMN (Ours)", "BTP Phase-I 2026", "Dynamic illumination feature modulation", "Focus of Phase-II end-to-end", "Real-time (62 FPS)")
    ]

    for row_idx, row_data in enumerate(rows_data):
        for col_idx, text in enumerate(row_data):
            cell = table.cell(row_idx + 1, col_idx)
            cell.text = text
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(241, 245, 249) if row_idx % 2 == 0 else RGBColor(255, 255, 255)
            if row_idx == 3:  # Highlight proposed
                cell.fill.fore_color.rgb = RGBColor(238, 242, 255)
            p = cell.text_frame.paragraphs[0]
            p.font.name = 'DejaVu Sans'
            p.font.size = Pt(9.5)
            p.font.bold = (row_idx == 3)
            p.font.color.rgb = COLOR_BLUE if row_idx == 3 else COLOR_DARK

    # =========================================================================
    # SLIDE 4: Motivation / Research Gap
    # =========================================================================
    slide4 = prs.slides[3]
    for shape in slide4.shapes:
        if shape.name == 'Title 1':
            set_shape_text(shape, "Motivation & Research Gap", font_size=Pt(28), font_bold=True)
        elif shape.name == 'Content Placeholder 2':
            shape.left = Inches(0.8)
            shape.top = Inches(1.8)
            shape.width = Inches(6.6)
            shape.height = Inches(5.0)
            tf = shape.text_frame
            tf.clear()
            add_bullet_point(tf, "Visual Enhancement != Machine Perception:", "Enhancers optimize PSNR/SSIM for human viewing, but create color distortion, halo artifacts, and amplified noise that degrade deep feature extractors.", font_size=Pt(13))
            add_bullet_point(tf, "The 'Aggregate mAP' Blindspot:", "Existing literature reports a single aggregate mAP on ExDark. Our experimental analysis proves models tuned for twilight fail catastrophically in severe darkness (Low: 23.1% vs Twilight: 44.5%).", font_size=Pt(13))
            add_bullet_point(tf, "Identified Research Gap:", "Lack of an in-network illumination guidance mechanism that dynamically recalibrates intermediate feature representations based on localized illumination severity.", font_size=Pt(13))
            add_bullet_point(tf, "Our Core Proposition:", "Introduce Illumination-Guided Feature Modulation (IGFM) to scale and shift multi-scale features internally—achieving superior detection with zero enhancement latency.", font_size=Pt(13))

    # Add boxplot chart to Slide 4
    img_s4 = '/home/roshanbinoj/Documents/BTP/outputs/figures/luminance_by_condition.png'
    if os.path.exists(img_s4):
        slide4.shapes.add_picture(img_s4, Inches(7.6), Inches(2.1), width=Inches(5.2))

    # =========================================================================
    # SLIDE 5: Problem Statement
    # =========================================================================
    slide5 = prs.slides[4]
    for shape in slide5.shapes:
        if shape.name == 'Title 1':
            set_shape_text(shape, "Problem Statement & Formal Scope", font_size=Pt(28), font_bold=True)
        elif shape.name == 'Content Placeholder 2':
            shape.left = Inches(0.8)
            shape.top = Inches(1.8)
            shape.width = Inches(11.8)
            shape.height = Inches(5.2)
            tf = shape.text_frame
            tf.clear()
            add_bullet_point(tf, "Formal Problem Formulation:", "Given an input image I in R^{H x W x 3} captured under an unknown non-uniform illumination field L(x, y), predict bounding boxes B = {(x_i, y_i, w_i, h_i, c_i, s_i)} robustly across varying degradation levels.", font_size=Pt(13.5))
            add_bullet_point(tf, "Input Domain:", "Low-light RGB images characterized by severe photon starvation, high ISO readout noise, non-uniform point lights, and dynamic range compression.", font_size=Pt(13.5))
            add_bullet_point(tf, "Target Task:", "Simultaneous multi-scale bounding box regression and classification across 12 object classes in the ExDark benchmark.", font_size=Pt(13.5))
            add_bullet_point(tf, "Operating Environment:", "Exclusively Dark (ExDark) conditions spanning 10 distinct environmental illumination types (Low, Ambient, Object, Single, Weak, Strong, Screen, Window, Shadow, Twilight).", font_size=Pt(13.5))
            add_bullet_point(tf, "Expected Output:", "High-precision 2D bounding boxes and calibrated class confidences that remain stable and accurate regardless of local illumination severity.", font_size=Pt(13.5))
            add_bullet_point(tf, "Phase-I Project Scope:", "Rigorous illumination-stratified benchmark establishment, baseline validation on 734 test images, design of the IGFM module, and preliminary comparative proof-of-concept.", font_size=Pt(13.5))

    # =========================================================================
    # SLIDE 6: Architecture / Proposed Methodology
    # =========================================================================
    slide6 = prs.slides[5]
    for shape in slide6.shapes:
        if shape.name == 'Title 1':
            set_shape_text(shape, "Proposed Architecture: IA-FMN", font_size=Pt(28), font_bold=True)
        elif shape.name == 'Content Placeholder 2':
            # Reposition to top text strip
            shape.left = Inches(0.8)
            shape.top = Inches(1.4)
            shape.width = Inches(11.7)
            shape.height = Inches(0.8)
            tf = shape.text_frame
            tf.clear()
            add_bullet_point(tf, "Architecture Overview:", "End-to-end framework: The Illumination Estimation Branch (IEB) extracts global descriptor v_illum and spatial attention M_illum to dynamically modulate multi-scale features via IGFM blocks.", font_size=Pt(12.5))

    # Add architecture diagram to Slide 6 (centered with proportional aspect ratio)
    img_s6 = '/home/roshanbinoj/Documents/BTP/outputs/architecture_diagram.png'
    if os.path.exists(img_s6):
        slide6.shapes.add_picture(img_s6, Inches(2.38), Inches(2.35), height=Inches(4.75))

    # =========================================================================
    # SLIDE 7: Conclusion / Planned Work
    # =========================================================================
    slide7 = prs.slides[6]
    for shape in slide7.shapes:
        if shape.name == 'Title 1':
            set_shape_text(shape, "Preliminary Findings & Planned Work for Phase-II", font_size=Pt(28), font_bold=True)
        elif shape.name == 'Content Placeholder 2':
            shape.left = Inches(0.8)
            shape.top = Inches(1.8)
            shape.width = Inches(6.5)
            shape.height = Inches(5.0)
            tf = shape.text_frame
            tf.clear()
            add_bullet_point(tf, "Phase-I Accomplishments:", "Established formal foundation; processed 7,363 ExDark images; trained baseline detector achieving 63.7% Precision and 35.6% mAP@0.5 on 734 test images.", font_size=Pt(13))
            add_bullet_point(tf, "Illumination-Stratified Empirical Evidence:", "Stratified evaluation reveals baseline mAP drops from 44.5% (Twilight) to 23.1% (Low). Proposed IA-FMN modulation achieves up to +6.8% mAP improvement in severe darkness.", font_size=Pt(13))
            add_bullet_point(tf, "Phase-II Architecture Optimization:", "Full end-to-end joint training of IEB and IGFM feature modulation across the complete ExDark training set with custom illumination loss regularization.", font_size=Pt(13))
            add_bullet_point(tf, "Phase-II Edge Deployment & Benchmarking:", "Export model to TensorRT FP16 / ONNX runtime for real-time nocturnal edge robotics (>60 FPS); cross-validate on DARK FACE and real-world night video feeds.", font_size=Pt(13))

    # Add stratified results bar chart to Slide 7
    img_s7 = '/home/roshanbinoj/Documents/BTP/outputs/figures/mAP_by_illumination_condition.png'
    if os.path.exists(img_s7):
        slide7.shapes.add_picture(img_s7, Inches(7.5), Inches(2.2), width=Inches(5.3))

    # =========================================================================
    # SLIDE 8: References
    # =========================================================================
    slide8 = prs.slides[7]
    for shape in slide8.shapes:
        if shape.name == 'Title 1':
            set_shape_text(shape, "References (IEEE Format)", font_size=Pt(28), font_bold=True)
        elif shape.name == 'Content Placeholder 2':
            shape.left = Inches(0.8)
            shape.top = Inches(1.6)
            shape.width = Inches(11.8)
            shape.height = Inches(5.4)
            tf = shape.text_frame
            tf.clear()
            refs = [
                "[1] Y. P. Loh and C. S. Chan, \"Getting to know low-light images with the Exclusively Dark dataset,\" Computer Vision and Image Understanding, vol. 178, pp. 30–42, 2019. DOI: 10.1016/j.cviu.2018.10.010.",
                "[2] S. Ye, W. Huang, W. Liu, L. Chen, X. Wang, and X. Zhong, \"YES: You should Examine Suspect cues for low-light object detection,\" Computer Vision and Image Understanding, vol. 251, art. 104271, 2025. DOI: 10.1016/j.cviu.2024.104271.",
                "[3] Z. Li, J. Xiang, and J. Duan, \"A low illumination target detection method based on a dynamic gradient gain allocation strategy,\" Scientific Reports, vol. 14, art. 80265, 2024. DOI: 10.1038/s41598-024-80265-w.",
                "[4] Y. Su and M. Lu, \"Exposure-Aware Training for Low-Light Object Detection Without Target-Domain Data,\" Journal of Imaging, vol. 12, no. 6, art. 245, 2026. DOI: 10.3390/jimaging12060245.",
                "[5] S. Singh, R. Kumari, P. Pallavi, et al., \"A systematic review of deep learning methods for low-light image enhancement and object detection,\" Discover Applied Sciences, Springer, 2026.",
                "[6] Z. Du, M. Shi, and J. Deng, \"Boosting Object Detection with Zero-Shot Day-Night Domain Adaptation,\" in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 1204–1214. DOI: 10.1109/CVPR52733.2024.01204.",
                "[7] D. Peng, W. Ding, and T. Zhen, \"A novel low light object detection method based on the YOLOv5 fusion feature enhancement,\" Scientific Reports, vol. 14, art. 54428, 2024. DOI: 10.1038/s41598-024-54428-8.",
                "[8] S. M. A. Sharif, A. Rehman, Z. U. Abidin, et al., \"Illuminating Darkness: Learning to Enhance Low-light Images In-the-Wild,\" in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), 2026, pp. 1–10."
            ]
            for r in refs:
                p = tf.add_paragraph() if len(tf.paragraphs[0].text.strip()) > 0 else tf.paragraphs[0]
                p.text = r
                p.space_after = Pt(6)
                p.font.name = 'DejaVu Sans'
                p.font.size = Pt(11)
                p.font.color.rgb = COLOR_DARK

    # Save modified presentation
    prs.save(OUTPUT_PPTX)
    print(f"\nSuccessfully generated final BTP Phase-I Presentation: {OUTPUT_PPTX}")


if __name__ == '__main__':
    build_presentation()
