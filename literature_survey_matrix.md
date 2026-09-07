# Literature Survey: Illumination-Aware Object Detection in Low-Light Environments

## 1. Executive Summary & Taxonomy
Object detection in low-light environments is constrained by photon starvation, high sensor noise (readout and shot noise), low dynamic range, severe contrast degradation, and non-uniform illumination fields. In recent literature (2019–2026), existing solutions can be classified into three major paradigms:

1. **Sequential Enhancement-then-Detection (Pipeline Paradigm):**
   - *Premise:* Use a pre-processing deep network (RetinexNet, Zero-DCE, EnlightenGAN, KinD) to brighten images, then pass them to a standard detector (e.g., Faster R-CNN, YOLO).
   - *Critical Limitations:* Optimization goals conflict (perceptual/visual quality != machine perception task loss). Enhancers amplify dark-region noise, produce chromatic aberrations and halos, and introduce severe compute latency (15–80 ms overhead), making edge deployment unfeasible.
2. **Domain Adaptation & Illumination Invariance (Cross-Domain Paradigm):**
   - *Premise:* Align day-to-night features using domain adversarial training or zero-shot Retinex reflectance extraction.
   - *Critical Limitations:* Assumes paired or source-domain day data; struggles with non-uniform artificial lighting (glare, mixed shadows, screen backlights) where domain shifts vary within the same image.
3. **End-to-End Illumination-Aware & Feature-Modulated Detectors (In-Network Modulation Paradigm):**
   - *Premise:* Directly integrate illumination priors into backbone and neck feature maps to dynamically reweight spatial and channel representations without generating an explicit enhanced RGB image.
   - *State of the Art:* YES (Ye et al., 2025), DimNet (Li et al., 2024), Exposure-Aware Training (Su & Lu, 2026), NLE-YOLO (Peng et al., 2024).

---

## 2. Comparative Literature Matrix (Key Benchmarks on ExDark)

| # | Paper & Authors | Venue & Year | Core Problem Addressed | Dataset(s) | Architecture / Method | Key Contribution / Results | Main Limitations | Relevance to BTP Phase-I |
|---|---|---|---|---|---|---|---|---|
| 1 | **Getting to know low-light images with ExDark**<br>Y. P. Loh & C. S. Chan | *CVIU*<br>2019 | Lack of dedicated low-light detection benchmarks | **ExDark** (7,363 imgs, 12 classes, 10 light types) | Handcrafted + early CNNs (VGG, Faster R-CNN, SSD, YOLOv2) | First large-scale low-light dataset; established 10 lighting taxonomy | Analyzed only legacy detectors; reported low mAP on deep darkness | Foundational benchmark & taxonomy for our entire BTP evaluation |
| 2 | **YES: You should Examine Suspect cues**<br>S. Ye, W. Huang, W. Liu, et al. | *CVIU*<br>2025 | Object-background confusion & low contrast under uneven lighting | ExDark, LOL | Suspect Cue Mining + Feature-driven Context Representation | Mines contextual subtle cues to separate foreground from dark backgrounds | Complex multi-stage context mining; high training compute | Direct inspiration for localized illumination feature cues |
| 3 | **Low Illumination Target Detection with Dynamic Gradient Gain**<br>Z. Li, J. Xiang, & J. Duan | *Scientific Reports*<br>2024 | Vanishing gradients for faint, low-contrast targets in dark images | ExDark | DimNet with Dynamic Gradient Gain Allocation | Dynamically reweights loss gradients for low-illumination targets; **75.60% mAP@50** | Fixed heuristic gradient reweighting; does not alter intermediate features | Highlights importance of treating low-illumination regions distinctly |
| 4 | **Exposure-Aware Training Without Target-Domain Data**<br>Y. Su & M. Lu | *Journal of Imaging*<br>2026 | Detector vulnerability to varying exposure degradation | ExDark, synthetic low-light | Exposure-Aware Training via degradation simulation | Evaluates detector degradation across discrete exposure levels | Synthetic degradation differs from real camera sensor noise | Directly validates our approach of **illumination-stratified evaluation** |
| 5 | **Systematic Review of DL for Low-Light Enhancement & Detection**<br>S. Singh, R. Kumari, P. Pallavi, et al. | *Discover Applied Sciences*<br>2026 | Fragmented landscape of enhancement vs detection methods | ExDark, LOL, DarkFace | Comprehensive benchmarking of YOLOv8 through YOLOv11 | Evaluates performance trade-offs, mAP metrics, and latency | Survey paper; does not introduce a novel unified architecture | Provides modern state-of-the-art baselines and comparative metrics |
| 6 | **Zero-Shot Day-Night Domain Adaptation**<br>Z. Du, M. Shi, & J. Deng | *CVPR*<br>2024 | Severe day-to-night domain shift in object detection | ExDark, BDD100k, Cityscapes | Retinex-based Illumination Invariant Feature Learning | Day-to-night transfer without night annotations via reflectance extraction | Retinex decomposition assumes ideal Lambertian reflectance | Explains theoretical benefits of Retinex-based illumination separation |
| 7 | **NLE-YOLO: Low Light Detection via Fusion Feature Enhancement**<br>D. Peng, W. Ding, & T. Zhen | *Scientific Reports*<br>2024 | Image noise and severe feature degradation in low-light images | ExDark | YOLOv5 + Noise suppression + Feature fusion + Decoupled head | Mitigates noise while enhancing semantic features in dark scenes | Backbone based on older YOLOv5; lacks dynamic illumination map | Demonstrates that internal feature fusion beats external enhancement |
| 8 | **Illuminating Darkness: Enhancing Low-light Images In-the-Wild**<br>S. M. A. Sharif, A. Rehman, et al. | *WACV*<br>2026 | Visual enhancement fidelity vs downstream perception | ExDark, LOL-v2, SICE | In-the-Wild Low-Light Enhancement Network | Reports **+6.80% mAP** downstream detection gain on ExDark | Two-stage pipeline adds 35ms latency; enhancement artifacts remain | Proves that illumination recovery directly boosts detection accuracy |
| 9 | **YOLO-AS for Complex Dark Environments**<br>B. Ren, Z. Xu, J. Zhao, et al. | *Scientific Reports*<br>2025 | Detection failure under mixed, irregular lighting | ExDark, LOL | YOLO-AS: Adaptive Spatial & Channel Attention Neck | Superior feature aggregation under low contrast and shadows | Evaluates only overall aggregate mAP; obscures failure modes | Validates channel/spatial attention for dark scenes |
| 10 | **YOLO-D: Domain Adaptive Low-Light Object Detection**<br>Research Consortium | *Procedia Computer Science*<br>2025 | Illumination shift across varying capture conditions | ExDark | Multi-level domain adaptation on YOLO features | Aligns feature distributions across illumination shifts | Requires complex adversarial domain discriminator training | Shows need for illumination-conditioned feature calibration |
| 11 | **Illumination-Adaptive Transformer (IAT)**<br>Z. Cui, K. Li, L. Gu, et al. | *ECCV*<br>2022 | Computational complexity of transformers for low-light vision | LOL, ExDark | Dual-branch transformer (illumination estimation + color restoration) | Fast ISP parameter estimation; highly lightweight (<1M params) | Designed primarily for enhancement; detection was secondary | Proves lightweight illumination parameter estimation is feasible |
| 12 | **Zero-Reference Deep Curve Estimation (Zero-DCE)**<br>C. Guo, C. Li, J. Guo, et al. | *CVPR*<br>2020 | Supervised enhancers require paired normal-light ground truth | LOL, ExDark, SICE | Non-reference quadratic curve parameter estimation | Real-time (500 FPS), no paired data needed for training | Curve brightening overamplifies sensor noise in extreme dark | Benchmark baseline for pre-processing enhancement comparisons |

---

## 3. Explicit Research Gap Formulation
$$\text{Existing Approach} \longrightarrow \text{Core Limitation} \longrightarrow \text{Targeted Research Gap} \longrightarrow \text{Proposed Solution}$$

1. **Enhancement Pipeline Failure:**
   $$\text{Pre-Enhancement (Zero-DCE / Retinex)} \longrightarrow \text{Noise Amplification \& Latency Penalty} \longrightarrow \text{Latency-critical perception cannot tolerate two stages} \longrightarrow \text{In-network Feature Modulation}$$
2. **Evaluation Bias towards Aggregate mAP:**
   $$\text{Single Aggregate ExDark mAP} \longrightarrow \text{Hides severe failures in extreme darkness (Low, Shadow, Screen)} \longrightarrow \text{Lack of condition-stratified benchmark} \longrightarrow \text{10-Condition Illumination-Stratified Evaluation}$$
3. **Static Feature Representation:**
   $$\text{Standard Convolutions} \longrightarrow \text{Illumination-invariant convolutions ignore local photon deficiency} \longrightarrow \text{Need dynamic feature calibration} \longrightarrow \text{Illumination-Guided Feature Modulation (IGFM)}$$
