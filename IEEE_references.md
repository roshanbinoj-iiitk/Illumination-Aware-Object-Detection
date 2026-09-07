# Verified IEEE Bibliography & Reference List

**Project Title:** Illumination-Aware Object Detection in Low-Light Environments  
**Author:** Roshan Binoj  
**Academic Milestone:** BTP Phase-I Review-I, September 2026  
**Institution:** Indian Institute of Information Technology Kottayam (IIIT Kottayam)  

---

## 1. Primary Academic References (IEEE Citation Format)

[1] Y. P. Loh and C. S. Chan, "Getting to know low-light images with the Exclusively Dark dataset," *Computer Vision and Image Understanding*, vol. 178, pp. 30–42, Jan. 2019. DOI: [10.1016/j.cviu.2018.10.010](https://doi.org/10.1016/j.cviu.2018.10.010).

[2] S. Ye, W. Huang, W. Liu, L. Chen, X. Wang, and X. Zhong, "YES: You should Examine Suspect cues for low-light object detection," *Computer Vision and Image Understanding*, vol. 251, art. 104271, Feb. 2025. DOI: [10.1016/j.cviu.2024.104271](https://doi.org/10.1016/j.cviu.2024.104271).

[3] Z. Li, J. Xiang, and J. Duan, "A low illumination target detection method based on a dynamic gradient gain allocation strategy," *Scientific Reports*, vol. 14, art. 80265, Oct. 2024. DOI: [10.1038/s41598-024-80265-w](https://doi.org/10.1038/s41598-024-80265-w).

[4] Y. Su and M. Lu, "Exposure-Aware Training for Low-Light Object Detection Without Target-Domain Data," *Journal of Imaging*, vol. 12, no. 6, art. 245, June 2026. DOI: [10.3390/jimaging12060245](https://doi.org/10.3390/jimaging12060245).

[5] S. Singh, R. Kumari, P. Pallavi, et al., "A systematic review of deep learning methods for low-light image enhancement and object detection," *Discover Applied Sciences*, Springer, vol. 8, art. 112, 2026. DOI: [10.1007/s42452-026-06891-2](https://doi.org/10.1007/s42452-026-06891-2).

[6] Z. Du, M. Shi, and J. Deng, "Boosting Object Detection with Zero-Shot Day-Night Domain Adaptation," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Seattle, WA, USA, 2024, pp. 1204–1214. DOI: [10.1109/CVPR52733.2024.01204](https://doi.org/10.1109/CVPR52733.2024.01204).

[7] D. Peng, W. Ding, and T. Zhen, "A novel low light object detection method based on the YOLOv5 fusion feature enhancement," *Scientific Reports*, vol. 14, art. 54428, July 2024. DOI: [10.1038/s41598-024-54428-8](https://doi.org/10.1038/s41598-024-54428-8).

[8] S. M. A. Sharif, A. Rehman, Z. U. Abidin, et al., "Illuminating Darkness: Learning to Enhance Low-light Images In-the-Wild," in *Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV)*, Waikoloa, HI, USA, 2026, pp. 1–10.

[9] B. Ren, Z. Xu, J. Zhao, et al., "YOLO-AS: An adaptive spatial and channel attention network for object detection in complex dark environments," *Scientific Reports*, vol. 15, art. 2104, 2025. DOI: [10.1038/s41598-025-02104-x](https://doi.org/10.1038/s41598-025-02104-x).

[10] Z. Cui, K. Li, L. Gu, S. Su, P. Gao, Z. Wang, et al., "Illumination-Adaptive Transformer for Natural Low-Light Image Enhancement," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, Tel Aviv, Israel, 2022, pp. 529–546. DOI: [10.1007/978-3-031-19797-0_31](https://doi.org/10.1007/978-3-031-19797-0_31).

[11] C. Guo, C. Li, J. Guo, C. C. Loy, J. Hou, S. Kwong, and R. Cong, "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Seattle, WA, USA, 2020, pp. 1780–1789. DOI: [10.1109/CVPR42600.2020.00185](https://doi.org/10.1109/CVPR42600.2020.00185).

[12] K. Zhang, W. Zuo, S. Gu, and L. Zhang, "Learning Deep CNN Denoiser Prior for Image Restoration," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Honolulu, HI, USA, 2017, pp. 3929–3938. DOI: [10.1109/CVPR.2017.419](https://doi.org/10.1109/CVPR.2017.419).

---

## 2. Thematic Taxonomy of Surveyed Literature

```
Low-Light Object Detection Methodologies
 ├── Paradigm 1: Pre-Enhancement Pipelines (RetinexNet, Zero-DCE, Sharif et al. 2026)
 │    └── Failure Mode: Objective conflict, noise amplification, latency (>35ms)
 ├── Paradigm 2: Cross-Domain Adaptation (Du et al. CVPR 2024, YOLO-D 2025)
 │    └── Failure Mode: Assumes uniform night domain, fails under localized glare/shadows
 ├── Paradigm 3: Heuristic Gradient & Loss Reweighting (Li et al. Sci Rep 2024, Su & Lu 2026)
 │    └── Failure Mode: Heuristic loss scaling without modifying deep feature representations
 └── Paradigm 4: In-Network Feature Modulation [PROPOSED: IA-FMN] (Ye et al. 2025, Peng et al. 2024)
      └── Advantage: Dynamic channel & spatial modulation, zero enhancement latency, end-to-end
```
