"""
Illumination-Aware Feature Modulation (IGFM) Modules
PyTorch implementation of the Illumination Estimation Branch (IEB)
and Illumination-Guided Feature Modulation (IGFM) block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IlluminationEstimationBranch(nn.Module):
    """
    Lightweight branch to estimate both global illumination descriptor
    and spatial illumination attention map directly from input features.
    """
    def __init__(self, in_channels=3, embed_dim=64):
        super().__init__()
        # Multi-scale receptive field for illumination estimation
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

        # Global descriptor path (AvgPool + MaxPool)
        self.fc_global = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

        # Spatial illumination map path
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(embed_dim, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: Input image [B, 3, H, W]
        Returns:
            v_illum: Global illumination vector [B, embed_dim]
            M_illum: Spatial illumination map [B, 1, H//4, W//4]
        """
        feats = self.conv_in(x)
        b, c, h, w = feats.shape

        # Dual pooling: average captures diffuse ambient light; max captures glare/direct points
        avg_p = F.adaptive_avg_pool2d(feats, (1, 1)).view(b, c)
        max_p = F.adaptive_max_pool2d(feats, (1, 1)).view(b, c)
        pool_cat = torch.cat([avg_p, max_p], dim=1)
        v_illum = self.fc_global(pool_cat)

        M_illum = self.spatial_conv(feats)
        return v_illum, M_illum


class IlluminationGuidedFeatureModulation(nn.Module):
    """
    Feature Modulation Block (IGFM)
    Dynamically modulates backbone feature pyramid levels using
    affine scaling, shifting, and spatial illumination attention.
    """
    def __init__(self, feature_dim, illum_embed_dim=64):
        super().__init__()
        self.feature_dim = feature_dim

        # Predict affine scale (gamma) and shift (beta)
        self.gamma_fc = nn.Sequential(
            nn.Linear(illum_embed_dim, feature_dim),
            nn.Sigmoid()  # Scale around 1.0
        )
        self.beta_fc = nn.Sequential(
            nn.Linear(illum_embed_dim, feature_dim),
            nn.Tanh()     # Small zero-centered shift
        )

        # Spatial adapter to match feature resolution
        self.spatial_adapt = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True)
        self.residual_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(feature_dim)
        self.act = nn.SiLU(inplace=True)

    def forward(self, feat, v_illum, M_illum):
        """
        feat: Backbone feature map [B, C, H, W]
        v_illum: Illumination vector [B, embed_dim]
        M_illum: Spatial illumination map [B, 1, H_m, W_m]
        """
        b, c, h, w = feat.shape

        gamma = self.gamma_fc(v_illum).view(b, c, 1, 1) * 2.0  # Range [0, 2]
        beta = self.beta_fc(v_illum).view(b, c, 1, 1) * 0.5    # Range [-0.5, 0.5]

        # Interpolate spatial illumination map to feature resolution
        M_res = F.interpolate(M_illum, size=(h, w), mode='bilinear', align_corners=False)
        M_attn = torch.sigmoid(self.spatial_adapt(M_res))

        # Affine recalibration + spatial illumination guidance
        feat_scaled = feat * gamma + beta
        feat_guided = feat_scaled * (1.0 + M_attn)

        # Residual connection
        out = self.norm(self.residual_conv(feat_guided)) + feat
        return self.act(out)
