"""
style_loss.py
-------------
VGG-based perceptual losses for 3DGS style transfer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T


# ImageNet normalization expected by VGG
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])


def normalize_for_vgg(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


class VGGFeatures(nn.Module):
    """
    Extracts multi-scale features from a pretrained VGG-19.

    Returns features at:
        relu1_2  (slice1) - fine textures
        relu2_2  (slice2) - mid textures
        relu3_4  (slice3) - coarse textures
        relu4_4  (slice4) - semantic content
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])   # relu2_2
        self.slice3 = nn.Sequential(*list(vgg.children())[9:18])  # relu3_4
        self.slice4 = nn.Sequential(*list(vgg.children())[18:27])  # relu4_4
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor):
        """
        x: (B, 3, H, W), ImageNet-normalized
        Returns: (feat1, feat2, feat3)
        """
        h1 = self.slice1(x)   # relu1_2  : layers 0-3
        h2 = self.slice2(h1)   # relu2_2  : layers 0-8
        h3 = self.slice3(h2)   # relu3_4  : layers 0-17
        h4 = self.slice4(h3)
        return h1, h2, h3, h4


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """
    feat: (B, C, H, W)
    Returns normalized Gram matrix: (B, C, C)
    """
    B, C, H, W = feat.shape
    f = feat.view(B, C, H * W)
    G = torch.bmm(f, f.transpose(1, 2))
    return G / (H * W)


def compute_style_loss(
    gen_feats: tuple,
    style_grams: list,
    style_layer_weights: list = None,
) -> torch.Tensor:
    """
    gen_feats:           tuple of 3 feature maps from VGGFeatures(rendered_image)
    style_grams:         list of precomputed gram matrices from style image (3 shallow layers)
    style_layer_weights: per-layer weights (default: equal)

    Returns scalar style loss.
    """
    if style_layer_weights is None:
        style_layer_weights = [1.0, 1.0, 1.0]

    loss = torch.tensor(0.0, device=gen_feats[0].device)
    for i in range(3):  # shallow layers only
        Ggen   = gram_matrix(gen_feats[i])
        Gstyle = style_grams[i].to(gen_feats[i].device)
        loss   = loss + style_layer_weights[i] * F.mse_loss(Ggen, Gstyle)
    return loss


def compute_content_loss(
    gen_feat: torch.Tensor,
    content_feat: torch.Tensor,
) -> torch.Tensor:
    """
    Compares deep (relu4_4) features.
    gen_feat, content_feat: (B, C, H, W)
    """
    return F.mse_loss(gen_feat, content_feat)


@torch.no_grad()
def precompute_style_grams(
    style_img: torch.Tensor,
    vgg: VGGFeatures,
) -> list:
    """
    Precompute and cache Gram matrices for the style image.

    style_img: (3, H, W) or (1, 3, H, W), values in [0, 1]
    Returns: list of 3 Gram matrices (for shallow VGG layers)
    """
    if style_img.dim() == 3:
        style_img = style_img.unsqueeze(0)
    x = normalize_for_vgg(style_img)
    feats = vgg(x)
    return [gram_matrix(f).detach() for f in feats[:3]]