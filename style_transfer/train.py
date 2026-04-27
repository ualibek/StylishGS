"""
train_style.py
--------------
Style transfer fine-tuning on top of a pretrained 3DGS scene.
Place in the root of the 3DGS repository (same level as train.py).

Usage
-----
    python train_style.py \
        -s data/my_scene \
        -m output/my_scene \
        --style_image style/vangogh.jpg
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

from arguments            import ModelParams, PipelineParams
from gaussian_renderer    import render
from scene                import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils  import safe_state
from utils.loss_utils     import l1_loss

from style_transfer.style_loss        import VGGFeatures, normalize_for_vgg, precompute_style_grams, compute_style_loss
from style_transfer.camera_sampler    import TrainCameraSampler
from style_transfer.reference_renders import ReferenceRenderCache

import matplotlib.pyplot as plt

# ── Hardcoded hyperparameters ─────────────────────────────────────────────────
ITERATIONS     = 100
VIEWS_PER_STEP = 4
LAMBDA_STYLE   = 1e4
LR             = 1e-3
VGG_MAX_DIM    = 512
SAVE_EVERY     = ITERATIONS // 4
# ─────────────────────────────────────────────────────────────────────────────

# helper functions

def save_loss_curves(l1_losses, style_losses, total_losses, out_dir="visualizations"):
    steps = range(1, len(l1_losses) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(steps, l1_losses, color="steelblue")
    axes[0].set_title("L1 Loss")
    axes[0].set_xlabel("Step")

    axes[1].plot(steps, style_losses, color="darkorange")
    axes[1].set_title("Style Loss")
    axes[1].set_xlabel("Step")

    axes[2].plot(steps, total_losses, color="seagreen")
    axes[2].set_title("Total Loss")
    axes[2].set_xlabel("Step")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=150)
    plt.close()

def copy_meta(src_model_path: str, dst_path: str):
    """
    Copy static SIBR meta files from a pretrained scene to the output folder.
    Call once before the training loop.
    """
    import shutil
    os.makedirs(dst_path, exist_ok=True)
    for fname in ["cameras.json", "cfg_args", "exposure.json"]:
        src = os.path.join(src_model_path, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(dst_path, fname))
    print(f"Copied meta files from {src_model_path} to {dst_path}")


def train_style(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load pretrained scene ─────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    sys.argv = [sys.argv[0], "-s", args.source_path, "-m", args.model_path]
    mp = ModelParams(parser).extract(parser.parse_args())
    pp = PipelineParams(parser).extract(parser.parse_args())

    safe_state(silent=True)

    gaussians = GaussianModel(mp.sh_degree)
    scene     = Scene(mp, gaussians, load_iteration=-1, shuffle=False)
    background = torch.ones(3, device=device) if mp.white_background else torch.zeros(3, device=device)

    print(f"Loaded {gaussians.get_xyz.shape[0]:,} Gaussians.")

    # ── Freeze geometry, only train SH colour ────────────────────────────────
    gaussians._xyz.requires_grad_(True)
    gaussians._scaling.requires_grad_(True)
    gaussians._rotation.requires_grad_(True)
    gaussians._opacity.requires_grad_(False)
    gaussians._features_dc.requires_grad_(True)
    gaussians._features_rest.requires_grad_(True)

    optimizer = torch.optim.Adam([
        {"params": gaussians._features_dc},
        {"params": gaussians._features_rest},
        {"params": gaussians._scaling},
        {"params": gaussians._rotation},
        {"params": gaussians._xyz}
    ], lr=LR, eps=1e-15)

    # ── Precompute reference renders ──────────────────────────────────────────
    train_cameras = scene.getTrainCameras()
    ref_cache = ReferenceRenderCache(gaussians, pp, background)
    ref_cache.precompute(train_cameras)

    # ── VGG + style grams ─────────────────────────────────────────────────────
    vgg         = VGGFeatures().to(device)
    style_img   = T.ToTensor()(Image.open(args.style_image).convert("RGB")).to(device)
    style_grams = precompute_style_grams(style_img, vgg)

    # ── Training loop ─────────────────────────────────────────────────────────
    sampler = TrainCameraSampler(train_cameras)
    out_dir = f"{args.model_path}_styled"
    num = 0
    while os.path.exists(f"{out_dir}{num}"):
        num += 1
    out_dir = f"{out_dir}{num}"
    os.makedirs(out_dir, exist_ok=True)

    copy_meta(scene.model_path, out_dir)

    l1_losses = []
    style_losses = []
    total_losses = []

    pbar = tqdm(range(1, ITERATIONS + 1), desc="Style transfer")
    for step in pbar:
        optimizer.zero_grad()
        total_l1    = torch.tensor(0.0, device=device)
        total_style = torch.tensor(0.0, device=device)
        total_loss = torch.tensor(0.0, device=device)

        for cam in sampler.sample(VIEWS_PER_STEP):
            rendered = render(cam, gaussians, pp, background)["render"].clamp(0, 1)
            ref      = ref_cache[cam].to(device)

            Ll1    = l1_loss(rendered, ref)
            r_vgg  = normalize_for_vgg(F.interpolate(rendered.unsqueeze(0), size=VGG_MAX_DIM))
            Lstyle = compute_style_loss(vgg(r_vgg), style_grams)

            total_l1    += Ll1.detach()
            total_style += Lstyle.detach()
            total_loss = total_loss + Ll1 + LAMBDA_STYLE * Lstyle

        (total_loss / VIEWS_PER_STEP).backward()
        optimizer.step()

        l1_losses.append((total_l1 / VIEWS_PER_STEP).item())
        style_losses.append((total_style / VIEWS_PER_STEP).item())
        total_losses.append((total_loss.detach() / VIEWS_PER_STEP).item())

        pbar.set_postfix({
            "L1": f"{(total_l1 / VIEWS_PER_STEP).item():.4f}",
            "style": f"{(total_style / VIEWS_PER_STEP).item():.6f}",
            "total": f"{(total_loss.detach() / VIEWS_PER_STEP).item():.4f}",
        })

        if step % SAVE_EVERY == 0 or step == ITERATIONS:
            scene.save_retrained(out_dir, step)

    save_loss_curves(l1_losses, style_losses, total_losses)
    print(f"Done. Stylised scene saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("-m", "--model_path",  required=True)
    parser.add_argument("--style_image",       required=True)
    args = parser.parse_args()
    train_style(args)