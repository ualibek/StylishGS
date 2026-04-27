"""
stylize_images.py
"""

import os
import argparse
from PIL import Image

import torch
import torchvision.transforms as T
import torchvision.utils as vutils
from tqdm import tqdm
import torch.nn.functional as F

from style_loss import (VGGFeatures, normalize_for_vgg,
                        precompute_style_grams, compute_style_loss)

# ── Hardcoded hyperparameters ─────────────────────────────────────────────────
ITERATIONS   = 100       # LBFGS steps (equivalent to ~1000 Adam steps)
MAX_SIZE     = 512       # resize longest edge to this before stylizing
LBFGS_MAX_ITER  = 20    # line search steps per LBFGS iteration
LBFGS_HISTORY   = 5     # gradient history size (keep low to avoid OOM)
LAMBDA_STYLE = 2
# ─────────────────────────────────────────────────────────────────────────────


def load_image(path, device, max_size=MAX_SIZE):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max_size / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return T.ToTensor()(img).unsqueeze(0).to(device)


def stylize_image(content_img, style_grams, vgg, device):
    canvas = content_img.clone().detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [canvas], lr=1.0,
        max_iter=LBFGS_MAX_ITER,
        history_size=LBFGS_HISTORY
    )

    # Precompute content target once
    with torch.no_grad():
        content_feat = vgg(normalize_for_vgg(content_img))[3].detach()  # relu4_4

    pbar = tqdm(range(ITERATIONS), desc="stylizing", position=1, leave=False)
    for _ in pbar:
        def closure():
            optimizer.zero_grad()
            feats    = vgg(normalize_for_vgg(canvas))
            Lstyle   = compute_style_loss(feats, style_grams)
            Lcontent = F.mse_loss(feats[3], content_feat)
            loss     = Lcontent + LAMBDA_STYLE * Lstyle
            loss.backward()
            with torch.no_grad():
                canvas.data.clamp_(0, 1)
            return loss

        loss = optimizer.step(closure)
        pbar.set_postfix({"loss": f"{loss.item():.4e}"})

    return canvas.detach()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    vgg = VGGFeatures().to(device).eval()

    style_img   = load_image(args.style_image, device)
    style_grams = precompute_style_grams(style_img, vgg)

    image_files = sorted([
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print(f"Stylizing {len(image_files)} images...")
    os.makedirs(args.output_dir, exist_ok=True)

    for fname in tqdm(image_files, desc="Images", position=0):
        out_path = os.path.join(args.output_dir, fname)
        if os.path.exists(out_path):
            tqdm.write(f"Skipping {fname} (already exists)")
            continue

        content_img = load_image(os.path.join(args.input_dir, fname), device)
        stylized    = stylize_image(content_img, style_grams, vgg, device)
        vutils.save_image(stylized, out_path)

    print(f"Done. Stylized images saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",   required=True)
    parser.add_argument("--style_image", required=True)
    parser.add_argument("--output_dir",  required=True)
    args = parser.parse_args()
    main(args)