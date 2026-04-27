"""
interpolate_scenes.py
---------------------
Interpolate between an original and stylized 3DGS scene.
Geometry is fixed (copied from original), only SH coefficients are blended.

Usage
-----
    python interpolate_scenes.py \
        --original  output/my_scene \
        --stylized  output/my_scene_styled \
        --alphas 0.25 0.5 0.75
"""

import os
import argparse
import numpy as np
from plyfile import PlyData, PlyElement


def load_ply(path):
    return PlyData.read(path)


def get_sh_fields(plydata):
    """Return names of all SH feature fields (f_dc and f_rest)."""
    names = [p.name for p in plydata.elements[0].properties]
    return [n for n in names if n.startswith("f_dc") or n.startswith("f_rest")]


def interpolate_ply(orig_ply, style_ply, alpha):
    """
    alpha=0.0 → original, alpha=1.0 → fully stylized
    Returns a new PlyData with interpolated SH coefficients.
    """
    orig_el  = orig_ply.elements[0]
    style_el = style_ply.elements[0]

    sh_fields = get_sh_fields(orig_ply)

    # Build new vertex data by copying original
    new_data = {name: orig_el[name].copy() for name in orig_el.data.dtype.names}

    # Lerp only SH fields
    for field in sh_fields:
        new_data[field] = (1 - alpha) * orig_el[field] + alpha * style_el[field]

    # Reconstruct vertex element
    new_vertices = np.rec.fromarrays(
        list(new_data.values()),
        dtype=orig_el.data.dtype
    )
    return PlyData([PlyElement.describe(new_vertices, "vertex")],
                   text=orig_ply.text)


def find_ply(model_path):
    """Find the latest point_cloud.ply in a model directory."""
    pc_dir = os.path.join(model_path, "point_cloud")
    iterations = sorted([
        int(d.split("_")[1]) for d in os.listdir(pc_dir)
        if d.startswith("iteration_")
    ])
    latest = iterations[-1]
    return os.path.join(pc_dir, f"iteration_{latest}", "point_cloud.ply"), latest


def main(args):
    orig_ply_path,  orig_iter  = find_ply(args.original)
    style_ply_path, style_iter = find_ply(args.stylized)

    print(f"Original:  {orig_ply_path}")
    print(f"Stylized:  {style_ply_path}")

    orig_ply  = load_ply(orig_ply_path)
    style_ply = load_ply(style_ply_path)

    assert len(orig_ply.elements[0].data) == len(style_ply.elements[0].data), \
        "Scenes have different number of Gaussians — were they trained from the same base?"

    for alpha in args.alphas:
        out_dir = os.path.join(args.output_dir, f"alpha_{alpha:.2f}",
                               f"point_cloud/iteration_{orig_iter}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "point_cloud.ply")

        interp = interpolate_ply(orig_ply, style_ply, alpha)
        interp.write(out_path)
        print(f"alpha={alpha:.2f} → {out_path}")

    # Copy meta files so each interpolated scene is SIBR-viewable
    import shutil
    for fname in ["cameras.json", "cfg_args"]:
        src = os.path.join(args.original, fname)
        if os.path.exists(src):
            for alpha in args.alphas:
                dst_dir = os.path.join(args.output_dir, f"alpha_{alpha:.2f}")
                shutil.copy(src, os.path.join(dst_dir, fname))

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original",   required=True)
    parser.add_argument("--stylized",   required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.25, 0.5, 0.75])
    args = parser.parse_args()
    main(args)