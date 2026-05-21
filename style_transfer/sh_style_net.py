"""
sh_style_net.py
---------------
PointNet++-inspired network that stylizes Spherical Harmonic coefficients
of a 3DGS scene given paired (original, stylized) training data.

Architecture:
  - Encoder: 3 levels of Set Abstraction (SA) — hierarchical local grouping
  - Decoder: 3 levels of Feature Propagation (FP) — interpolation-based upsampling
  - Head: MLP per point predicting delta SH coefficients

The network predicts a RESIDUAL (delta) on top of the original SH,
which makes training easier since the network only needs to learn
what changes, not reconstruct the full SH from scratch.

Usage
-----
    # Training
    python sh_style_net.py --mode train \
        --data_dir  pairs/ \
        --output    checkpoints/

    # Inference on a new scene
    python sh_style_net.py --mode infer \
        --model     checkpoints/best.pt \
        --scene     output/new_scene \
        --out       output/new_scene_styled
"""

import os
import sys
import glob
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm


# ── Hardcoded hyperparameters ─────────────────────────────────────────────────
SH_DEGREE    = 3
BATCH_SIZE   = 1          # one scene per batch (scenes differ in N)
LR           = 1e-4
EPOCHS       = 200
N_SUBSAMPLE  = 50_000     # subsample this many Gaussians per scene for training
HIDDEN_DIM   = 256
# ─────────────────────────────────────────────────────────────────────────────

SH_COEFFS = (SH_DEGREE + 1) ** 2 * 3   # total SH floats per Gaussian
DC_DIM    = 3                            # just the DC term
REST_DIM  = SH_COEFFS - DC_DIM


# ─────────────────────────────────────────────────────────────────────────────
# PointNet++ building blocks
# ─────────────────────────────────────────────────────────────────────────────

def square_distance(src, dst):
    """
    Pairwise squared distances between two point sets.
    src: (B, N, C)
    dst: (B, M, C)
    Returns: (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.bmm(src, dst.permute(0, 2, 1))   # (B, N, M)
    dist += (src ** 2).sum(-1, keepdim=True)             # (B, N, 1)
    dist += (dst ** 2).sum(-1).unsqueeze(1)              # (B, 1, M)
    return dist.clamp(min=0)


def farthest_point_sample(xyz, n_samples):
    """
    Farthest Point Sampling — pick n_samples points maximally spread out.
    xyz: (B, N, 3)
    Returns: (B, n_samples) indices
    """
    B, N, _ = xyz.shape
    device   = xyz.device
    idx      = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    dist     = torch.full((B, N), float("inf"), device=device)
    farthest = torch.randint(0, N, (B,), device=device)

    for i in range(n_samples):
        idx[:, i] = farthest
        centroid  = xyz[torch.arange(B), farthest].unsqueeze(1)  # (B,1,3)
        d         = ((xyz - centroid) ** 2).sum(-1)               # (B,N)
        dist      = torch.min(dist, d)
        farthest  = dist.argmax(dim=1)

    return idx


def knn_query(k, xyz, query):
    """
    Find k nearest neighbors in xyz for each point in query.
    xyz:   (B, N, 3)
    query: (B, S, 3)
    Returns: (B, S, k) indices into xyz
    """
    dist = square_distance(query, xyz)       # (B, S, N)
    return dist.topk(k, dim=-1, largest=False).indices


def index_points(points, idx):
    """
    Gather points by index.
    points: (B, N, C)
    idx:    (B, S) or (B, S, k)
    Returns: same shape as idx but with C appended
    """
    B = points.shape[0]
    device = points.device
    if idx.dim() == 2:
        return points[torch.arange(B, device=device).unsqueeze(1), idx]
    elif idx.dim() == 3:
        B, S, k = idx.shape
        idx_flat = idx.reshape(B, -1)
        gathered = points[torch.arange(B, device=device).unsqueeze(1), idx_flat]
        return gathered.reshape(B, S, k, -1)


class SharedMLP(nn.Module):
    """1x1 conv stack acting as per-point MLP."""
    def __init__(self, dims, bn=True):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Conv1d(dims[i], dims[i+1], 1))
            if bn:
                layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SetAbstraction(nn.Module):
    """
    PointNet++ Set Abstraction layer.
    Groups local neighborhoods around sampled centroids,
    extracts features with a shared MLP, then max-pools.
    """
    def __init__(self, n_centroids, k, in_dim, mlp_dims):
        """
        n_centroids: number of output points (FPS sampling)
        k:           number of neighbors per centroid
        in_dim:      input feature dimension (excluding xyz)
        mlp_dims:    list of MLP layer dims
        """
        super().__init__()
        self.n_centroids = n_centroids
        self.k           = k
        self.mlp         = SharedMLP([in_dim + 3] + mlp_dims)

    def forward(self, xyz, features):
        """
        xyz:      (B, N, 3)
        features: (B, N, C) or None
        Returns:
            new_xyz:      (B, n_centroids, 3)
            new_features: (B, n_centroids, mlp_dims[-1])
        """
        B, N, _ = xyz.shape

        # Sample centroids via FPS
        idx      = farthest_point_sample(xyz, self.n_centroids)  # (B, S)
        new_xyz  = index_points(xyz, idx)                         # (B, S, 3)

        # Group k neighbors around each centroid
        nn_idx   = knn_query(self.k, xyz, new_xyz)                # (B, S, k)
        grouped  = index_points(xyz, nn_idx)                      # (B, S, k, 3)

        # Relative positions
        grouped  = grouped - new_xyz.unsqueeze(2)                 # (B, S, k, 3)

        if features is not None:
            grouped_feat = index_points(features, nn_idx)         # (B, S, k, C)
            grouped      = torch.cat([grouped, grouped_feat], -1) # (B, S, k, 3+C)

        # MLP + max pool
        B, S, k, D = grouped.shape
        x = grouped.permute(0, 1, 3, 2).reshape(B*S, D, k)       # (B*S, D, k)
        x = self.mlp(x)                                           # (B*S, D', k)
        x = x.max(dim=-1).values                                  # (B*S, D')
        new_features = x.reshape(B, S, -1)                        # (B, S, D')

        return new_xyz, new_features


class FeaturePropagation(nn.Module):
    """
    PointNet++ Feature Propagation layer.
    Upsamples from S points back to N points via inverse-distance interpolation,
    then refines with a shared MLP.
    """
    def __init__(self, in_dim, skip_dim, mlp_dims):
        super().__init__()
        self.mlp = SharedMLP([in_dim + skip_dim] + mlp_dims)

    def forward(self, xyz1, xyz2, feat1, feat2):
        """
        xyz1:  (B, N, 3) — points to upsample TO (more points)
        xyz2:  (B, S, 3) — points to upsample FROM (fewer points)
        feat1: (B, N, C1) — skip connection features at xyz1
        feat2: (B, S, C2) — features at xyz2 to be upsampled
        """
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            # Edge case: broadcast single point
            interp = feat2.expand(B, N, -1)
        else:
            # Inverse distance weighted interpolation from xyz2 to xyz1
            dist  = square_distance(xyz1, xyz2)        # (B, N, S)
            dist, idx = dist.topk(3, dim=-1, largest=False)
            dist  = dist.clamp(min=1e-10)
            w     = 1.0 / dist                         # (B, N, 3)
            w     = w / w.sum(dim=-1, keepdim=True)
            interp_feat = index_points(feat2, idx)     # (B, N, 3, C2)
            interp = (interp_feat * w.unsqueeze(-1)).sum(dim=2)  # (B, N, C2)

        # Concatenate with skip features
        if feat1 is not None:
            x = torch.cat([interp, feat1], dim=-1)    # (B, N, C1+C2)
        else:
            x = interp

        x = x.permute(0, 2, 1)                        # (B, C, N)
        x = self.mlp(x)                               # (B, D', N)
        return x.permute(0, 2, 1)                     # (B, N, D')


# ─────────────────────────────────────────────────────────────────────────────
# Full network
# ─────────────────────────────────────────────────────────────────────────────

class SHStyleNet(nn.Module):
    """
    PointNet++-style encoder-decoder for SH coefficient stylization.

    Input:  xyz (N,3) + SH features (N, SH_COEFFS)
    Output: delta SH coefficients (N, SH_COEFFS)
            final SH = original + delta
    """
    def __init__(self, in_sh_dim=SH_COEFFS, out_sh_dim=SH_COEFFS):
        super().__init__()
        H = HIDDEN_DIM

        # Encoder — 3 SA levels, progressively fewer points
        self.sa1 = SetAbstraction(n_centroids=2048, k=16,
                                  in_dim=in_sh_dim,
                                  mlp_dims=[64, 64, 128])
        self.sa2 = SetAbstraction(n_centroids=512,  k=16,
                                  in_dim=128,
                                  mlp_dims=[128, 128, 256])
        self.sa3 = SetAbstraction(n_centroids=128,  k=16,
                                  in_dim=256,
                                  mlp_dims=[256, 256, 512])

        # Decoder — 3 FP levels, upsample back to N
        self.fp3 = FeaturePropagation(in_dim=512, skip_dim=256,
                                      mlp_dims=[256, 256])
        self.fp2 = FeaturePropagation(in_dim=256, skip_dim=128,
                                      mlp_dims=[128, 128])
        self.fp1 = FeaturePropagation(in_dim=128, skip_dim=in_sh_dim,
                                      mlp_dims=[128, 128])

        # Per-point prediction head
        self.head = nn.Sequential(
            nn.Conv1d(128, H, 1),
            nn.BatchNorm1d(H),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(H, out_sh_dim, 1),
        )

    def forward(self, xyz, sh):
        """
        xyz: (B, N, 3)
        sh:  (B, N, SH_COEFFS)
        Returns: (B, N, SH_COEFFS) delta SH
        """
        # Encoder
        xyz1, f1 = self.sa1(xyz, sh)
        xyz2, f2 = self.sa2(xyz1, f1)
        xyz3, f3 = self.sa3(xyz2, f2)

        # Decoder — propagate features back up
        f2 = self.fp3(xyz2, xyz3, f2, f3)
        f1 = self.fp2(xyz1, xyz2, f1, f2)
        f0 = self.fp1(xyz,  xyz1, sh, f1)

        # Per-point head
        out = self.head(f0.permute(0, 2, 1))   # (B, out_sh_dim, N)
        return out.permute(0, 2, 1)            # (B, N, out_sh_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ScenePairDataset(Dataset):
    """
    Loads paired (original, stylized) Gaussian scenes.

    Expected directory structure:
        data_dir/
            scene_01/
                original/point_cloud/iteration_N/point_cloud.ply
                styled/point_cloud/iteration_N/point_cloud.ply
            scene_02/
                ...
    """
    def __init__(self, data_dir, n_subsample=N_SUBSAMPLE):
        self.pairs      = []
        self.n_subsample = n_subsample
        self._find_pairs(data_dir)
        print(f"Found {len(self.pairs)} scene pairs.")

    def _find_pairs(self, data_dir):
        for scene_dir in sorted(Path(data_dir).iterdir()):
            if not scene_dir.is_dir():
                continue
            orig_plys   = sorted(glob.glob(str(scene_dir / "original/point_cloud/iteration_*/point_cloud.ply")))
            styled_plys = sorted(glob.glob(str(scene_dir / "styled/point_cloud/iteration_*/point_cloud.ply")))
            if orig_plys and styled_plys:
                self.pairs.append((orig_plys[-1], styled_plys[-1]))

    def _load_ply(self, path):
        from plyfile import PlyData
        ply  = PlyData.read(path)
        el   = ply.elements[0]

        xyz  = np.stack([el["x"], el["y"], el["z"]], axis=-1).astype(np.float32)

        # DC SH: f_dc_0, f_dc_1, f_dc_2
        dc   = np.stack([el["f_dc_0"], el["f_dc_1"], el["f_dc_2"]], axis=-1).astype(np.float32)

        # Higher-order SH: f_rest_0 ... f_rest_N
        rest_names = sorted([p.name for p in el.properties if p.name.startswith("f_rest_")],
                            key=lambda n: int(n.split("_")[-1]))
        rest = np.stack([el[n] for n in rest_names], axis=-1).astype(np.float32)

        sh = np.concatenate([dc, rest], axis=-1)   # (N, SH_COEFFS)
        return xyz, sh

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        orig_path, styled_path = self.pairs[idx]
        xyz_o, sh_o = self._load_ply(orig_path)
        xyz_s, sh_s = self._load_ply(styled_path)

        N = min(len(xyz_o), len(xyz_s))

        # Random subsample for training
        if N > self.n_subsample:
            sel    = np.random.choice(N, self.n_subsample, replace=False)
            xyz_o  = xyz_o[sel]
            sh_o   = sh_o[sel]
            sh_s   = sh_s[sel]

        # Normalize xyz to unit cube for stable training
        xyz_min = xyz_o.min(0)
        xyz_max = xyz_o.max(0)
        xyz_o   = (xyz_o - xyz_min) / (xyz_max - xyz_min + 1e-8)

        return {
            "xyz":    torch.tensor(xyz_o, dtype=torch.float32),
            "sh_in":  torch.tensor(sh_o,  dtype=torch.float32),
            "sh_tgt": torch.tensor(sh_s,  dtype=torch.float32),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    dataset = ScenePairDataset(args.data_dir)
    loader  = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    model   = SHStyleNet().to(device)
    optim   = torch.optim.Adam(model.parameters(), lr=LR)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)

    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            xyz    = batch["xyz"].to(device)     # (1, N, 3)
            sh_in  = batch["sh_in"].to(device)   # (1, N, SH_COEFFS)
            sh_tgt = batch["sh_tgt"].to(device)  # (1, N, SH_COEFFS)

            optim.zero_grad()

            delta  = model(xyz, sh_in)            # (1, N, SH_COEFFS)
            sh_pred = sh_in + delta               # residual prediction

            # L1 + L2 loss — L1 for sharper colors, L2 for stability
            loss = F.l1_loss(sh_pred, sh_tgt) + 0.5 * F.mse_loss(sh_pred, sh_tgt)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            epoch_loss += loss.item()

        sched.step()
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch}: loss={avg_loss:.6f}  lr={sched.get_last_lr()[0]:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.output, "best.pt"))
            print(f"  ↳ saved best model (loss={best_loss:.6f})")

    print("Training done.")


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def infer(args):
    import shutil
    from plyfile import PlyData, PlyElement

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SHStyleNet().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"Loaded model from {args.model}")

    # Find PLY
    plys = sorted(glob.glob(os.path.join(args.scene,
                  "point_cloud/iteration_*/point_cloud.ply")))
    assert plys, f"No point_cloud.ply found in {args.scene}"
    ply_path  = plys[-1]
    iteration = ply_path.split("iteration_")[1].split("/")[0]

    from plyfile import PlyData
    ply = PlyData.read(ply_path)
    el  = ply.elements[0]

    xyz_np = np.stack([el["x"], el["y"], el["z"]], axis=-1).astype(np.float32)
    dc_np  = np.stack([el["f_dc_0"], el["f_dc_1"], el["f_dc_2"]], axis=-1).astype(np.float32)
    rest_names = sorted([p.name for p in el.properties if p.name.startswith("f_rest_")],
                        key=lambda n: int(n.split("_")[-1]))
    rest_np = np.stack([el[n] for n in rest_names], axis=-1).astype(np.float32)
    sh_np   = np.concatenate([dc_np, rest_np], axis=-1)

    N = xyz_np.shape[0]
    print(f"Loaded {N:,} Gaussians.")

    # Normalize xyz
    xyz_min = xyz_np.min(0)
    xyz_max = xyz_np.max(0)
    xyz_norm = (xyz_np - xyz_min) / (xyz_max - xyz_min + 1e-8)

    # Run in chunks to avoid OOM on large scenes
    CHUNK = 50_000
    all_deltas = []

    with torch.no_grad():
        for i in tqdm(range(0, N, CHUNK), desc="Stylizing"):
            xyz_chunk = torch.tensor(xyz_norm[i:i+CHUNK]).unsqueeze(0).to(device)
            sh_chunk  = torch.tensor(sh_np[i:i+CHUNK]).unsqueeze(0).to(device)
            delta     = model(xyz_chunk, sh_chunk)
            all_deltas.append(delta.squeeze(0).cpu().numpy())

    delta_np = np.concatenate(all_deltas, axis=0)  # (N, SH_COEFFS)
    sh_styled = sh_np + delta_np

    # Write new PLY — copy all original fields, replace SH
    new_data = {p.name: el[p.name].copy() for p in el.properties}
    new_data["f_dc_0"] = sh_styled[:, 0]
    new_data["f_dc_1"] = sh_styled[:, 1]
    new_data["f_dc_2"] = sh_styled[:, 2]
    for j, name in enumerate(rest_names):
        new_data[name] = sh_styled[:, 3 + j]

    new_verts = np.rec.fromarrays(list(new_data.values()), dtype=el.data.dtype)
    out_ply   = PlyData([PlyElement.describe(new_verts, "vertex")], text=ply.text)

    out_dir = os.path.join(args.out, f"point_cloud/iteration_{iteration}")
    os.makedirs(out_dir, exist_ok=True)
    out_ply.write(os.path.join(out_dir, "point_cloud.ply"))

    for fname in ["cameras.json", "cfg_args"]:
        src = os.path.join(args.scene, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, fname))

    print(f"Saved styled scene to {args.out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",     choices=["train", "infer"], required=True)

    # Train args
    parser.add_argument("--data_dir", help="Directory of scene pairs")
    parser.add_argument("--output",   help="Where to save checkpoints")

    # Infer args
    parser.add_argument("--model",    help="Path to checkpoint .pt file")
    parser.add_argument("--scene",    help="Path to unseen scene model_path")
    parser.add_argument("--out",      help="Output path for styled scene")

    args = parser.parse_args()

    if args.mode == "train":
        assert args.data_dir and args.output, "--data_dir and --output required for training"
        train(args)
    else:
        assert args.model and args.scene and args.out, "--model, --scene, --out required for inference"
        infer(args)