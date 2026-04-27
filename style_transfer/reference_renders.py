"""
reference_renders.py
--------------------
Precompute and cache "frozen" renders of the pretrained 3DGS scene.
These are used as content anchors during style fine-tuning so that
the photometric loss does not fight the style loss by pulling back
toward the original photo colours.

Usage
-----
    from reference_renders import ReferenceRenderCache

    cache = ReferenceRenderCache(gaussians, pipe, background)
    cache.precompute(train_cameras)          # run once before style loop

    # Inside style training loop:
    ref = cache[viewpoint_cam]               # (3, H, W) tensor, detached
"""

import os
from typing import Dict, List, Optional

import torch
from tqdm import tqdm


class ReferenceRenderCache:
    """
    Renders every training camera once with the frozen pretrained Gaussians
    and stores the results in a dict keyed by camera uid.
    """

    def __init__(self, gaussians, pipe, background: torch.Tensor):
        """
        gaussians  : GaussianModel (already trained, will not be modified here)
        pipe       : PipelineParams
        background : torch.Tensor (3,) background colour
        """
        self.gaussians  = gaussians
        self.pipe       = pipe
        self.background = background
        self._cache: Dict[int, torch.Tensor] = {}

    def precompute(self, cameras, verbose: bool = True):
        """
        Render all cameras and cache the results.

        cameras: list of 3DGS Camera objects
        """
        # Import here to avoid circular imports when placed inside 3dgs repo
        from gaussian_renderer import render

        iterator = tqdm(cameras, desc="Precomputing reference renders") if verbose else cameras

        with torch.no_grad():
            for cam in iterator:
                pkg = render(cam, self.gaussians, self.pipe, self.background)
                img = pkg["render"].detach().clamp(0, 1)   # (3, H, W)
                self._cache[cam.uid] = img


    def __getitem__(self, cam) -> torch.Tensor:
        """
        Return the cached reference render for `cam`.
        Raises KeyError if camera was not precomputed.
        """
        if cam.uid not in self._cache:
            raise KeyError(
                f"Camera uid={cam.uid} not found in reference cache. "
                "Did you call precompute() first?"
            )
        return self._cache[cam.uid]

    def __contains__(self, cam) -> bool:
        return cam.uid in self._cache

    def save(self, path: str):
        """Save cache to disk so you don't have to recompute every run."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self._cache, path)
        print(f"Reference cache saved to {path}")

    def load(self, path: str):
        """Load a previously saved cache."""
        self._cache = torch.load(path, map_location="cpu")
        print(f"Reference cache loaded from {path} ({len(self._cache)} views)")