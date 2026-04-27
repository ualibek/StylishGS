"""
camera_sampler.py
-----------------
Utilities for sampling camera viewpoints uniformly over the scene.

The standard 3DGS repo stores training cameras as a list of Camera objects
(scene.getTrainCameras()). This module provides:

  - uniform random sampling from that list
  - a spherical sampler that generates novel viewpoints on a sphere
    around the scene centre (useful if you want cameras beyond the
    training set)
"""

import math
import random
from typing import List

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Simple uniform sampler over existing training cameras
# ---------------------------------------------------------------------------

class TrainCameraSampler:
    """
    Wraps the list returned by scene.getTrainCameras() and samples
    uniformly at random, with optional shuffling per epoch.
    """

    def __init__(self, cameras, shuffle: bool = True, seed: int = 42):
        self.cameras = list(cameras)
        self.shuffle  = shuffle
        self._rng     = random.Random(seed)
        self._queue: list = []

    def _refill(self):
        self._queue = list(self.cameras)
        if self.shuffle:
            self._rng.shuffle(self._queue)

    def next(self):
        """Return the next camera, refilling the queue when exhausted."""
        if not self._queue:
            self._refill()
        return self._queue.pop()

    def sample(self, n: int = 1):
        """Return a list of n cameras sampled without replacement per epoch."""
        return [self.next() for _ in range(n)]

    def __len__(self):
        return len(self.cameras)


# ---------------------------------------------------------------------------
# Spherical sampler for novel viewpoints
# ---------------------------------------------------------------------------

def _look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Build a 4x4 camera-to-world matrix (R | t) from eye/center/up.
    """
    z = eye - center
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)

    R = np.stack([x, y, z], axis=1)          # (3, 3)
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3,  3] = eye
    return c2w


def sample_spherical_cameras(
    n: int,
    radius: float,
    center: np.ndarray = None,
    elevation_range: tuple = (-30, 60),
    seed: int = 0,
) -> List[np.ndarray]:
    """
    Sample n camera positions uniformly on a sphere around `center`.

    Returns a list of n (4, 4) camera-to-world matrices.
    These are NOT 3DGS Camera objects — use them as reference poses or
    convert them with `c2w_to_3dgs_camera()` below.

    Parameters
    ----------
    n               : number of cameras
    radius          : sphere radius (in scene units)
    center          : (3,) scene centre; defaults to origin
    elevation_range : (min_deg, max_deg) elevation angle range
    seed            : RNG seed for reproducibility
    """
    if center is None:
        center = np.zeros(3)

    rng = np.random.default_rng(seed)
    up  = np.array([0.0, 1.0, 0.0])

    poses = []
    for _ in range(n):
        # uniform azimuth
        azimuth   = rng.uniform(0, 2 * math.pi)
        # uniform elevation in degrees → radians
        elev_deg  = rng.uniform(*elevation_range)
        elevation = math.radians(elev_deg)

        eye = center + radius * np.array([
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
            math.cos(elevation) * math.cos(azimuth),
        ])

        c2w = _look_at(eye, center, up)
        poses.append(c2w)

    return poses


def estimate_scene_center_and_radius(cameras) -> tuple:
    """
    Estimate scene bounding sphere from training camera positions.

    cameras: list of 3DGS Camera objects (must have .camera_center attribute)
    Returns: (center np.ndarray (3,), radius float)
    """
    positions = np.stack(
        [cam.camera_center.cpu().numpy() for cam in cameras], axis=0
    )  # (N, 3)
    center = positions.mean(axis=0)
    radius = np.linalg.norm(positions - center, axis=1).max()
    return center, float(radius)