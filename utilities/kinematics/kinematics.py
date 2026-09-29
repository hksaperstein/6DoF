"""Forward kinematics for the custom 6-DoF arm.

Pure numpy. No file I/O, no plotting, no globals.

Frame indexing, which is the single most likely source of a silent error:
`frames[0]` is the base and `frames[i+1]` is the frame after applying DH row
`i`. Joint `i`'s axis is the z axis of the frame PRECEDING it, `frames[i]`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import ArmConfig


def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Classical DH: Rot(z,theta) Trans(z,d) Trans(x,a) Rot(x,alpha)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])


@dataclass(frozen=True)
class Pose:
    frames: np.ndarray  # (7, 4, 4): base, then one per DH row

    @property
    def joint_origins(self) -> np.ndarray:
        """(7, 3) — base origin, each joint origin, then the TCP."""
        return self.frames[:, :3, 3]

    @property
    def joint_axes(self) -> np.ndarray:
        """(6, 3) unit vectors. Joint i's axis is z of the frame before it."""
        return self.frames[:6, :3, 2]

    @property
    def tcp(self) -> np.ndarray:
        return self.frames[6]

    @property
    def tcp_position(self) -> np.ndarray:
        return self.frames[6][:3, 3]

    def joint_axis(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """(point on the axis, unit direction) for joint i."""
        return self.frames[i][:3, 3], self.frames[i][:3, 2]


def forward_kinematics(cfg: ArmConfig, q) -> Pose:
    q = np.asarray(q, dtype=float)
    if q.shape != (len(cfg.rows),):
        raise ValueError(f"q must have {len(cfg.rows)} elements, got {q.shape}")

    frames = np.empty((len(cfg.rows) + 1, 4, 4))
    frames[0] = np.eye(4)
    T = np.eye(4)
    for i, row in enumerate(cfg.rows):
        T = T @ dh_transform(q[i], row.d, row.a, row.alpha)
        frames[i + 1] = T
    return Pose(frames=frames)


def axis_distance(p1, d1, p2, d2) -> float:
    """Minimum distance between two infinite lines, parallel case handled."""
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    d1 = np.asarray(d1, float) / np.linalg.norm(d1)
    d2 = np.asarray(d2, float) / np.linalg.norm(d2)
    n = np.cross(d1, d2)
    norm_n = np.linalg.norm(n)
    if norm_n < 1e-9:  # parallel
        w = p2 - p1
        return float(np.linalg.norm(w - np.dot(w, d1) * d1))
    return float(abs(np.dot(p2 - p1, n)) / norm_n)
