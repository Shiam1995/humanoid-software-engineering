"""
extract_map.py
Converts a Gaussian Splat .ply point cloud into a 2D occupancy grid
for robot navigation.
"""

import numpy as np
from pathlib import Path
import json
import sys

try:
    from plyfile import PlyData
except ImportError:
    raise ImportError("Run: pip install plyfile")

import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────────────────
GRID_RESOLUTION = 0.05  # metres per cell
HEIGHT_MIN      = -1.0  # ignore points below this
HEIGHT_MAX      = 3.0   # ignore points above this
OBSTACLE_THRESH = 2     # min points in a cell to mark as obstacle
# ─────────────────────────────────────────────────────────────────────────────


def load_ply(ply_path: str) -> np.ndarray:
    print(f"Loading {ply_path}...")
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    xyz = np.stack([np.array(v["x"]), np.array(v["y"]), np.array(v["z"])], axis=1)
    print(f"  Loaded {len(xyz):,} points")
    return xyz


def build_occupancy_grid(xyz: np.ndarray, resolution: float = GRID_RESOLUTION):
    mask = (xyz[:, 1] > HEIGHT_MIN) & (xyz[:, 1] < HEIGHT_MAX)
    xyz_f = xyz[mask]
    print(f"  After height filter: {len(xyz_f):,} points")

    x = xyz_f[:, 0]
    z = xyz_f[:, 2]

    x_min, x_max = x.min(), x.max()
    z_min, z_max = z.min(), z.max()

    nx = int((x_max - x_min) / resolution) + 1
    nz = int((z_max - z_min) / resolution) + 1
    print(f"  Grid size: {nx} x {nz} cells")

    grid = np.zeros((nz, nx), dtype=np.int32)
    xi = np.clip(((x - x_min) / resolution).astype(int), 0, nx - 1)
    zi = np.clip(((z - z_min) / resolution).astype(int), 0, nz - 1)
    for xi_, zi_ in zip(xi, zi):
        grid[zi_, xi_] += 1

    occupancy = (grid >= OBSTACLE_THRESH).astype(np.uint8)
    meta = {
        "x_min": float(x_min), "x_max": float(x_max),
        "z_min": float(z_min), "z_max": float(z_max),
        "resolution": resolution,
        "nx": nx, "nz": nz
    }
    return occupancy, meta


def save_map(occupancy, meta, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "occupancy.npy", occupancy)
    with open(output_dir / "map_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    plt.figure(figsize=(12, 10))
    plt.imshow(occupancy, cmap="gray_r", origin="lower")
    plt.title("Occupancy Grid from Gaussian Splat")
    plt.xlabel("X (cells)")
    plt.ylabel("Z (cells)")
    plt.savefig(output_dir / "occupancy_map.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved map to {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_map.py <point_cloud.ply> <output_dir>")
        sys.exit(1)
    xyz = load_ply(sys.argv[1])
    occupancy, meta = build_occupancy_grid(xyz)
    save_map(occupancy, meta, sys.argv[2])
    print(f"  Free space: {(1 - occupancy.mean()) * 100:.1f}%")
    print("Done!")
