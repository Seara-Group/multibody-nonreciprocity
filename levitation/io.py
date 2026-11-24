"""Loading the experimental data.

Raft A is read from the trajectory cache DATA/_traj_cache_all.npz (466 particles x 62 999
frames; displacements about each particle's mean position, in microns).  The cache holds
exactly the positions of the raw tracker file (checked 2026-11-23: mean positions identical,
no gaps); displacements are stored in single precision (~1e-6 um).  If the cache is missing it
is rebuilt from the raw file into OUT/derived/.
"""
import os
from dataclasses import dataclass

import numpy as np

from . import config


@dataclass
class Trajectories:
    dx: np.ndarray          # (N, T, 2) displacement about the mean position, um (float32)
    mean_pos: np.ndarray    # (N, 2) time-averaged position, um
    dt: float               # s

    @property
    def n_particles(self):
        return self.dx.shape[0]

    @property
    def n_frames(self):
        return self.dx.shape[1]

    def positions(self, dtype=np.float64):
        """(N, T, 2) absolute positions, um."""
        return self.mean_pos[:, None, :].astype(dtype) + self.dx.astype(dtype)


def raw_file(tag="A"):
    """Path of a raft's trajectory file (pickled DataFrame), see config.raw_trajectory."""
    return config.raw_trajectory(tag)


def _build_cache_from_raw(path_out):
    """Raw tracker file -> cache (same conversion as the old utils.load_trajectories)."""
    import pandas as pd
    trn = pd.read_pickle(raw_file("A"))
    px_to_um = config.DIAMETER_UM / (2.0 * float(trn["radius"].mean()))
    f0 = trn["frame"].min()
    particles = np.sort(trn["particle"].unique())
    T = int(trn["frame"].max() - f0 + 1)
    idx = {p: i for i, p in enumerate(particles)}
    pos = np.full((len(particles), T, 2), np.nan)
    pi = trn["particle"].map(idx).to_numpy()
    fi = (trn["frame"] - f0).to_numpy()
    pos[pi, fi, 0] = trn["x"].to_numpy() * px_to_um
    pos[pi, fi, 1] = trn["y"].to_numpy() * px_to_um
    if np.isnan(pos).any():
        raise ValueError("raft A has gaps; the analyses assume gap-free trajectories")
    mean_pos = pos.mean(1)
    dx = (pos - mean_pos[:, None, :]).astype(np.float32)
    np.savez(path_out, dx=dx, mean_pos=mean_pos, dt=config.DT)
    return dx, mean_pos


def load_raft_a():
    """Raft A trajectories (cache in DATA, else OUT/derived, else built from the raw file)."""
    for p in (os.path.join(config.DATA, "_traj_cache_all.npz"),
              os.path.join(config.DERIVED, "_traj_cache_all.npz")):
        if os.path.exists(p):
            z = np.load(p)
            return Trajectories(dx=z["dx"], mean_pos=z["mean_pos"], dt=config.DT)
    config.ensure_dirs(config.DERIVED)
    dx, mean_pos = _build_cache_from_raw(os.path.join(config.DERIVED, "_traj_cache_all.npz"))
    return Trajectories(dx=dx, mean_pos=mean_pos, dt=config.DT)
