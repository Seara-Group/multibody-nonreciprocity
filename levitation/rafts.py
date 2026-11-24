"""Rafts A-E read frame by frame from their tracker files, and the lattice spacing of each.

The tracker files are pickled pandas DataFrames (columns x, y, radius, label, frame, and in
some files particle), several GB each.  `load_columns` memory-maps them and returns the
columns as numpy arrays without unpickling, so a raft can be read one frame at a time.

Lattice spacing (as in the published table): the raft-average spacing, the median Delaunay bond
length.  Raft A uses its mean positions (54.01 um).  Rafts B-E rotate during the record, so the
median is taken in 20 frames spread through the record and averaged, after merging detections
closer than 3/4 of the median closest-neighbour distance (the tracker sometimes reports one
particle twice).  This reproduces raft_average_spacing.csv of the published table.
"""
import json
import mmap
import os
import re
import struct

import numpy as np
from scipy.spatial import cKDTree

from . import config

# ---- reading the tracker files ----------------------------------------------------------------
_NOARG = set(b']})(.NRbaesut012') | {0x81, 0x8f, 0x90, 0x91, 0x92, 0x96, 0x97}


def _walk_pickle(mm, sz, on_bytes, on_str=None):
    """Step through a pickle's opcodes without materialising any payload."""
    i = 0
    while i < sz:
        op = mm[i]; o0 = i; i += 1
        if op == 0x80: i += 1
        elif op == 0x95: i += 8
        elif op in (0x94, 0x93, 0x85, 0x86, 0x87, 0x88, 0x89): pass
        elif op == 0x8d: i += 8 + struct.unpack_from('<Q', mm, i)[0]
        elif op == 0x8c:
            n = mm[i]
            if on_str:
                on_str(o0, bytes(mm[i + 1:i + 1 + n]).decode('utf8', 'replace'))
            i += 1 + n
        elif op == 0x58: i += 4 + struct.unpack_from('<I', mm, i)[0]
        elif op == 0x8e:
            n = struct.unpack_from('<Q', mm, i)[0]; on_bytes(i + 8, n); i += 8 + n
        elif op == 0x42:
            n = struct.unpack_from('<I', mm, i)[0]; on_bytes(i + 4, n); i += 4 + n
        elif op == 0x43:
            n = mm[i]; on_bytes(i + 1, n); i += 1 + n
        elif op == 0x4a: i += 4
        elif op == 0x4b: i += 1
        elif op == 0x4d: i += 2
        elif op == 0x47: i += 8
        elif op == 0x8a: i += 1 + mm[i]
        elif op == 0x8b: i += 4 + struct.unpack_from('<I', mm, i)[0]
        elif op == 0x71: i += 1
        elif op == 0x72: i += 4
        elif op == 0x68: i += 1
        elif op == 0x6a: i += 4
        elif op == 0x2e: return i
        elif op in _NOARG: pass
        else:
            raise ValueError(f"unhandled pickle opcode {op:#x} at {o0}")
    return i


def load_columns(fn):
    """Memory-mapped columns of a tracker pickle: ({name: read-only array}, handle).

    Keep `handle` alive while the arrays are in use.
    """
    fh = open(fn, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    ev, st = [], {"dt": None}

    def on_str(o, s):
        if re.fullmatch(r"[fiu][1248]", s):
            st["dt"] = s
        ev.append(("S", o, s))

    def on_bytes(o, n):
        if n > 1000:
            ev.append(("B", o, (n, st["dt"])))

    _walk_pickle(mm, os.path.getsize(fn), on_bytes, on_str)
    i_x = [i for i, (k, o, v) in enumerate(ev) if k == "S" and v == "x"][0]
    names = []
    for k, o, v in ev[i_x:]:            # column names: the run of short strings from 'x'
        if k != "S":
            continue
        if v == "name" or v.startswith("pandas"):
            break
        names.append(v)
    blocks = [(o, v[0], v[1]) for k, o, v in ev[:i_x] if k == "B"]
    total = sum(n // np.dtype(d).itemsize for o, n, d in blocks)
    if total % len(names):
        raise ValueError(f"{total} values do not split into the columns {names}")
    N = total // len(names)
    col, j = {}, 0
    for off, n, d in blocks:
        it = np.dtype(d).itemsize
        for m in range((n // it) // N):
            col[names[j]] = np.frombuffer(mm, dtype=d, count=N, offset=off + m * N * it)
            j += 1
    return col, (fh, mm)


class Raft:
    """One experiment, read frame by frame (pixels).  Stray detections far outside the raft are
    dropped in every frame: points farther from the median centre than 1.2 x the 90th-percentile
    radius."""

    def __init__(self, tag):
        self.tag = tag
        fn, self.D_um, self.label = config.RAFTS[tag]
        path = os.path.join(config.DATA, "expt_trajs", fn)
        if not os.path.exists(path):
            path = os.path.join(config.DATA, fn)
        self.col, self._h = load_columns(path)
        self.x, self.y, fr = self.col["x"], self.col["y"], self.col["frame"]
        self.f0 = int(fr[0])
        self.T = int(fr[-1]) - self.f0 + 1
        self.bounds = np.searchsorted(fr, np.arange(self.f0, self.f0 + self.T + 1))
        P = self.frame(self.T // 2)
        d = np.linalg.norm(P[:, None] - P[None], axis=-1)
        np.fill_diagonal(d, np.inf)
        self.a_closest_px = float(np.median(d.min(1)))      # grid of the cached spectra
        r = np.asarray(self.col["radius"][::997], float)
        self.um_per_px = self.D_um / (2 * r.mean())

    def _raw(self, t):
        s, e = self.bounds[t], self.bounds[t + 1]
        return np.column_stack([np.asarray(self.x[s:e], float), np.asarray(self.y[s:e], float)]), s, e

    def _keep(self, P):
        r = np.linalg.norm(P - np.median(P, 0), axis=1)
        return r < 1.2 * np.percentile(r, 90)

    def frame(self, t):
        P, _, _ = self._raw(t)
        return P[self._keep(P)]

    def radii(self, t):
        P, s, e = self._raw(t)
        return np.asarray(self.col["radius"][s:e], float)[self._keep(P)]


# ---- lattice spacing --------------------------------------------------------------------------
def merge_close(P, frac=0.75):
    """Merge detections closer than `frac` x the median closest-neighbour distance."""
    d, j = cKDTree(P).query(P, k=2)
    a = np.median(d[:, 1])
    keep = np.ones(len(P), bool)
    Q = P.copy()
    for i in np.argsort(d[:, 1]):
        if d[i, 1] < frac * a and keep[i] and keep[j[i, 1]]:
            Q[i] = 0.5 * (P[i] + P[j[i, 1]])
            keep[j[i, 1]] = False
    return Q[keep], int((~keep).sum())


def spacings(n_frames=20, refresh=False):
    """{raft: {"ell_um", "um_per_px", "a_closest_px", "merged_per_frame", "n_particles"}}, ell_um
    the raft-average spacing (median Delaunay bond).

    Cached in OUT/derived/raft_bond_spacing.json (reading the five tracker files takes a minute).
    """
    from . import io, lattice
    path = os.path.join(config.DERIVED, "raft_bond_spacing.json")
    if os.path.exists(path) and not refresh:
        return json.load(open(path))
    out = {}
    for tag in config.RAFTS:
        R = Raft(tag)
        meds, merged, npart = [], [], []
        for t in np.linspace(0, R.T - 1, n_frames).astype(int):
            P = R.frame(t) * R.um_per_px
            Q, m = merge_close(P)
            meds.append(lattice.median_bond(Q))
            merged.append(m)
            npart.append(len(P))
        ell = lattice.median_bond(io.load_raft_a().mean_pos) if tag == "A" else float(np.mean(meds))
        out[tag] = dict(ell_um=ell, um_per_px=R.um_per_px, a_closest_px=R.a_closest_px,
                        merged_per_frame=float(np.mean(merged)), n_particles=int(npart[0]))
        print(f"raft {tag}: spacing {ell:.2f} um ({'mean positions' if tag == 'A' else f'{n_frames} frames'}), "
              f"{np.mean(merged):.1f} merged detections per frame, {npart[0]} particles")
    config.ensure_dirs(config.DERIVED)
    json.dump(out, open(path, "w"), indent=1)
    return out
