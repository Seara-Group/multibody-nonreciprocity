"""Lattice geometry of a raft: neighbours, topological defects, hexatic order, spacing.

Conventions (decided 2026-11-23):
  * neighbours and defects come from the plain Delaunay triangulation of the positions,
    with no distance cutoff;
  * the lattice spacing is the median Delaunay bond of the positions (`median_bond`;
    54.01 um for raft A's mean positions), the "raft average" of the paper.  Fig 4b's wavevector
    axis and the SI stripes' spectra use the first peak of g(r) instead (`lattice_spacing`,
    52.48 um), as published.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, cKDTree


@dataclass
class Lattice:
    pos: np.ndarray                 # (N, 2)
    neighbors: list                 # neighbors[i] = sorted int array
    edges: np.ndarray               # (E, 2) i < j, sorted
    triangles: np.ndarray           # (F, 3)
    coordination: np.ndarray        # (N,)
    is_boundary: np.ndarray         # (N,) convex-hull particles
    fives: np.ndarray = field(default=None)     # indices of 5-coordinated particles
    sevens: np.ndarray = field(default=None)    # indices of 7-coordinated particles
    dislocations: list = field(default=None)    # [(i5, i7)] adjacent 5-7 pairs
    disclinations: np.ndarray = field(default=None)   # 5-fold sites left unpaired

    @property
    def interior(self):
        return ~self.is_boundary

    @property
    def dislocation_positions(self):
        if not self.dislocations:
            return np.empty((0, 2))
        return np.array([(self.pos[a] + self.pos[b]) / 2 for a, b in self.dislocations])

    def summary(self):
        n5b = int(self.is_boundary[self.fives].sum())
        ndb = int(self.is_boundary[self.disclinations].sum())
        return (f"{len(self.pos)} particles, {len(self.edges)} Delaunay bonds, "
                f"{int(self.is_boundary.sum())} on the convex hull; "
                f"5-fold {len(self.fives)} ({n5b} on hull), 7-fold {len(self.sevens)}; "
                f"{len(self.dislocations)} 5-7 dislocations, {len(self.disclinations)} "
                f"unpaired 5-fold disclinations ({ndb} on hull)")


def delaunay_lattice(pos):
    """Neighbours, coordination and 5/7 defects from the Delaunay triangulation.

    Dislocations: each 5-fold site is paired with an adjacent, still-unpaired 7-fold site,
    interior 5-fold sites first, then those on the convex hull, each in index order.
    """
    pos = np.asarray(pos, float)
    N = len(pos)
    tri = Delaunay(pos)
    nb = [set() for _ in range(N)]
    for s in tri.simplices:
        for a in s:
            for b in s:
                if a != b:
                    nb[a].add(int(b))
    neighbors = [np.array(sorted(n), dtype=int) for n in nb]
    edges = np.array(sorted({(min(a, b), max(a, b)) for a in range(N) for b in nb[a]}))
    coordination = np.array([len(n) for n in nb])
    is_boundary = np.zeros(N, bool)
    is_boundary[ConvexHull(pos).vertices] = True

    fives = np.where(coordination == 5)[0]
    sevens = np.where(coordination == 7)[0]
    seven_set = set(sevens.tolist())
    used, pairs = set(), []
    for group in (fives[~is_boundary[fives]], fives[is_boundary[fives]]):
        for i5 in group:
            for i7 in neighbors[i5]:
                if i7 in seven_set and i5 not in used and i7 not in used:
                    pairs.append((int(i5), int(i7)))
                    used.update((int(i5), int(i7)))
                    break
    disclinations = np.array([i for i in fives if i not in used], dtype=int)
    return Lattice(pos=pos, neighbors=neighbors, edges=edges, triangles=tri.simplices.copy(),
                   coordination=coordination, is_boundary=is_boundary, fives=fives,
                   sevens=sevens, dislocations=pairs, disclinations=disclinations)


def psi6(pos, neighbors):
    """Per-particle hexatic order psi6_i = <exp(6 i theta_ij)> over the Delaunay neighbours."""
    pos = np.asarray(pos, float)
    out = np.full(len(pos), np.nan, dtype=complex)
    for i, nbrs in enumerate(neighbors):
        if len(nbrs):
            d = pos[nbrs] - pos[i]
            out[i] = np.mean(np.exp(6j * np.arctan2(d[:, 1], d[:, 0])))
    return out


def median_bond(pos):
    """Median Delaunay bond length, um (54.01 um for raft A's mean positions)."""
    pos = np.asarray(pos, float)
    e = set()
    for sm in Delaunay(pos).simplices:
        for i in range(3):
            e.add(tuple(sorted((sm[i], sm[(i + 1) % 3]))))
    e = np.array(sorted(e))
    return float(np.median(np.linalg.norm(pos[e[:, 0]] - pos[e[:, 1]], axis=1)))


def lattice_spacing(mean_pos, r_cut=120.0, nbin=400):
    """First peak of g(r) of the mean positions, um (52.48 um for raft A)."""
    bins = np.linspace(1e-6, r_cut, nbin)
    xc = 0.5 * (bins[1:] + bins[:-1])
    pairs = cKDTree(mean_pos).query_pairs(r=r_cut, output_type="ndarray")
    dd = np.linalg.norm(mean_pos[pairs[:, 0]] - mean_pos[pairs[:, 1]], axis=1)
    g = np.histogram(dd, bins=bins)[0] / (2 * np.pi * xc * (bins[1] - bins[0]))
    return float(xc[np.argmax(g)])


def closed_shell(pos, ell, rc_factor=1.45):
    """True where a particle's nearest-neighbour shell closes, i.e. it is not on the free edge.

    Nearest-neighbour bonds are the Delaunay edges shorter than rc_factor * ell; a particle's
    shell closes when it touches as many nearest-neighbour triangles as it has bonds.  On the
    free edge the fan of triangles is open (one short), and so are the particles in the dents
    of a ragged rim that the convex hull misses.  Used to keep 5/7 defects to the bulk.
    """
    pos = np.asarray(pos, float)
    rc = rc_factor * ell
    sm = Delaunay(pos).simplices
    side = lambda u, v: np.linalg.norm(pos[sm[:, u]] - pos[sm[:, v]], axis=1)
    short = np.stack([side(0, 1), side(1, 2), side(2, 0)], 1) < rc
    tris = sm[short.all(1)]
    e = np.concatenate([sm[:, [0, 1]][short[:, 0]], sm[:, [1, 2]][short[:, 1]], sm[:, [2, 0]][short[:, 2]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    nbond = np.bincount(e.ravel(), minlength=len(pos))
    ntri = np.bincount(tris.ravel(), minlength=len(pos))
    return (ntri == nbond) & (nbond > 0)
