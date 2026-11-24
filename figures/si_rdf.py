"""SI figure -- the lattice spacing of raft A grows toward its edge.

Panels written to OUT/figures/si_rdf/ (one file, three square panels, as published):
    rdf.pdf   a: mean positions coloured by 100-um ring from the centre;
              b: g(r) of each ring, averaged over 1000 frames;
              c: first peak of each ring's g(r) against the ring's mean distance from the
                 centre, with the raft-average spacing (median Delaunay bond, 54.0 um; dashed).

    python si_rdf.py
"""
import os
import sys

import numpy as np
from matplotlib import cm
from matplotlib.patches import Wedge
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from levitation import io, lattice  # noqa: E402
from levitation.plotting import plt, save_panel, save_derived  # noqa: E402

FIGURE = "si_rdf"
STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans", "pdf.fonttype": 42, "axes.linewidth": 1.0,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "xtick.minor.width": 0.8, "ytick.minor.width": 0.8,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "savefig.dpi": 300,
}


def ring_rdf(tr, dR=100.0, rmax=200.0, dr=2.0, nt=1000):
    """g(r) around the particles of each ring (rings by mean distance from the centre).

    The last ring runs out to the outermost particle so every particle is in a ring.
    Pairs are counted from each ring's particles to all particles, frame by frame.
    """
    R = np.linalg.norm(tr.mean_pos, axis=1)
    R_edges = np.append(np.arange(0, R.max() - dR / 2, dR), R.max() + 1e-6)
    ring = np.digitize(R, R_edges) - 1
    n_ring = len(R_edges) - 1
    masks = [ring == k for k in range(n_ring)]
    M = np.array([m.sum() for m in masks])
    bins = np.arange(0, rmax + dr, dr)
    counts = np.zeros((n_ring, len(bins) - 1))
    nt = min(nt, tr.n_frames)
    for t in range(nt):
        xy = (tr.mean_pos + tr.dx[:, t]).astype(np.float32)
        Dm = cdist(xy, xy)
        np.fill_diagonal(Dm, np.inf)
        for k, m in enumerate(masks):
            d = Dm[m].ravel()
            counts[k] += np.histogram(d[d < rmax], bins)[0]
    r = 0.5 * (bins[:-1] + bins[1:])
    shell = np.pi * (bins[1:] ** 2 - bins[:-1] ** 2)
    rho = tr.n_particles / 1500**2          # fixed area that encloses the raft
    gr = np.array([counts[k] / (M[k] * rho * shell * nt) if M[k] else np.full(len(r), np.nan)
                   for k in range(n_ring)])
    return r, gr, R_edges, ring, M


def first_peak(r, g, search=(30, 90), half=2):
    """Sub-bin position of the first g(r) peak (parabola through the tallest bin +- half)."""
    win = (r >= search[0]) & (r <= search[1]) & ~np.isnan(g)
    if not win.any():
        return np.nan
    i = np.nanargmax(np.where(win, g, -np.inf))
    lo, hi = max(i - half, 0), min(i + half + 1, len(r))
    good = ~np.isnan(g[lo:hi])
    if good.sum() < 3:
        return r[i]
    a, b, _ = np.polyfit(r[lo:hi][good], g[lo:hi][good], 2)
    return r[i] if a == 0 else -b / (2 * a)


def main():
    tr = io.load_raft_a()
    r, gr, R_edges, ring, M = ring_rdf(tr)
    r_peak = np.array([first_peak(r, gr[k]) if M[k] else np.nan for k in range(len(M))])
    ell = lattice.median_bond(tr.mean_pos)          # raft-average spacing, as published
    R = np.linalg.norm(tr.mean_pos, axis=1)
    R_mean = np.array([R[ring == k].mean() if M[k] else np.nan for k in range(len(M))])
    cols = [cm.viridis(x) for x in np.linspace(0.05, 0.85, len(M))]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.35))
        for k in range(len(M)):
            m = ring == k
            ax[0].add_patch(Wedge((0, 0), R_edges[k + 1], 0, 360, width=R_edges[k + 1] - R_edges[k],
                                  color=cols[k], alpha=0.15, lw=0, zorder=0))
            ax[0].plot(tr.mean_pos[m, 0], tr.mean_pos[m, 1], ".", ms=2.5, color=cols[k])
            if M[k]:
                ax[1].plot(r, gr[k], color=cols[k], lw=1.2)
                ax[2].plot(R_mean[k], r_peak[k], "o", ms=4.5, color=cols[k], zorder=3)
        lim = R_edges[-1] * 1.03
        ax[0].set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel=r"$x$ ($\mu$m)", ylabel=r"$y$ ($\mu$m)")
        ax[1].set(xlim=(0, 200), ylim=(0, None), xlabel=r"$r$ ($\mu$m)", ylabel=r"$g(r)$")
        ax[2].axhline(ell, color="0.35", lw=1.0, ls=(0, (4, 2)), zorder=1)
        ax[2].text(0.97, ell, f"raft average {ell:.1f} ", transform=ax[2].get_yaxis_transform(),
                   ha="right", va="bottom", color="0.35", fontsize=7)
        ax[2].set(xlim=(0, R_edges[-1]), xlabel=r"distance from center ($\mu$m)",
                  ylabel=r"first peak of $g(r)$ ($\mu$m)")
        for a_ in ax:
            a_.set_box_aspect(1)
        fig.tight_layout(w_pad=1.2)
        save_panel(fig, FIGURE, "rdf")

    print(f"raft-average spacing (median Delaunay bond) = {ell:.2f} um")
    for k in range(len(M)):
        print(f"  ring {R_edges[k]:5.0f}-{R_edges[k + 1]:5.0f} um: {M[k]:3d} particles, "
              f"mean radius {R_mean[k]:6.1f} um, first peak {r_peak[k]:.2f} um")
    save_derived(FIGURE, dict(r=r, gr=gr, R_edges=R_edges, M=M, r_peak=r_peak, R_mean=R_mean, ell=ell))


if __name__ == "__main__":
    main()
