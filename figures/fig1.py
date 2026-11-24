"""Fig 1 -- acoustic levitation creates a crystalline solid (raft A).

Panels written to OUT/figures/fig1/:
    c_abs_psi6.pdf   magnitude of the hexatic order parameter psi6 of each particle
    c_arg_psi6.pdf   its argument (lattice orientation), arg(psi6)/6 in degrees
    d_rmsd.pdf       each particle drawn at its mean position with radius = its RMSD
    e_msd.pdf        MSD / d^2 against lag time, mean +- std over particles
(a is a drawing and b a camera frame.)

    python fig1.py [--panel c d e]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from levitation import config, io, lattice  # noqa: E402
from levitation.plotting import plt, save_panel, save_derived  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
import cmasher as cmr  # noqa: E402

FIGURE = "fig1"
D = config.DIAMETER_UM


def panel_c(tr):
    lat = lattice.delaunay_lattice(tr.mean_pos)
    p6 = lattice.psi6(tr.mean_pos, lat.neighbors)
    a6 = np.abs(p6)
    inn = lat.interior
    print("raft A lattice:", lat.summary())
    print(f"psi6: <|psi6_i|> interior {a6[inn].mean():.3f}; |<psi6_i>| interior "
          f"{np.abs(p6[inn].mean()):.3f}, all {np.abs(p6.mean()):.3f}; "
          f"fraction |psi6_i| > 0.8 (interior) {(a6[inn] > 0.8).mean():.3f}")

    x, y = tr.mean_pos.T
    for name, c, cmap, lim, ticks, title in (
            ("c_abs_psi6", a6, cmr.lavender, (0, 1), [0, 0.5, 1], r"$|\psi_6|$"),
            ("c_arg_psi6", np.degrees(np.angle(p6)) / 6.0 % 60.0, cmr.seasons_s, (0, 60),
             [0, 20, 40, 60], r"$\arg(\psi_6)/6$ (deg)")):
        fig, ax = plt.subplots(figsize=(2.7, 3))
        sc = ax.scatter(x, y, c=c, cmap=cmap, vmin=lim[0], vmax=lim[1], s=14, edgecolors="k")
        ax.set_aspect("equal")
        ax.set_title(title)
        cax = ax.inset_axes([0.1, -0.1, 0.8, 0.05])
        fig.colorbar(sc, ax=ax, cax=cax, orientation="horizontal", ticks=ticks)
        ax.set_axis_off()
        save_panel(fig, FIGURE, name)
    return dict(psi6=p6, coordination=lat.coordination, is_boundary=lat.is_boundary)


def panel_d(tr):
    """Radius of each circle = the particle's RMSD about its mean position."""
    rmsd = np.sqrt((tr.dx.astype(np.float64) ** 2).sum(2).mean(1))
    print(f"RMSD: median {np.median(rmsd):.2f} um, max {rmsd.max():.2f} um "
          f"(particle radius {D / 2:.2f} um)")
    fig, ax = plt.subplots(dpi=144, figsize=(3, 3))
    circles = [plt.Circle(p, radius=r) for p, r in zip(tr.mean_pos, rmsd)]
    ax.add_collection(PatchCollection(circles, facecolor="C0", alpha=0.7, edgecolors="k"))
    ax.add_patch(plt.Circle([-600, 600], radius=D / 2, facecolor="C0", edgecolor="k",
                            alpha=0.7, fill=True))
    ax.text(-550, 600, r"$d$", fontsize=10, va="center", ha="left")
    ax.set_aspect("equal")
    ax.set(xlim=(-650, 650), xticks=[-500, 0, 500], ylim=(-650, 650), yticks=[-500, 0, 500])
    save_panel(fig, FIGURE, "d_rmsd")
    return dict(rmsd=rmsd)


def msd(tr, n_lags=200, chunk=32):
    """Per-particle MSD at ~log-spaced integer lags (frames), um^2."""
    T = tr.n_frames
    lags = np.unique(np.logspace(0, np.log10(T), num=n_lags).astype(int))[:-1]
    out = np.zeros((tr.n_particles, len(lags)))
    for i0 in range(0, tr.n_particles, chunk):
        r = tr.dx[i0:i0 + chunk].astype(np.float64)
        for j, lag in enumerate(lags):
            out[i0:i0 + chunk, j] = ((r[:, lag:] - r[:, :-lag]) ** 2).sum(2).mean(1)
    return lags, out


def panel_e(tr):
    lags, m = msd(tr)
    t = lags * tr.dt
    mean, sd = m.mean(0), m.std(0)
    fig, ax = plt.subplots(dpi=144, figsize=(3, 3))
    ax.loglog(t[1:], mean[1:] / D**2, color="k", lw=2)
    ax.fill_between(t[1:], (mean - sd)[1:] / D**2, (mean + sd)[1:] / D**2, color="gray", alpha=0.5)
    ax.set(xlabel=r"$\Delta t$ (s)", ylabel=r"MSD / $d^2$", xticks=[1e-3, 1e-1, 1e1])
    save_panel(fig, FIGURE, "e_msd")
    return dict(lags=lags, dt=tr.dt, msd=m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", nargs="+", default=["c", "d", "e"])
    a = ap.parse_args()
    tr = io.load_raft_a()
    data = dict(mean_pos=tr.mean_pos)
    for p in a.panel:
        data.update({"c": panel_c, "d": panel_d, "e": panel_e}[p](tr))
    save_derived(FIGURE, data)


if __name__ == "__main__":
    main()
