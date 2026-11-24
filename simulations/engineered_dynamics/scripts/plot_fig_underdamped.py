#!/usr/bin/env python3
"""
plot_fig_underdamped.py -- the six panels of the engineered-activity figure,
every one from integrated UNDERDAMPED runs (zeta = 0.01, no prestress).

    python plot_fig_underdamped.py [--zeta 0.01] [--runs ../data/runs] [--out ../figs]

Layout follows the paper figure (overleaf/figures/engineered_activity.pdf):
  a  tilt wall of 3 dislocations, with particle trails (alpha1 = +0.05)
  b  climb displacement dy(t), walls n = 1..4, alpha1 = -0.05 (dashed), +0.05 (solid)
  c  climb velocity v_y against n, for alpha1 = -0.05, 0, +0.05
  d  ring of six dislocations, with particle trails (alpha1 = +0.1)
  e  orientation theta(t), one curve per alpha1 in -0.1 .. 0.1
  f  angular velocity Omega against alpha1

Units: lattice units a = 1, k_a = 1, m = 1; time in sqrt(m/k_a); gamma = 2 zeta.
Rates are the run's own fit over its last 60% (run.py), nothing extrapolated.

Needs the runs of configs/grid_fig.csv (jobs/submit_fig.sh) plus grid.csv rows
16, 17, 19 (rotor 0.1, 0.033, 0) and 91 (wall4 0).  Missing runs are skipped and
listed, so it can be run on a partial set.

Writes OUT/fig_engineered_activity_ud.{pdf,png} (all six panels) and the two
trail maps alone as OUT/panel_a_trails_ud.pdf, OUT/panel_d_trails_ud.pdf.
"""
import argparse, os
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import EllipseCollection, LineCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import Delaunay

plt.rcParams["pdf.fonttype"] = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ROTOR_ALPHAS = [-0.1, -0.066, -0.033, 0.0, 0.033, 0.066, 0.1]
WALL_NS = [1, 2, 3, 4]
WALL_ALPHA = 0.05

try:                                           # same colour maps as the paper figure
    import cmasher as cmr
    WALL_COLS = cmr.take_cmap_colors("cmr.eclipse", N=5, cmap_range=(0.1, 0.9))[1:]
    ROT_COLS = cmr.take_cmap_colors("cmr.iceburn", N=7, cmap_range=(0.1, 0.9))
except ImportError:
    WALL_COLS = [plt.cm.cividis(x) for x in np.linspace(0.1, 0.9, 5)][1:]
    ROT_COLS = [plt.cm.coolwarm(x) for x in np.linspace(0.05, 0.95, 7)]

# ---------------------------------------------------------------- trail maps --
C_FIVE, C_SEVEN, R_POLY, DOT = "magenta", "green", 0.62, "0.75"
D_PART = 41.1 / 55.4
TRAIL_CMAP = LinearSegmentedColormap.from_list(
    "trail", ["#dfe7f1", "#7d97b8", "#1f3a5f", "#0b1626"])   # light (early) -> dark (now)


def coordination_defects(pos, rc=1.45):
    sm = Delaunay(pos).simplices
    e = np.stack([np.linalg.norm(pos[sm[:, u]] - pos[sm[:, v]], axis=1)
                  for u, v in ((0, 1), (1, 2), (2, 0))], 1)
    tri = sm[(e < rc).all(1)]
    ed = np.concatenate([sm[:, [0, 1]], sm[:, [1, 2]], sm[:, [2, 0]]])
    ed = np.unique(np.sort(ed[np.concatenate([e[:, 0], e[:, 1], e[:, 2]]) < rc], 1), axis=0)
    coord = np.bincount(ed.ravel(), minlength=len(pos))
    ntri = np.bincount(tri.ravel(), minlength=len(pos))
    closed = (ntri == coord) & (coord > 0)
    return closed & (coord == 5), closed & (coord == 7)


def spread_subset(P, dmin):
    order = np.argsort(np.linalg.norm(P - P.mean(0), axis=1))
    keep = []
    for i in order:
        if all(np.hypot(*(P[i] - P[j])) >= dmin for j in keep):
            keep.append(i)
    return np.array(keep)


def trail_map(ax, R, k_end, lw=0.9, n_seg=80, spacing=3.5):
    idx = np.unique(np.linspace(0, k_end, n_seg + 1).round().astype(int))
    path = R[idx][:, spread_subset(R[k_end], spacing)]
    segs = np.stack([path[:-1], path[1:]], 2).reshape(-1, 2, 2)
    cols = TRAIL_CMAP(np.repeat(np.linspace(0, 1, len(idx) - 1), path.shape[1]))
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw,
                                     capstyle="round", zorder=4))
    P = R[k_end]
    f5, f7 = coordination_defects(P)
    for mask, nv, col in ((f5, 5, C_FIVE), (f7, 7, C_SEVEN)):
        for p in P[mask]:
            ax.add_patch(RegularPolygon(tuple(p), numVertices=nv, radius=R_POLY,
                                        facecolor=col, edgecolor="none", zorder=1.6))
    ax.add_collection(EllipseCollection(widths=D_PART, heights=D_PART, angles=0.0,
                                        units="xy", offsets=P, offset_transform=ax.transData,
                                        facecolors=DOT, edgecolors="none", zorder=3))
    allp = np.concatenate([P, R[0]])
    pad = 0.5 * D_PART + 0.5
    ax.set_xlim(allp[:, 0].min() - pad, allp[:, 0].max() + pad)
    ax.set_ylim(allp[:, 1].min() - pad, allp[:, 1].max() + pad)
    ax.set_aspect("equal"); ax.set_axis_off()


# ------------------------------------------------------------------- loading --
def name(design, alpha, zeta):
    return "%s_none_z%g_a%g" % (design, zeta, alpha)


def load(runs, nm, missing):
    d = os.path.join(runs, nm)
    f, p = os.path.join(d, nm + ".npz"), os.path.join(d, nm + "_params.yaml")
    if not (os.path.exists(f) and os.path.exists(p)):
        missing.append(nm)
        return None
    z = np.load(f)
    r = yaml.safe_load(open(p))["result"]
    return dict(z=z, r=r, t=z["log_t"], theta=np.unwrap(z["log_theta"]),
                com=np.c_[z["log_com0"], z["log_com1"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zeta", type=float, default=0.01)
    ap.add_argument("--runs", default=os.path.join(ROOT, "data", "runs"))
    ap.add_argument("--out", default=os.path.join(ROOT, "figs"))
    ap.add_argument("--swim_dist", type=float, default=4.0, help="panel a trail length [a]")
    ap.add_argument("--rot_deg", type=float, default=30.0, help="panel d trail rotation [deg]")
    ap.add_argument("--map_wall", type=int, default=3, help="wall shown in panel a")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    miss = []

    walls = {(n, s): load(a.runs, name("wall%d" % n, s * WALL_ALPHA, a.zeta), miss)
             for n in WALL_NS for s in (-1, 0, 1)}
    rotor = {al: load(a.runs, name("rotor", al, a.zeta), miss) for al in ROTOR_ALPHAS}

    fig, ax = plt.subplots(2, 3, figsize=(9, 6))
    tunit = r"$\sqrt{m/k_a}$"

    # a ---------------------------------------------------------------------
    w = walls.get((a.map_wall, 1))
    if w is not None:
        R = w["z"]["trajectory_R"].astype(float)
        c = R.mean(1); dd = np.linalg.norm(c - c[0], axis=1)
        k = int(np.argmin(np.abs(dd - a.swim_dist)))
        trail_map(ax[0, 0], R, k)
        print("a: wall%d, trail t = 0 .. %.4g (%.0f bond periods), %.2f a"
              % (a.map_wall, w["z"]["times"][k], w["z"]["times"][k] / (2 * np.pi), dd[k]))
    else:
        ax[0, 0].set_axis_off()

    # b ---------------------------------------------------------------------
    tend = [walls[k]["t"][-1] for k in walls if walls[k] is not None and k[1] != 0]
    twin = min(tend) if tend else None
    for (n, s), w in walls.items():
        if w is None:
            continue
        m = w["t"] <= (twin if twin else w["t"][-1])
        col = "k" if s == 0 else WALL_COLS[n - 1]
        ls = "-" if s >= 0 else "--"
        ax[0, 1].plot(w["t"][m], w["com"][m, 1] - w["com"][0, 1], ls, color=col, lw=1.2)
    ax[0, 1].axhline(0, color="0.8", zorder=-1)
    ax[0, 1].set(xlabel="time  [%s]" % tunit, ylabel=r"$\Delta y \quad [\ell]$")

    # c ---------------------------------------------------------------------
    for s, ls, mk in ((1, "-", "o"), (-1, "--", "o"), (0, "-.", "s")):
        ns, vs = [], []
        for n in WALL_NS:
            w = walls[(n, s)]
            if w is None:
                continue
            v = w["r"]["v"][1]
            ns.append(n); vs.append(v)
            ax[0, 2].plot(n, v * 1e2, mk, mfc="k" if s == 0 else WALL_COLS[n - 1],
                          mec="w", mew=1.5, ms=8, zorder=3)
        if ns:
            ax[0, 2].plot(ns, np.array(vs) * 1e2, ls, color="0.3", zorder=1)
    ax[0, 2].set(xlabel="# dislocations", xticks=WALL_NS, xlim=(0.5, 4.5),
                 ylabel=r"$v_y \quad [\ell/\sqrt{m/k_a}] \times 10^{-2}$")

    # d ---------------------------------------------------------------------
    w = rotor.get(0.1)
    if w is not None:
        R = w["z"]["trajectory_R"].astype(float)
        tf = w["z"]["times"]
        t_end = np.interp(np.radians(a.rot_deg), np.abs(w["theta"]), w["t"])
        k = int(np.argmin(np.abs(tf - t_end)))
        trail_map(ax[1, 0], R, k)
        print("d: rotor, trail t = 0 .. %.4g (%.0f bond periods)" % (tf[k], tf[k] / (2 * np.pi)))
    else:
        ax[1, 0].set_axis_off()

    # e ---------------------------------------------------------------------
    tend = [w["t"][-1] for al, w in rotor.items() if w is not None and al != 0]
    twin = min(tend) if tend else None
    for al, col in zip(ROTOR_ALPHAS, ROT_COLS):
        w = rotor[al]
        if w is None:
            continue
        m = w["t"] <= (twin if twin else w["t"][-1])
        ax[1, 1].plot(w["t"][m], w["theta"][m], color=col, lw=1.2)
    ax[1, 1].axhline(0, color="0.8", zorder=-1)
    ax[1, 1].set(
        xlabel="time  [%s]" % tunit,
        ylabel=r"$\theta \quad [\mathrm{rad}]$",
        ylim=[-7, 7],
        yticks=[-2*np.pi, -np.pi, 0, np.pi, 2*np.pi],
        yticklabels=[r"$-2\pi$", r"$-\pi$", "0", r"$\pi$", r"$2\pi$"]
    )

    # f ---------------------------------------------------------------------
    al_ok, om = [], []
    for al, col in zip(ROTOR_ALPHAS, ROT_COLS):
        w = rotor[al]
        if w is None:
            continue
        al_ok.append(al); om.append(w["r"]["Omega"])
        ax[1, 2].plot(al, w["r"]["Omega"] * 1e5, "o", mfc=col, mec="w", mew=1.5, ms=8, zorder=3)
    if al_ok:
        ax[1, 2].plot(al_ok, np.array(om) * 1e5, "-", color="0.3", zorder=1)
    ax[1, 2].axhline(0, color="0.8", zorder=-1); ax[1, 2].axvline(0, color="0.8", zorder=-1)
    ax[1, 2].set(
        xlabel=r"$\alpha_3 / k_a$",
        ylabel=r"$\Omega \quad [\mathrm{rad}/\sqrt{m/k_a}] \times 10^{-5}$",
        ylim=[-2*np.pi, 2*np.pi],
        yticks=[-2*np.pi, -np.pi, 0, np.pi, 2*np.pi],
        yticklabels=[r"$-2\pi$", r"$-\pi$", 0, r"$\pi$", r"$2\pi$"]
        )

    for x, lab in zip(ax.ravel(), "abcdef"):
        x.set_box_aspect(1)
        x.text(-0.12, 1.05, lab, transform=x.transAxes, fontsize=12, fontweight="bold")
    fig.tight_layout()
    base = os.path.join(a.out, "fig_engineered_activity_ud")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    for x, lab in ((ax[0, 0], "a"), (ax[1, 0], "d")):     # trail maps alone, for Illustrator
        ext = x.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(a.out, "panel_%s_trails_ud.pdf" % lab), bbox_inches=ext)
    plt.close(fig)
    print("wrote", base + ".pdf")
    if miss:
        print("missing runs (%d), skipped:" % len(miss))
        for m_ in miss:
            print("   ", m_)


if __name__ == "__main__":
    main()
