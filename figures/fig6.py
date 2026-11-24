"""Fig 6 -- engineered activity: rafts built with dislocations swim and spin.

Every panel is from integrated underdamped runs of simulations/engineered_dynamics (damping ratio
zeta = 0.01, no pre-stress, the minimal model with every coefficient zero except k_a and alpha3).
Lattice units ell = k_a = m = 1, time in sqrt(m/k_a).  Runs are read from
config.ENGINEERED_RUNS (environment variable LEVITATION_ENGINEERED_RUNS).

Panels written to OUT/figures/fig6/:
    a_wall_trails.pdf    tilt wall of 3 dislocations, particle trails while the raft travels 4 ell
                         (alpha3/k_a = +0.05)
    b_climb.pdf          climb displacement dy(t), walls of 1-4 dislocations, alpha3/k_a = -0.05
                         (dashed), +0.05 (solid), 0 (black)
    c_climb_velocity.pdf climb velocity v_y against the number of dislocations
    d_ring_trails.pdf    ring of six dislocations, trails over 30 degrees of rotation (alpha3/k_a = 0.1)
    e_rotation.pdf       orientation theta(t), alpha3/k_a from -0.1 to 0.1
    f_rotation_rate.pdf  angular velocity Omega against alpha3/k_a

Rates (v_y, Omega) are each run's own fit over its last 60% (run.py), as stored in its params
file.  Defects (magenta pentagons 5-fold, green heptagons 7-fold) are Delaunay coordinations where
the nearest-neighbour shell closes (lattice.closed_shell with ell = 1), as in Fig 5.

    python fig6.py [--panel a b c d e f]
"""
import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from levitation import config, lattice  # noqa: E402
from levitation.plotting import plt, save_derived, save_panel  # noqa: E402
from matplotlib.collections import EllipseCollection, LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import RegularPolygon  # noqa: E402
import cmasher as cmr  # noqa: E402

FIGURE = "fig6"
ZETA = 0.01
ROTOR_ALPHAS = [-0.1, -0.066, -0.033, 0.0, 0.033, 0.066, 0.1]
WALL_NS = [1, 2, 3, 4]
WALL_ALPHA = 0.05
MAP_WALL = 3                  # wall shown in panel a
SWIM_DIST = 4.0               # panel a trail length, ell
ROT_DEG = 30.0                # panel d trail rotation, degrees
WALL_COLS = cmr.take_cmap_colors("cmr.eclipse", N=5, cmap_range=(0.1, 0.9))[1:]
ROT_COLS = cmr.take_cmap_colors("cmr.iceburn", N=7, cmap_range=(0.1, 0.9))
C_FIVE, C_SEVEN, R_POLY, DOT = "magenta", "green", 0.62, "0.75"
D_PART = 41.1 / 55.4          # particle diameter drawn, ell (as published)
TRAIL_CMAP = LinearSegmentedColormap.from_list("trail", ["#dfe7f1", "#7d97b8", "#1f3a5f", "#0b1626"])
TUNIT = r"$\sqrt{m/k_a}$"


# ------------------------------------------------------------------ runs
def run_name(design, alpha):
    return "%s_none_z%g_a%g" % (design, ZETA, alpha)


def load(nm, missing):
    d = os.path.join(config.ENGINEERED_RUNS, nm)
    f, p = os.path.join(d, nm + ".npz"), os.path.join(d, nm + "_params.yaml")
    if not (os.path.exists(f) and os.path.exists(p)):
        missing.append(nm)
        return None
    z = np.load(f)
    r = yaml.safe_load(open(p))["result"]
    return dict(z=z, r=r, t=z["log_t"], theta=np.unwrap(z["log_theta"]), com=np.c_[z["log_com0"], z["log_com1"]])


def load_all():
    miss = []
    walls = {(n, s): load(run_name("wall%d" % n, s * WALL_ALPHA), miss) for n in WALL_NS for s in (-1, 0, 1)}
    rotor = {al: load(run_name("rotor", al), miss) for al in ROTOR_ALPHAS}
    if miss:
        print(f"missing runs in {config.ENGINEERED_RUNS} ({len(miss)}), skipped: " + ", ".join(miss))
    return walls, rotor


# ------------------------------------------------------------------ trail maps
def _spread_subset(P, dmin):
    keep = []
    for i in np.argsort(np.linalg.norm(P - P.mean(0), axis=1)):
        if all(np.hypot(*(P[i] - P[j])) >= dmin for j in keep):
            keep.append(i)
    return np.array(keep)


def trail_map(ax, R, k_end, lw=0.9, n_seg=80, spacing=3.5):
    idx = np.unique(np.linspace(0, k_end, n_seg + 1).round().astype(int))
    path = R[idx][:, _spread_subset(R[k_end], spacing)]
    segs = np.stack([path[:-1], path[1:]], 2).reshape(-1, 2, 2)
    cols = TRAIL_CMAP(np.repeat(np.linspace(0, 1, len(idx) - 1), path.shape[1]))
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw, capstyle="round", zorder=4))
    P = R[k_end]
    closed = lattice.closed_shell(P, 1.0)
    co = lattice.delaunay_lattice(P).coordination
    for mask, nv, col in ((closed & (co == 5), 5, C_FIVE), (closed & (co == 7), 7, C_SEVEN)):
        for p in P[mask]:
            ax.add_patch(RegularPolygon(tuple(p), numVertices=nv, radius=R_POLY, facecolor=col,
                                        edgecolor="none", zorder=1.6))
    ax.add_collection(EllipseCollection(widths=D_PART, heights=D_PART, angles=0.0, units="xy", offsets=P,
                                        offset_transform=ax.transData, facecolors=DOT, edgecolors="none", zorder=3))
    allp = np.concatenate([P, R[0]])
    pad = 0.5 * D_PART + 0.5
    ax.set_xlim(allp[:, 0].min() - pad, allp[:, 0].max() + pad)
    ax.set_ylim(allp[:, 1].min() - pad, allp[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return int((closed & (co == 5)).sum()), int((closed & (co == 7)).sum())


def _panel(name, draw, square=True):
    fig, ax = plt.subplots(figsize=(3, 3))
    draw(ax)
    if square:                     # trail maps keep their own equal aspect instead
        ax.set_box_aspect(1)
    save_panel(fig, FIGURE, name)


# ------------------------------------------------------------------ panels
def panel_a(walls):
    w = walls[(MAP_WALL, 1)]
    R = w["z"]["trajectory_R"].astype(float)
    dd = np.linalg.norm(R.mean(1) - R.mean(1)[0], axis=1)
    k = int(np.argmin(np.abs(dd - SWIM_DIST)))
    out = {}
    _panel("a_wall_trails", lambda ax: out.update(defects=trail_map(ax, R, k)), square=False)
    print(f"a: wall of {MAP_WALL}, trail t = 0 .. {w['z']['times'][k]:.4g} ({dd[k]:.2f} ell); "
          f"defects {out['defects'][0]} five-fold, {out['defects'][1]} seven-fold")


def panel_b(walls):
    tend = [w["t"][-1] for k, w in walls.items() if w is not None and k[1] != 0]
    twin = min(tend) if tend else None

    def draw(ax):
        for (n, s), w in walls.items():
            if w is None:
                continue
            m = w["t"] <= (twin if twin else w["t"][-1])
            ax.plot(w["t"][m], w["com"][m, 1] - w["com"][0, 1], "-" if s >= 0 else "--",
                    color="k" if s == 0 else WALL_COLS[n - 1], lw=1.2)
        ax.axhline(0, color="0.8", zorder=-1)
        ax.set(xlabel="time  [%s]" % TUNIT, ylabel=r"$\Delta y \quad [\ell]$")
    _panel("b_climb", draw)


def panel_c(walls):
    def draw(ax):
        for s, ls, mk in ((1, "-", "o"), (-1, "--", "o"), (0, "-.", "s")):
            ns, vs = [], []
            for n in WALL_NS:
                w = walls[(n, s)]
                if w is None:
                    continue
                ns.append(n)
                vs.append(w["r"]["v"][1])
                ax.plot(n, vs[-1] * 1e2, mk, mfc="k" if s == 0 else WALL_COLS[n - 1], mec="w", mew=1.5, ms=8, zorder=3)
            if ns:
                ax.plot(ns, np.array(vs) * 1e2, ls, color="0.3", zorder=1)
        ax.set(xlabel="# dislocations", xticks=WALL_NS, xlim=(0.5, 4.5),
               ylabel=r"$v_y \quad [\ell/\sqrt{m/k_a}] \times 10^{-2}$")
    _panel("c_climb_velocity", draw)


def panel_d(rotor):
    w = rotor[0.1]
    R = w["z"]["trajectory_R"].astype(float)
    tf = w["z"]["times"]
    k = int(np.argmin(np.abs(tf - np.interp(np.radians(ROT_DEG), np.abs(w["theta"]), w["t"]))))
    out = {}
    _panel("d_ring_trails", lambda ax: out.update(defects=trail_map(ax, R, k)), square=False)
    print(f"d: ring, trail t = 0 .. {tf[k]:.4g}; defects {out['defects'][0]} five-fold, {out['defects'][1]} seven-fold")


def panel_e(rotor):
    tend = [w["t"][-1] for al, w in rotor.items() if w is not None and al != 0]
    twin = min(tend) if tend else None

    def draw(ax):
        for al, col in zip(ROTOR_ALPHAS, ROT_COLS):
            w = rotor[al]
            if w is None:
                continue
            m = w["t"] <= (twin if twin else w["t"][-1])
            ax.plot(w["t"][m], w["theta"][m], color=col, lw=1.2)
        ax.axhline(0, color="0.8", zorder=-1)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
        ax.set(xlabel="time  [%s]" % TUNIT, ylabel=r"$\theta \quad [\mathrm{rad}]$", ylim=[-7, 7],
               yticks=[-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi],
               yticklabels=[r"$-2\pi$", r"$-\pi$", "0", r"$\pi$", r"$2\pi$"])
    _panel("e_rotation", draw)


def panel_f(rotor):
    def draw(ax):
        al_ok, om = [], []
        for al, col in zip(ROTOR_ALPHAS, ROT_COLS):
            w = rotor[al]
            if w is None:
                continue
            al_ok.append(al)
            om.append(w["r"]["Omega"])
            ax.plot(al, om[-1] * 1e5, "o", mfc=col, mec="w", mew=1.5, ms=8, zorder=3)
        if al_ok:
            ax.plot(al_ok, np.array(om) * 1e5, "-", color="0.3", zorder=1)
        ax.axhline(0, color="0.8", zorder=-1)
        ax.axvline(0, color="0.8", zorder=-1)
        ax.set(xlabel=r"$\alpha_3 / k_a$", ylabel=r"$\Omega \quad [\mathrm{rad}/\sqrt{m/k_a}] \times 10^{-5}$",
               ylim=[-2 * np.pi, 2 * np.pi], yticks=[-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi],
               yticklabels=[r"$-2\pi$", r"$-\pi$", "0", r"$\pi$", r"$2\pi$"])
    _panel("f_rotation_rate", draw)


def report(walls, rotor):
    print("climb velocity v_y (ell per sqrt(m/k_a)):")
    for n in WALL_NS:
        print(f"  {n} dislocations: " + ", ".join(
            f"alpha3/k_a {s * WALL_ALPHA:+.2f}: {walls[(n, s)]['r']['v'][1]:+.3e}" for s in (-1, 0, 1)
            if walls[(n, s)] is not None))
    print("rotation rate Omega (rad per sqrt(m/k_a)): " + ", ".join(
        f"{al:+.3f}: {w['r']['Omega']:+.3e}" for al, w in rotor.items() if w is not None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", nargs="+", default=list("abcdef"))
    args = ap.parse_args()
    walls, rotor = load_all()
    report(walls, rotor)
    save_derived(FIGURE, dict(
        v_y=np.array([[walls[(n, s)]["r"]["v"][1] if walls[(n, s)] is not None else np.nan for s in (-1, 0, 1)]
                      for n in WALL_NS]),
        Omega=np.array([w["r"]["Omega"] if w is not None else np.nan for w in rotor.values()])))
    for p in args.panel:
        {"a": lambda: panel_a(walls), "b": lambda: panel_b(walls), "c": lambda: panel_c(walls),
         "d": lambda: panel_d(rotor), "e": lambda: panel_e(rotor), "f": lambda: panel_f(rotor)}[p]()


if __name__ == "__main__":
    main()
