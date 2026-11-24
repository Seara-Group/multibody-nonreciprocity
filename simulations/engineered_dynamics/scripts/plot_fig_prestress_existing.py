#!/usr/bin/env python3
"""
plot_fig_prestress_existing.py -- supplementary pre-stress figure built ONLY
from runs already in data/runs (configs/grid.csv): no new simulations.

Uniform edge pressure (prestress = 'pressure', P ~ 0.28 k_a, 7.5% uniform
compression) against no pre-stress, for the two designs the main grid ran: the
tilt wall of FOUR dislocations ('wall4') and the six-dislocation ring ('rotor'),
at alpha3/k_a = 0, 0.003, 0.033, 0.1 (positive only).  Underdamped, zeta = 0.01,
except panels d and h, which scan zeta.

  a  wall of 4 under pressure, particle trails over 4 l (alpha3 = 0.1)
  b  dy(t), none (solid) vs pressure (dashed), one colour per alpha3
  c  climb velocity v_y against alpha3: none (filled) vs pressure (open)
  d  v_y * gamma / alpha3 against damping ratio zeta (alpha3 = 0.033)
  e  ring of 6 under pressure, particle trails over 30 deg (alpha3 = 0.1)
  f  theta(t), none (solid) vs pressure (dashed)
  g  angular velocity against alpha3: none (filled) vs pressure (open)
  h  theta_dot * gamma / alpha3 against zeta (alpha3 = 0.033)

Units: l = k_a = m = 1, time in sqrt(m/k_a), gamma = 2 zeta.  Rates are each
run's own fit over its last 60% (run.py); d and h read figs/summary.csv.

    python plot_fig_prestress_existing.py [--runs ../data/runs] [--out ../figs]

Writes OUT/fig_engineered_activity_prestress_existing.{pdf,png}.
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot_fig_underdamped as U          # trail maps, loader

ROOT = os.path.dirname(HERE)
ALPHAS = [0.0, 0.003, 0.033, 0.1]
ZETA_SCAN_ALPHA = 0.033                    # above zeta_c at every zeta in the grid
try:
    import cmasher as cmr                  # warm half of the main figure's iceburn
    _c = cmr.take_cmap_colors("cmr.iceburn", N=7, cmap_range=(0.1, 0.9))[4:]
except ImportError:
    _c = [plt.cm.Oranges(x) for x in (0.45, 0.7, 0.95)]
ACOL = {0.0: "k", 0.003: _c[0], 0.033: _c[1], 0.1: _c[2]}
STYLE = {"none": dict(ls="-", filled=True), "pressure": dict(ls="--", filled=False)}


def pi_axis(ax, lo, hi):
    t = [v for v in (-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi) if lo <= v <= hi]
    lab = {-2: r"$-2\pi$", -1: r"$-\pi$", 0: "0", 1: r"$\pi$", 2: r"$2\pi$"}
    ax.set(ylim=[lo, hi], yticks=t, yticklabels=[lab[int(round(v / np.pi))] for v in t])


def marker(ax, x, y, col, filled, **kw):
    ax.plot(x, y, "o", ms=4.5, mew=1.0, zorder=3,
            mfc=col if filled else "w", mec="w" if filled else col, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.path.join(ROOT, "data", "runs"))
    ap.add_argument("--summary", default=os.path.join(ROOT, "figs", "summary.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "figs"))
    ap.add_argument("--zeta", type=float, default=0.01)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    miss = []
    nm = lambda d, p, al: "%s_%s_z%g_a%g" % (d, p, a.zeta, al)
    run = {(d, p, al): U.load(a.runs, nm(d, p, al), miss)
           for d in ("wall4", "rotor") for p in STYLE for al in ALPHAS}

    fig, ax = plt.subplots(2, 4, figsize=(6.5, 3.9), dpi=144,   # 6.5 in = text width of a 1-in-margin letter page
                           gridspec_kw=dict(width_ratios=[1.3, 1, 1, 1]))   # wide raft column
    tunit = r"$\sqrt{m/k_a}$"

    # a, e: trail maps under pressure -----------------------------------------
    w = run[("wall4", "pressure", 0.1)]
    if w is not None:
        R = w["z"]["trajectory_R"].astype(float)
        c = R.mean(1); dd = np.linalg.norm(c - c[0], axis=1)
        k = int(np.argmin(np.abs(dd - 4.0)))
        U.trail_map(ax[0, 0], R, k, lw=0.5)
        print("a: trail t = 0 .. %.4g, %.2f l" % (w["z"]["times"][k], dd[k]))
    w = run[("rotor", "pressure", 0.1)]
    if w is not None:
        R = w["z"]["trajectory_R"].astype(float); tf = w["z"]["times"]
        k = int(np.argmin(np.abs(tf - np.interp(np.radians(30), np.abs(w["theta"]), w["t"]))))
        U.trail_map(ax[1, 0], R, k, lw=0.5)
        print("e: trail t = 0 .. %.4g" % tf[k])

    # b, f: time traces --------------------------------------------------------
    for row, d, key in ((0, "wall4", "dy"), (1, "rotor", "theta")):
        ref = run[(d, "pressure", 0.1)]
        twin = ref["t"][-1] if ref is not None else None
        for (dd_, p, al), w in run.items():
            if dd_ != d or w is None:
                continue
            m = w["t"] <= (twin if twin else w["t"][-1])
            y = w["com"][m, 1] - w["com"][0, 1] if key == "dy" else w["theta"][m]
            ax[row, 1].plot(w["t"][m], y, STYLE[p]["ls"], color=ACOL[al], lw=1.2)
        ax[row, 1].axhline(0, color="0.8", zorder=-1)
        ax[row, 1].set(xlabel="time")
    ax[0, 1].set(ylabel=r"$\Delta y$")
    ax[1, 1].set(ylabel=r"$\theta$")
    pi_axis(ax[1, 1], -2.15 * np.pi, 0.15 * np.pi)   # all rates are negative

    # c, g: rate against alpha3 --------------------------------------------------
    for row, d, get, scale in ((0, "wall4", lambda r: r["v"][1], 1e2),
                               (1, "rotor", lambda r: r["Omega"], 1e5)):
        for p, st in STYLE.items():
            xs, ys = [], []
            for al in ALPHAS:
                w = run[(d, p, al)]
                if w is None:
                    continue
                xs.append(al); ys.append(get(w["r"]) * scale)
                marker(ax[row, 2], al, ys[-1], ACOL[al], st["filled"])
            ax[row, 2].plot(xs, ys, st["ls"].replace("--", ":"), color="0.3", lw=1, zorder=1)
        ax[row, 2].axhline(0, color="0.8", zorder=-1)
        ax[row, 2].set(xlabel=r"$\alpha_3 / k_a$")
    ax[0, 2].set(ylabel=r"$v_y \times 10^{2}$")
    ax[1, 2].set(ylabel=r"$\dot\theta \times 10^{5}$")
    pi_axis(ax[1, 2], -2.3 * np.pi, 0.15 * np.pi)

    # d, h: damping scan from summary.csv ---------------------------------------
    s = pd.read_csv(a.summary)
    s = s[(s["mode"] == "underdamped") & (s.kT == 0) & np.isclose(s.alpha, ZETA_SCAN_ALPHA)]
    for row, d, col_, lab, sc in ((0, "wall4", "vy", r"$v_y\,\gamma / \alpha_3 \times 10^{3}$", 1e3),
                              (1, "rotor", "Omega", r"$\dot\theta\,\gamma / \alpha_3 \times 10^{5}$", 1e5)):
        for p, st in STYLE.items():
            x = s[(s.design == d) & (s.prestress == p)].sort_values("zeta")
            y = x[col_] * x.gamma / ZETA_SCAN_ALPHA * sc
            ax[row, 3].plot(x.zeta, y, st["ls"].replace("--", ":"), color="0.3", lw=1, zorder=1)
            for z_, y_ in zip(x.zeta, y):
                marker(ax[row, 3], z_, y_, ACOL[ZETA_SCAN_ALPHA], st["filled"])
        ax[row, 3].set(xscale="log", xlabel=r"$\zeta$", ylabel=lab)
        lo, hi = ax[row, 3].get_ylim(); pad = 0.6 * (hi - lo)
        ax[row, 3].set_ylim(lo - pad, hi + pad)        # flat lines: keep the range honest

    # legend ----------------------------------------------------------------------
    h = [Line2D([], [], color=ACOL[al], lw=2, label=r"$\alpha_3/k_a = %g$" % al) for al in ALPHAS]
    h += [Line2D([], [], color="0.3", ls="-", marker="o", mfc="0.3", mec="w", label="no pre-stress"),
          Line2D([], [], color="0.3", ls="--", marker="o", mfc="w", mec="0.3", label="uniform edge pressure")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=9)

    for x, lab in zip(ax.ravel(), "abcdefgh"):
        if x in (ax[0, 0], ax[1, 0]):
            continue                                     # raft letters placed after layout
        x.set_box_aspect(1)
        x.text(-0.15, 1.05, lab, transform=x.transAxes, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.15, 1, 1), pad=0.3, w_pad=0.9, h_pad=0.5)
    # raft maps: fill the whole row height and the space up to panel b's labels
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for r in (0, 1):
        bbs = [inv.transform_bbox(x.get_tightbbox(rend)) for x in ax[r, 1:]]
        y0, y1 = min(b.y0 for b in bbs), max(b.y1 for b in bbs)
        x1 = bbs[0].x0 - 0.005
        ax[r, 0].set_position([0.02, y0, x1 - 0.02, y1 - y0])
        ly = inv.transform(ax[r, 1].transAxes.transform((0, 1.05)))[1]   # same height as b / f
        fig.text(0.01, ly, "ae"[r], fontsize=12, fontweight="bold")
    base = os.path.join(a.out, "fig_engineered_activity_prestress_existing")
    fig.savefig(base + ".pdf")                    # no bbox cropping: page is exactly 6.5 in wide
    fig.savefig(base + ".png", dpi=144)
    plt.close(fig)
    print("wrote", base + ".pdf")

    print("\nratio pressure / none (zeta = %g):" % a.zeta)
    for al in ALPHAS[1:]:
        r = [run[(d, p, al)] for d in ("wall4", "rotor") for p in STYLE]
        if all(x is not None for x in r):
            print("  alpha3=%.3f  wall4 v_y: %.3f   rotor Omega: %.3f"
                  % (al, r[1]["r"]["v"][1] / r[0]["r"]["v"][1], r[3]["r"]["Omega"] / r[2]["r"]["Omega"]))
    if miss:
        print("missing:", miss)


if __name__ == "__main__":
    main()
