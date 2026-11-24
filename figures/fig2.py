"""Fig 2 -- individual and collective activity in the rafts (raft A).

Panels written to OUT/figures/fig2/:
    a_particle_winding.pdf   winding number of every particle about its mean position, three
                             highlighted (largest +, largest -, closest to zero), with 100 ms of
                             their trajectories above
    b_L_map.pdf              time-averaged specific angular momentum L_i of each particle
                             (first 29 999 frames)
    c_L_histogram.pdf        histogram of L_i (all frames) against a phase-randomised noise floor
    d_pca_modes.pdf          the five highest-variance PCA modes, with their variance fractions
    e_pair_winding.pdf       winding number of every pair of the first 48 modes; PC3-PC4,
                             PC5-PC7 and a control pair highlighted, with 20 ms of trajectory
    f_cycling_matrix.pdf     cycling rate omega_mn of the first 48 modes; one phase-randomised
                             surrogate fills the redundant triangle
    g_cycling_spectrum.pdf   eigenvalue spectrum of the cycling matrix against 100 surrogates

Rigid translation and rotation of the raft are removed before panels a and d-g (not for b, c).

    python fig2.py [--panel a b c d e f g]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from levitation import io, modes  # noqa: E402
from levitation.plotting import plt, save_panel, save_derived  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
import cmasher as cmr  # noqa: E402

FIGURE = "fig2"
N_MODES = 48                 # modes in panels e-g
N_SHOW = 5                   # modes drawn in panel d
PAIRS_E = [(2, 3), (4, 6)]   # PC3-PC4 and PC5-PC7 (0-indexed); one winds each way
N_NULL_PARTICLE = 20         # surrogates per particle, panels a and c
N_NULL_MODES = 100           # surrogates of the mode amplitudes, panels e-g
CMAP = cmr.get_sub_cmap("cmr.viola", 0.15, 0.85)
ARROW = 1.5 * 54.0           # arrow length of the mode shapes, um (as published)


# ------------------------------------------------------------------ panel a
def panel_a(tr, Uc, sub=60, window_s=0.1):
    dt = tr.dt
    tt, turns, rate = modes.particle_winding(Uc, dt, sub)
    nc, nr = modes.particle_winding_null(Uc, dt, N_NULL_PARTICLE, sub)
    lo, hi = np.percentile(nc, [2.5, 97.5], axis=0)
    thr = np.percentile(np.abs(nr), 97.5)
    N = turns.shape[0]
    print(f"particle winding: rate {rate.min():+.1f} to {rate.max():+.1f} Hz, median |rate| "
          f"{np.median(np.abs(rate)):.1f} Hz, {100 * (rate > 0).mean():.0f}% positive; null "
          f"median |rate| {np.median(np.abs(nr)):.2f} Hz; {100 * (np.abs(rate) > thr).mean():.1f}% "
          f"of particles exceed the null's 97.5th percentile ({thr:.2f} Hz)")

    pos, neg, ctl = int(np.argmax(rate)), int(np.argmin(rate)), int(np.argmin(np.abs(rate)))
    sel, colors = [neg, pos, ctl], [CMAP(0.1), CMAP(0.9), "forestgreen"]
    ux = Uc[:, 0::2] / Uc[:, 0::2].std(0)
    uy = Uc[:, 1::2] / Uc[:, 1::2].std(0)

    fig, ax0 = plt.subplots(figsize=(3, 3.5), dpi=144)
    ax0.plot(tt, lo, color="0.25", lw=0.8, ls=(0, (4, 2)), zorder=2.7)
    ax0.plot(tt, hi, color="0.25", lw=0.8, ls=(0, (4, 2)), zorder=2.7)
    ax0.add_collection(LineCollection([np.column_stack([tt, turns[i]]) for i in range(N)],
                                      colors="k", linewidths=0.4, alpha=0.10, rasterized=True, zorder=2))
    for i, c in zip(sel, colors):
        ax0.plot(tt, turns[i], color=c, lw=1.8, solid_capstyle="round", zorder=3)
    ymax = 1.05 * np.abs(turns).max()
    ax0.set(xlim=(0, tt[-1]), ylim=(-ymax, ymax), xlabel="time (s)", ylabel="winding number $W_i$")
    ax0.axhline(0, color="0.6", lw=0.5, zorder=1)
    ax0.set_box_aspect(1)
    ax0.text(0.03, 0.46, f"all {N} particles", transform=ax0.transAxes, fontsize=7, va="top")
    nw = int(window_s / dt)
    ax0.text(0.055, 0.98, f"{1e3 * window_s:.0f} ms of trajectory", transform=ax0.transAxes,
             fontsize=6, va="top")
    for k, (i, c) in enumerate(zip(sel, colors)):
        axi = ax0.inset_axes([0.055 + 0.30 * k, 1.05, 0.25, 0.25], zorder=5)
        axi.plot(ux[:nw, i], uy[:nw, i], color=c, lw=0.5)
        axi.set(xlim=(-3.2, 3.2), ylim=(-3.2, 3.2), xticks=[], yticks=[])
        axi.set_box_aspect(1)
        lab = f"{rate[i]:+.0f} Hz" if abs(rate[i]) >= 0.5 else f"{rate[i]:+.2f} Hz"
        axi.text(0.5, -0.16, lab, transform=axi.transAxes, fontsize=6, va="top", ha="center", color=c)
    save_panel(fig, FIGURE, "a_particle_winding")
    return dict(winding_rate=rate, winding_turns=turns, winding_null_rate=nr,
                winding_highlighted=np.array(sel))


# --------------------------------------------------------------- panels b, c
def panel_b(tr):
    L = modes.angular_momentum(tr.dx, tr.dt, frames=slice(0, 29999))
    vmax = 2e3
    fig, ax = plt.subplots(dpi=144, figsize=(3.5, 3.5))
    sc = ax.scatter(*tr.mean_pos.T, c=L, cmap=CMAP, edgecolors="None", vmin=-vmax, vmax=vmax)
    cax = ax.inset_axes([0.1, -0.05, 0.8, 0.05])
    fig.colorbar(sc, cax=cax, label=r"$\overline{L}_i$ ($\mu$m$^2$/s)", ticks=[-vmax, 0, vmax],
                 orientation="horizontal", extend="both")
    ax.set_axis_off()
    save_panel(fig, FIGURE, "b_L_map")
    return dict(L_map=L)


def panel_c(tr):
    L = modes.angular_momentum(tr.dx, tr.dt)
    null = modes.angular_momentum_null(tr.dx, tr.dt, N_NULL_PARTICLE)
    print(f"L_i: data std {L.std():.0f} um^2/s, noise floor std {null.std():.0f} um^2/s "
          f"({L.std() / null.std():.1f}x); {N_NULL_PARTICLE} surrogates x {null.shape[1]} particles")
    bins_pos = np.arange(-300, 9000, 600)
    bins = np.concatenate([-bins_pos[::-1][:-2], bins_pos])
    fig, ax = plt.subplots(dpi=144, figsize=(3, 3))
    ax.hist(L, bins=bins, color="0.2", edgecolor="w", density=True, log=True, lw=1, label="data")
    fb = np.linspace(bins[0], bins[-1], 121)
    h, _ = np.histogram(null.ravel(), bins=fb, density=True)
    c = 0.5 * (fb[1:] + fb[:-1])
    ax.plot(c[h > 0], h[h > 0], color="firebrick", lw=1.5, label="noise floor")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set(xlabel=r"$\overline{L}_i$ ($\mu$m$^2$/s)", ylabel="pdf", ylim=[1e-6, 5e-3], xlim=[-1e4, 1e4])
    ax.set_box_aspect(1)
    save_panel(fig, FIGURE, "c_L_histogram")
    return dict(L=L, L_null=null)


# ------------------------------------------------------------ panels d - g
def mode_analysis(tr, Uc):
    V, var, var_total = modes.pca(Uc, N_MODES)
    a, adot = modes.mode_amplitudes(Uc, V, tr.dt)
    om = modes.cycling_matrix(a, adot)
    om_null, lam_null = modes.cycling_null(a, tr.dt, N_NULL_MODES)
    Z = (om - om_null.mean(0)) / np.where(om_null.std(0) > 0, om_null.std(0), np.inf)
    lam = modes.cycling_spectrum(om)[:lam_null.shape[1]]
    print("variance fractions of the first 10 modes:", np.round(var[:10] / var_total, 3))
    iu = np.triu_indices(N_MODES, 1)
    print(f"cycling matrix ({N_MODES} modes): max|omega| {np.abs(om).max():.0f} rad/s vs null "
          f"{np.abs(om_null).max(axis=(1, 2)).mean():.0f} +- {np.abs(om_null).max(axis=(1, 2)).std():.0f}; "
          f"lambda_1 {lam[0]:.0f} vs null {lam_null[:, 0].mean():.0f} +- {lam_null[:, 0].std():.0f} "
          f"(z = {(lam[0] - lam_null[:, 0].mean()) / lam_null[:, 0].std():.0f}); "
          f"{100 * (np.abs(Z[iu]) > 5).mean():.0f}% of pairs exceed |z| = 5")
    return dict(V=V, var=var, var_total=var_total, a=a, om=om, om_null=om_null, Z=Z,
                lam=lam, lam_null=lam_null)


def panel_d(tr, m):
    pos = tr.mean_pos
    fig, axes = plt.subplots(1, N_SHOW, figsize=(2.2 * N_SHOW, 2.4), dpi=144)
    for k, ax in enumerate(axes):
        ax.scatter(*pos.T, s=2, color="0.85", rasterized=True)
        ax.quiver(pos[:, 0], pos[:, 1], ARROW * m["V"][0::2, k], ARROW * m["V"][1::2, k],
                  pivot="mid", color="k", width=0.006)
        ax.set_title(f"PC{k + 1}  {100 * m['var'][k] / m['var_total']:.1f}%", fontsize=9)
        ax.set_aspect("equal")
        ax.set_axis_off()
    save_panel(fig, FIGURE, "d_pca_modes")


def panel_e(tr, m, sub=60, window_s=0.02):
    om, Z, dt = m["om"], m["Z"], tr.dt
    ah = m["a"] / m["a"].std(0)
    pairs, tt, turns, rate = modes.pair_winding(m["a"], dt, sub)
    row = {pr: i for i, pr in enumerate(pairs)}
    # control: the pair among the first 10 modes whose cycling rate is closest to its null
    Zc = np.abs(Z[:10, :10]) + np.tril(np.full((10, 10), np.inf))
    ctl = tuple(int(i) for i in np.unravel_index(np.argmin(Zc), Zc.shape))
    neg = next(pr for pr in PAIRS_E if rate[row[pr]] < 0)
    pos = next(pr for pr in PAIRS_E if rate[row[pr]] > 0)
    sel = [neg, pos, ctl]
    norm = Normalize(*(np.array([-1, 1]) * float(np.abs(om[np.triu_indices(len(om), 1)]).max())))
    colors = [CMAP(norm(om[p])) for p in sel[:2]] + ["forestgreen"]

    fig = plt.figure(figsize=(3.4 + 1.55 * 3, 2.9), dpi=144)
    gs = fig.add_gridspec(1, 4, width_ratios=[2.1, 1, 1, 1], wspace=0.45)
    ax0 = fig.add_subplot(gs[0])
    ax0.add_collection(LineCollection([np.column_stack([tt, turns[i]]) for i in range(len(pairs))],
                                      colors="k", linewidths=0.4, alpha=0.05, rasterized=True))
    for p, c in zip(sel, colors):
        ax0.plot(tt, turns[row[p]], color=c, lw=1.8, solid_capstyle="round", zorder=3)
    ymax = 1.04 * np.abs(turns).max()
    ax0.set(xlim=(0, len(ah) * dt), ylim=(-ymax, ymax), xlabel="time (s)", ylabel="winding number")
    ax0.axhline(0, color="0.7", lw=0.5, zorder=0)
    ax0.set_box_aspect(1)
    nw = int(window_s / dt)
    for k, ((i, j), role, c) in enumerate(zip(sel, ("negative", "positive", "control"), colors)):
        ax = fig.add_subplot(gs[1 + k])
        ax.plot(ah[:nw, i], ah[:nw, j], color=c, lw=0.7)
        ax.set(xlim=(-3, 3), ylim=(-3, 3), xticks=[-2, 0, 2], yticks=[-2, 0, 2])
        ax.tick_params(labelsize=6)
        ax.set_xlabel(rf"$a_{{{i + 1}}}/\sigma_{{{i + 1}}}$", fontsize=7)
        ax.set_ylabel(rf"$a_{{{j + 1}}}/\sigma_{{{j + 1}}}$", fontsize=7)
        ax.set_title(f"PC{i + 1}-PC{j + 1}", fontsize=8, color=c)
        ax.text(0.04, 0.96, f"{rate[row[(i, j)]]:+.0f} Hz\n" + rf"$z={Z[i, j]:+.0f}$",
                transform=ax.transAxes, fontsize=6, va="top", ha="left")
        ax.set_box_aspect(1)
        print(f"  {role:8s} PC{i + 1}-PC{j + 1}: winding {rate[row[(i, j)]]:+.1f} Hz, "
              f"{turns[row[(i, j)], -1]:+.0f} turns, omega {om[i, j]:+.0f} rad/s, z {Z[i, j]:+.1f}")
    save_panel(fig, FIGURE, "e_pair_winding")
    return dict(pair_rate=rate, pairs_highlighted=np.array(sel))


def panel_f(m):
    om, surr = m["om"], m["om_null"][0].astype(np.float64)
    H = len(om)
    up, lo = np.tril_indices(H, -1), np.triu_indices(H, 1)
    Mx = np.full((H, H), np.nan)
    Mx[lo] = om[lo]
    Mx[up] = surr[up]
    vmax = float(np.abs(om[lo]).max())
    fig, ax = plt.subplots(figsize=(3.6, 3.2), dpi=144)
    ax.set_facecolor("w")
    im = ax.imshow(np.ma.masked_invalid(Mx), cmap=CMAP, vmin=-vmax, vmax=vmax, origin="lower")
    ax.axline([0, 0], slope=1, color="k", lw=0.8)
    ticks = [0] + list(range(9, H, 10))
    ax.set_xticks(ticks, [str(i + 1) for i in ticks])
    ax.set_yticks(ticks, [str(i + 1) for i in ticks])
    ax.set(xlabel="PC mode", ylabel="PC mode")
    ax.text(0.04, 0.95, "null", transform=ax.transAxes, fontsize=8, va="top", ha="left")
    ax.text(0.96, 0.06, "data", transform=ax.transAxes, fontsize=8, va="bottom", ha="right")
    cax = ax.inset_axes([0, 1.05, 1, 0.05])
    fig.colorbar(im, cax=cax, ax=ax, orientation="horizontal", label=r"cycling rate $\omega_{mn}$ (rad/s)")
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")
    save_panel(fig, FIGURE, "f_cycling_matrix")


def panel_g(m):
    lam, lam_null = m["lam"], m["lam_null"]
    k = np.arange(1, len(lam) + 1)
    lo, hi = np.percentile(lam_null, [2.5, 97.5], axis=0)
    fig, ax = plt.subplots(figsize=(3.6, 3.0), dpi=144)
    ax.fill_between(k, lo, hi, color="0.8", label="phase-randomized null (95%)")
    ax.plot(k, lam_null.mean(0), color="0.4", lw=0.8)
    ax.plot(k, lam, "o-", ms=3, color="C3", lw=1.0, label="data")
    ax.set_yscale("log")
    ax.set(xlabel="cycling plane $k$", ylabel=r"$\lambda_k$ of $\Omega$ (rad/s)")
    ax.legend(fontsize=7, frameon=False)
    ax.set_box_aspect(1)
    save_panel(fig, FIGURE, "g_cycling_spectrum")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", nargs="+", default=list("abcdefg"))
    want = set(ap.parse_args().panel)
    tr = io.load_raft_a()
    out = dict(mean_pos=tr.mean_pos)
    if "b" in want:
        out.update(panel_b(tr))
    if "c" in want:
        out.update(panel_c(tr))
    if want & set("adefg"):
        Uc = modes.internal_displacements(tr)
        if "a" in want:
            out.update(panel_a(tr, Uc))
        if want & set("defg"):
            m = mode_analysis(tr, Uc)
            out.update(V=m["V"], variances=m["var"], var_total=m["var_total"], omega=m["om"],
                       omega_z=m["Z"], lam=m["lam"], lam_null=m["lam_null"])
            if "d" in want:
                panel_d(tr, m)
            if "e" in want:
                out.update(panel_e(tr, m))
            if "f" in want:
                panel_f(m)
            if "g" in want:
                panel_g(m)
    save_derived(FIGURE, out)


if __name__ == "__main__":
    main()
