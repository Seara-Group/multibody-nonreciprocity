"""SI figure -- non-dispersive lines in the power spectrum are raft-wide collective oscillations.

Panels written to OUT/figures/si_stripes/:
    abc.pdf          (one file, in the published arrangement; 6.5 in wide)
                     a  transverse spectrum along q_y = 0, and the leading-mode share at each frequency
                        (fraction of the raft's power at that frequency carried by its strongest spatial
                        pattern), against a phase-randomised baseline (grey)
                     b  participation ratio of the leading mode, and its weight on particles within
                        1.5 ell of an interior 5- or 7-fold site (Delaunay)
                     c  the leading modes at 78.4 and 143.6 Hz
    d_overlap.pdf    |e_i . e_j| between the eigenvectors of the COMSOL dynamical matrix
    e_mode_power.pdf power of the measured motion projected onto each COMSOL mode; red points: the modes'
                     own frequencies (every hundredth); white cross: fastest-growing mode
    f_growth.pdf     growth rate of the COMSOL modes against frequency without damping; red lines: the
                     measured lines; dashed: half the assumed air drag (187/s)

Spectra: raft A, Welch with 4096-frame Hann windows and 75% overlap.  Lines are peaks of the median over
224 wavevectors (four directions, L and T, q ell = 0.3-3) of each spectrum divided by its median over f.
ell is raft A's raft-average spacing (median Delaunay bond, 54.01 um); panel a's spectrum is the Fig 4b
spectrum (computed on a wavevector grid in units of the g(r) peak, 52.48 um) drawn against q_x ell, as
published.

    python si_stripes.py [--panel abc d e f]
"""
import argparse
import os
import sys

import numpy as np
from scipy.signal import find_peaks, medfilt, welch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from levitation import comsol, io, lattice, spectra  # noqa: E402
from levitation.plotting import plt, save_panel, save_derived  # noqa: E402

FIGURE = "si_stripes"
NPER, NOV = 4096, 3072
FMIN, FMAX = 10.0, 400.0
MAP_LINES = (78.4, 143.6)
MAP_COLORS = ("#E69F00", "#7570B3")        # not red (lines), blue (A/S), magenta/green (defects)
C_FIVE, C_SEVEN, R_POLY = "magenta", "green", 0.62
STRIPES_HZ = (78.75, 104.0, 152.0, 157.5, 236.0, 315.0)   # the lines marked in panel f
GAMMA = 187.0                                              # assumed Stokes drag, 1/s
STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42, "axes.linewidth": 1.0, "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 1.0, "ytick.major.width": 1.0, "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "savefig.dpi": 300,
}


# ------------------------------------------------------------------------------ a - c
def _peaks(f, spec, n=40):
    db = 10 * np.log10(np.maximum(spec, 1e-300))
    ex = db - medfilt(db, max(11, (len(db) // 60) | 1))
    pk, _ = find_peaks(ex, prominence=3.0, distance=3)    # 3 bins: lines 3-4 Hz apart stay distinct
    pk = pk[f[pk] > 5]
    pk = pk[np.argsort(ex[pk])[::-1]][:n]
    return np.sort(f[pk])


def line_analysis(tr, ell):
    dx, r0, dt = tr.dx, tr.mean_pos, tr.dt
    N, T, _ = dx.shape
    v = np.gradient(dx, dt, axis=1)
    f, _ = welch(v[:1, :, :1], fs=1 / dt, nperseg=NPER, noverlap=NOV, axis=1)
    spec = []
    qs = np.linspace(0.3, 3.0, 28)
    for th in np.radians([0, 30, 60, 90]):
        for br in ("L", "T"):
            e = np.array([np.cos(th), np.sin(th)]) if br == "L" else np.array([-np.sin(th), np.cos(th)])
            j = np.exp(-1j * (np.outer(qs / ell, [np.cos(th), np.sin(th)]) @ r0.T)) @ np.einsum("ntc,c->nt", v, e)
            _, Pj = welch(j, fs=1 / dt, nperseg=NPER, noverlap=NOV, axis=1, return_onesided=False)
            spec.append(Pj[:, :len(f)])
    Pk = np.concatenate(spec, 0)
    lines = _peaks(f, np.median(Pk / np.median(Pk, axis=1, keepdims=True), axis=0))

    lat = lattice.delaunay_lattice(r0)
    defect = (np.isin(np.arange(N), lat.fives) | np.isin(np.arange(N), lat.sevens)) & lat.interior
    near = cKDTree(r0[defect]).query(r0)[0] / ell < 1.5

    # leading coherent pattern at every bin: SVD of the (segments x 2N) matrix of transforms
    vv = v.transpose(0, 2, 1).reshape(2 * N, T)
    w = np.hanning(NPER)
    starts = list(range(0, T - NPER + 1, NPER - NOV))
    fb = np.fft.rfftfreq(NPER, dt)
    bins = np.where((fb >= FMIN) & (fb <= FMAX))[0]
    share, pr, near_w = (np.zeros(len(bins)) for _ in range(3))
    modes = {}
    map_bins = {int(np.argmin(np.abs(fb - x))): x for x in MAP_LINES}
    X = np.stack([np.fft.rfft(vv[:, s:s + NPER] * w, axis=1)[:, bins] for s in starts])
    for i, b in enumerate(bins):
        _, sv, Vh = np.linalg.svd(X[:, :, i], full_matrices=False)
        p = (np.abs(Vh[0].conj().reshape(N, 2)) ** 2).sum(1)
        p /= p.sum()
        share[i], near_w[i], pr[i] = sv[0] ** 2 / (sv ** 2).sum(), p[near].sum(), 1.0 / (N * (p ** 2).sum())
        if b in map_bins:
            modes[map_bins[b]] = p
    del X
    # baseline: every coordinate keeps its spectrum, random Fourier phases remove all coherence
    rng = np.random.default_rng(7)
    F = np.fft.rfft(vv, axis=1)
    ph = np.exp(2j * np.pi * rng.random(F.shape))
    ph[:, 0] = 1.0
    if T % 2 == 0:
        ph[:, -1] = 1.0
    vs = np.fft.irfft(F * ph, n=T, axis=1)
    del F, ph
    Xs = np.stack([np.fft.rfft(vs[:, s:s + NPER] * w, axis=1)[:, bins] for s in starts])
    del vs
    share_null = np.array([(lambda s: s[0] ** 2 / (s ** 2).sum())(np.linalg.svd(Xs[:, :, i], compute_uv=False))
                           for i in range(len(bins))])
    rng = np.random.default_rng(0)

    def pr_of(u):
        q = (np.abs(u.reshape(N, 2)) ** 2).sum(1)
        q /= q.sum()
        return 1.0 / (N * (q ** 2).sum())
    pr_random = float(np.mean([pr_of(rng.standard_normal(2 * N) + 1j * rng.standard_normal(2 * N)) for _ in range(200)]))
    d = dict(lines=lines, f_modes=fb[bins], share=share, share_null=share_null, pr=pr, near_w=near_w,
             modes=modes, pr_random=pr_random, near_frac=float(near.mean()), defect=defect,
             fives=lat.fives[lat.interior[lat.fives]], sevens=lat.sevens[lat.interior[lat.sevens]],
             nseg=len(starts), N=N)
    report(d)
    return d


def report(d):
    fm = d["f_modes"]
    print(f"{d['nseg']} Welch segments; lines (elevated at most wavevectors): {np.round(d['lines'], 1)}")
    for x in d["lines"]:
        i = int(np.argmin(np.abs(fm - x)))
        if abs(fm[i] - x) < 1:
            print(f"  {fm[i]:6.1f} Hz  leading-mode share {d['share'][i]:.2f}  PR {d['pr'][i]:.2f}")
    off = np.ones(len(fm), bool)
    for x in d["lines"]:
        off &= np.abs(fm - x) > 3
    print(f"baseline share: median {np.median(d['share_null']):.3f}, 95th percentile "
          f"{np.percentile(d['share_null'], 95):.3f}; between lines: median share {np.median(d['share'][off]):.2f}, "
          f"median PR {np.median(d['pr'][off]):.2f}")
    print(f"weight within 1.5 ell of an interior defect ({100 * d['near_frac']:.0f}% of particles): lines "
          f"{np.median(d['near_w'][~off]):.2f}, between lines {np.median(d['near_w'][off]):.2f} (medians); "
          f"random extended PR {d['pr_random']:.2f}")


def panels_abc(d, W, r0, ell):
    from matplotlib.patches import RegularPolygon
    fm = d["f_modes"]
    matched = [x for x in d["lines"] if FMIN <= x <= FMAX]
    isline = np.zeros(len(fm), bool)
    for x in matched:
        isline |= np.abs(fm - x) < 0.5
    kx, fc, S = W["kx_cut"], W["f_cut"], W["ST_cut"]
    allZ = np.concatenate([np.log10(np.maximum(W[k], 1e-30)).ravel() for k in ("SL_cut", "ST_cut", "SL_az", "ST_az")])
    allZ = allZ[np.isfinite(allZ)]
    vmin, vmax = np.percentile(allZ, 40.0), np.percentile(allZ, 99.5)
    fsel = (fc >= FMIN) & (fc <= FMAX)
    with plt.rc_context(STYLE):
        Wf = 6.5
        L, gs_, ws, gp, R = 0.52, 0.06, 0.95, 0.62, 0.20
        h = (Wf - L - gs_ - ws - gp - R) / 2
        T, gv, B = 0.22, 0.78, 0.45
        Hf = T + h + gv + h + B
        fig = plt.figure(figsize=(Wf, Hf))
        ax_in = lambda x, y, w, hh, **kw: fig.add_axes([x / Wf, y / Hf, w / Wf, hh / Hf], **kw)
        x_spec, x_share = L, L + h + gs_
        x_right, y_top, y_bot = x_share + ws + gp, B + h + gv, B
        mcol = dict(zip(MAP_LINES, MAP_COLORS))
        imap = {x0: int(np.argmin(np.abs(fm - x0))) for x0 in MAP_LINES}

        ax_s = ax_in(x_spec, y_top, h, h)
        ax_s.pcolormesh(kx * ell, fc[fsel], np.log10(np.maximum(S[:, fsel].T, 1e-30)), cmap="gray", vmin=vmin,
                        vmax=vmax, shading="auto", rasterized=True)
        ax_s.set(xlim=(-4, 4), ylim=(FMIN, FMAX), xlabel=r"$q_x \ell$", ylabel="$f$ (Hz)")
        ax_s.set_xticks([-4, -2, 0, 2])
        ax_s.set_title(r"transverse, $q_y = 0$", fontsize=7, loc="left")

        ax_h = ax_in(x_share, y_top, ws, h, sharey=ax_s)
        for x in matched:
            ax_h.axhline(x, color="C3", lw=0.4, alpha=0.6, zorder=0)
        ax_h.fill_betweenx(fm, 0, d["share_null"], color="0.85", lw=0, zorder=0.5)
        ax_h.plot(d["share"], fm, color="k", lw=0.7)
        ax_h.plot(d["share"][isline], fm[isline], "o", ms=2.5, color="C3")
        for x0 in MAP_LINES:
            ax_h.axhline(fm[imap[x0]], color=mcol[x0], lw=0.9, zorder=1)
            ax_h.plot(d["share"][imap[x0]], fm[imap[x0]], "o", ms=3.5, color=mcol[x0], zorder=4)
        ax_h.set(xlim=(0, 1), xlabel="leading-mode\nshare")
        plt.setp(ax_h.get_yticklabels(), visible=False)

        ax_p = ax_in(x_right, y_top, h, h)
        ax_p.semilogy(fm[~isline], d["pr"][~isline], ".", ms=1.2, color="0.6")
        ax_p.semilogy(fm[isline], d["pr"][isline], "o", ms=2.5, color="C3")
        for x0 in MAP_LINES:
            ax_p.semilogy(fm[imap[x0]], d["pr"][imap[x0]], "o", ms=3.5, color=mcol[x0], zorder=4)
        for val, lab, ls in ((d["pr_random"], "random extended", (0, (3, 2))),
                             (5 / d["N"], "5 particles", ":"), (20 / d["N"], "20 particles", ":")):
            ax_p.axhline(val, color="0.35", lw=0.7, ls=ls)
            ax_p.text(FMAX * 0.98, val * 1.12, lab, ha="right", va="bottom", fontsize=6, color="0.35")
        ax_p.set(xlim=(FMIN, FMAX), ylim=(5e-3, 1.5), xlabel="$f$ (Hz)")
        ax_p.set_title("participation ratio", fontsize=7, loc="left")

        P = {x0: d["modes"][x0] * d["N"] for x0 in MAP_LINES}
        vm = np.percentile(np.concatenate(list(P.values())), 99)
        xy = r0 / ell
        mw = (x_share + ws - x_spec) / 2
        for j, x0 in enumerate(MAP_LINES):
            a = ax_in(x_spec + j * mw, y_bot + 0.42, mw, h - 0.42 - 0.02)
            for idx, nv, col in ((d["fives"], 5, C_FIVE), (d["sevens"], 7, C_SEVEN)):
                for q in xy[idx]:
                    a.add_patch(RegularPolygon(tuple(q), numVertices=nv, radius=R_POLY, orientation=0.0,
                                               facecolor=col, edgecolor="none", zorder=1))
            sc = a.scatter(xy[:, 0], xy[:, 1], c=P[x0], s=6, cmap="Greys", vmin=-0.08 * vm, vmax=vm,
                           linewidths=0, zorder=2)
            a.set_aspect("equal")
            a.axis("off")
            a.set_title(f"leading mode, {fm[imap[x0]]:.1f} Hz", fontsize=7, loc="left", color=mcol[x0])
        span = x_share + ws - x_spec
        cb = plt.colorbar(sc, cax=ax_in(x_spec + 0.25 * span, y_bot + 0.28, 0.5 * span, 0.07), orientation="horizontal")
        cb.set_label(r"$N p_i$, share of the mode on particle $i$ times $N$", labelpad=2)

        ax_w = ax_in(x_right, y_bot, h, h)
        ax_w.plot(fm[~isline], d["near_w"][~isline], ".", ms=1.2, color="0.6")
        ax_w.plot(fm[isline], d["near_w"][isline], "o", ms=2.5, color="C3")
        for x0 in MAP_LINES:
            ax_w.plot(fm[imap[x0]], d["near_w"][imap[x0]], "o", ms=3.5, color=mcol[x0], zorder=4)
        ax_w.axhline(d["near_frac"], color="0.35", lw=0.7, ls=(0, (3, 2)))
        ax_w.text(FMAX * 0.98, d["near_frac"] * 0.93, "uniform", ha="right", va="top", fontsize=6, color="0.35")
        ax_w.set(xlim=(FMIN, FMAX), ylim=(0, 0.7), xlabel="$f$ (Hz)",
                 ylabel="mode weight within $1.5\\ell$\nof a defect")
        save_panel(fig, FIGURE, "abc", bbox_inches=None)


# ------------------------------------------------------------------------------ d - f
def comsol_modes(tr):
    lat, D = comsol.experimental_dynamical_matrix(neighbours=50)
    freq, E, ev, gr = comsol.eigenmodes(D)
    E = E / np.linalg.norm(E, axis=0)
    fD = freq * comsol.HZ
    fastest = int(np.argsort(np.where(np.isfinite(gr) & (freq > 0), gr, -np.inf))[-1])
    s0 = np.sqrt(-ev * (2 * np.pi * comsol.HZ) ** 2 + 0j)
    s0 = np.where(s0.real >= 0, s0, -s0)
    fmode, grow = np.abs(s0.imag) / 2 / np.pi, s0.real
    tol = 1e-9 * np.abs(ev).max()
    cplx = (np.abs(ev.imag) > tol) & (ev.real > 0)
    f, PD = comsol.mode_power(comsol.displacements_comsol_order(tr), E.conj(), tr.dt)
    print(f"COMSOL modes: {cplx.sum() // 2} complex pairs with gain/loss, {(grow[cplx] > GAMMA / 2).sum()} grow "
          f"faster than half the drag; fastest-growing: mode {fastest}, {fmode[fastest]:.0f} Hz, {grow[fastest]:.0f}/s")
    return dict(E=E, fD=fD, fastest=fastest, fmode=fmode, grow=grow, cplx=cplx, f=f, PD=PD)


def panel_d(m):
    fig, ax = plt.subplots(figsize=(2.6, 2.6), dpi=144)
    c = ax.pcolormesh(np.abs(m["E"].T @ m["E"]), vmin=0, vmax=0.1, cmap="Blues", rasterized=True)
    n = m["PD"].shape[0]
    ax.set(xlim=(0, n), ylim=(0, n), xlabel="mode index", ylabel="mode index")
    ax.set_box_aspect(1)
    fig.colorbar(c, cax=ax.inset_axes([1.05, 0, 0.05, 1]), label=r"$|\hat{e}_i \cdot \hat{e}_j|$",
                 ticks=[0, 0.05, 0.1], extend="max")
    save_panel(fig, FIGURE, "d_overlap")


def panel_e(m):
    P = m["PD"]
    Z = np.log10(P.T)
    v0, v1 = np.percentile(Z, [5, 99.7])
    fig, ax = plt.subplots(figsize=(2.6, 2.6), dpi=144)
    c = ax.pcolormesh(np.arange(P.shape[0]), m["f"], Z, cmap="gray_r", vmin=v0, vmax=v1, shading="auto", rasterized=True)
    idx = np.arange(100, 800, 100)
    ax.plot(idx, m["fD"][idx], "o", c="tab:red", ms=2.5, alpha=0.7)
    ax.plot([m["fastest"]], [m["fD"][m["fastest"]]], "X", c="w", mec="k", ms=6, mew=0.6)
    ax.set(xlim=(0, P.shape[0]), ylim=(0, 1400), xlabel="mode index", ylabel="Frequency (Hz)")
    ax.set_box_aspect(1)
    fig.colorbar(c, cax=ax.inset_axes([1.05, 0, 0.05, 1]), label=r"$\log_{10}$ power")
    save_panel(fig, FIGURE, "e_mode_power")


def panel_f(m):
    cplx, grow, fmode, ms = m["cplx"], m["grow"], m["fmode"], m["fastest"]
    gmax = 1.08 * grow[cplx].max()
    fig, ax = plt.subplots(figsize=(2.6, 2.6), dpi=144)
    for i, q in enumerate(STRIPES_HZ):
        ax.axvline(q, c="tab:red", lw=0.7, alpha=0.8, zorder=0, label="measured stripes" if i == 0 else None)
    ax.axhline(GAMMA / 2, c="k", ls="--", lw=0.7, label="air drag (assumed)")
    ax.plot(fmode[cplx], grow[cplx], ".", c="0.25", ms=3.5, label="FEM modes")
    ax.plot([fmode[ms]], [grow[ms]], "X", c="w", mec="k", ms=6, mew=0.6, label="fastest-growing")
    ax.set(xlim=(0, 400), ylim=(0, 1.45 * gmax), xlabel="mode frequency (Hz)", ylabel="growth rate (1/s)")
    ax.set_box_aspect(1)
    ax.legend(fontsize=6, frameon=False, loc="upper center", ncol=2, handlelength=1.2, columnspacing=0.8,
              labelspacing=0.3)
    save_panel(fig, FIGURE, "f_growth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", nargs="+", default=["abc", "d", "e", "f"])
    want = set(ap.parse_args().panel)
    tr = io.load_raft_a()
    out = {}
    if "abc" in want:
        W = spectra.raft_a_spectra()
        ell = lattice.median_bond(tr.mean_pos)     # raft-average spacing (54.01 um), as published
        d = line_analysis(tr, ell)
        panels_abc(d, W, tr.mean_pos, ell)
        out.update(stripe_lines=d["lines"], f_modes=d["f_modes"], share=d["share"], pr=d["pr"],
                   near_w=d["near_w"], share_null=d["share_null"])
    if want & {"d", "e", "f"}:
        m = comsol_modes(tr)
        out.update(comsol_fD=m["fD"], comsol_grow=m["grow"], comsol_fmode=m["fmode"], comsol_fastest=m["fastest"])
        if "d" in want:
            panel_d(m)
        if "e" in want:
            panel_e(m)
        if "f" in want:
            panel_f(m)
    save_derived(FIGURE, out)


if __name__ == "__main__":
    main()
