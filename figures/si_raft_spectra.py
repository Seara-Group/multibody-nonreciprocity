"""SI figure -- measured spectra and the fitted dispersion for the three raft-forming conditions.

Panels written to OUT/figures/si_raft_spectra/, one row per raft (A, B, D):
    raft_<X>.pdf   left: one frame of the raft, particles at the tracker's radius, interior 5- and
                   7-fold sites (Delaunay) pink and green; middle and right: longitudinal and
                   transverse velocity-flux spectra along M <- Gamma -> K in the co-rotating lattice
                   frame, each frequency row divided by its median over |k|, with the ridge points
                   that entered the fit and the fitted dispersion (L solid, T dashed).

    python si_raft_spectra.py [--rafts A B D]
"""
import argparse
import os
import sys

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from levitation import dispersion, lattice, rafts  # noqa: E402
from levitation.dispersion import DIRS, EDGE  # noqa: E402
from levitation.plotting import plt, save_panel  # noqa: E402

FIGURE = "si_raft_spectra"
RED = "FireBrick"
STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42, "axes.linewidth": 1.0, "xtick.direction": "out", "ytick.direction": "out",
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6,
}


def raft_row(tag, fit):
    D = dispersion.load_segments(tag)
    SL, ST = np.mean([d["SL"] for d in D], 0), np.mean([d["ST"] for d in D], 0)
    q = D[0]["ka"] * fit["ell_over_closest"]                 # k a (a: closest-neighbour spacing, as published)
    f = dispersion.freqs()
    R = rafts.Raft(tag)
    t = R.T // 2
    P = (R.frame(t) - R.frame(t).mean(0)) * R.um_per_px
    rad = R.radii(t) * R.um_per_px
    a_um = R.a_closest_px * R.um_per_px                       # closest-neighbour spacing
    lat = lattice.delaunay_lattice(P)
    rr = np.linalg.norm(P, axis=1)
    interior = rr < rr.max() - 1.5 * a_um
    z = lat.coordination

    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.55), constrained_layout=True)
        axs[2].sharey(axs[1])
        ax = axs[0]
        cols = np.where(interior & (z == 5), "#ca239d", np.where(interior & (z == 7), "#23991c", "0.75"))
        ax.add_collection(PatchCollection([Circle(p, r) for p, r in zip(P, rad)], facecolors=cols,
                                          edgecolors="0.3", linewidths=0.15))
        lim = np.abs(P).max() + 2 * rad.max()
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="x (µm)", ylabel="y (µm)")
        ax.set_aspect("equal")
        ax.set_box_aspect(1)
        ax.set_title(f"Raft {tag}: {R.label}\n{len(P)} particles, a = {a_um:.1f} µm\n"
                     "5-fold pink, 7-fold green")

        mM = q <= EDGE["M"] + 1e-9
        x = np.concatenate([-q[mM][::-1], q])
        sel = f <= 1000
        fz = fit["fits"]["all"]
        for ax, S, name, br in ((axs[1], SL, "Longitudinal", "L"), (axs[2], ST, "Transverse", "T")):
            Z = np.concatenate([S[1][mM][::-1], S[0]], 0)[:, sel]
            Z = Z / np.median(Z, axis=0, keepdims=True)
            im = ax.pcolormesh(x, f[sel], Z.T, norm=LogNorm(0.3, np.percentile(Z, 99.5)), shading="auto",
                               cmap="gray", rasterized=True)
            ax.axvline(0, color="0.6", lw=0.6)
            for c, sgn in (("M", -1), ("K", +1)):
                qq = np.linspace(1e-3, EDGE[c], 400)
                fd = fz["scale_Hz"] * np.sqrt(np.maximum([dispersion.lam(v, DIRS[c], fz["k"], br) for v in qq], 0.0))
                ax.plot(sgn * qq, fd, color=RED, lw=2.0, zorder=4, ls="-" if br == "L" else "--",
                        label="fitted dispersion" if c == "K" else None)
            pb = [((-1 if c == "M" else 1) * kk, fc) for b, c, kk, fc in fit["points"] if b == br]
            ax.plot(*zip(*pb), ls="none", marker="o", ms=2.2, mfc="w", mec=RED, mew=0.6, zorder=5,
                    label=f"ridge ({len(pb)})")
            ax.legend(loc="upper center", ncol=2, framealpha=0.85, handlelength=1.1, columnspacing=0.9,
                      handletextpad=0.4, borderpad=0.25)
            ax.set_xticks([-EDGE["M"], -2, -1, 0, 1, 2, 3, EDGE["K"]], ["M", "2", "1", "Γ", "1", "2", "3", "K"])
            ax.set(xlim=(-EDGE["M"], EDGE["K"]), ylim=(0, 1000), title=name)
            ax.set_xlabel("|k| a,  M ← Γ → K")
            ax.set_box_aspect(1)
            if br == "L":
                ax.set_ylabel("f (Hz)")
            else:
                ax.tick_params(labelleft=False)
                cbar = fig.colorbar(im, ax=ax, cax=ax.inset_axes([1.05, 0, 0.05, 1]))
                cbar.set_label("S / median over k")
        save_panel(fig, FIGURE, f"raft_{tag}", bbox_inches=None, dpi=300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rafts", nargs="+", default=["A", "B", "D"])
    fits = dispersion.fit_all()
    for tag in ap.parse_args().rafts:
        raft_row(tag, fits[tag])


if __name__ == "__main__":
    main()
