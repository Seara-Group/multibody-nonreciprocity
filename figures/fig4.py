"""Fig 4 -- levitated rafts have anomalously large Poisson ratios.

Panels written to OUT/figures/fig4/:
    c_speeds_poisson.pdf   longitudinal and transverse sound speeds, each divided by the raft's own
                           scale c0 = ell sqrt(k_rr/m), and the Poisson ratio, against
                           kappa = k_phiphi/k_rr from the COMSOL lattice Jacobians (circles: first
                           shell only; squares: with the second shell).  Rafts A, B, D from their
                           fitted dispersion; the two rafts that melted from phonon_data.csv.
    b_cut.pdf              velocity-flux spectra along q_y = 0 (longitudinal left, transverse right),
                           with the minimal-model dispersion: longitudinal solid, transverse dashed,
                           each panel's own branch at full strength; the band is the spread over
                           +-11.2 deg of lattice orientation (the raft's measured orientational
                           disorder) about the measured lattice direction (-2.72 deg)
    b_azimuthal.pdf        the same averaged around rings of constant |q|, band over all directions
(a is a drawing.)

Panel b: the model has one fitted number, the overall frequency scale (least squares against the
constant-frequency ridge, peak in q at fixed f, of both branches); every shape is a prediction.

Theory (nearest-neighbour spring lattice): c_L/c0 = sqrt((9 + 3k)/8), c_T/c0 = sqrt((3 + 9k)/8),
nu = (1 - 5k)/(3 + k); c_T = 0 (shear instability) at k = -1/3.
Error bars are ranges: horizontal over the COMSOL run and its _095/_105 variants; vertical over
the fits to the ridge-point subsets "all", "L + T(ka>1)", "T only".

    python fig4.py [--panel c]
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from levitation import comsol, config, dispersion, minimal_model, spectra  # noqa: E402
import cmasher as cmr  # noqa: E402
from levitation.plotting import plt, save_panel, save_derived  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

FIGURE = "fig4"
CRYSTALS = [("A", "41.1um_34700hz", "A  41 µm, 34.7 kHz"),     # raft, COMSOL run, label
            ("B", "41.1um_64500hz", "B  41 µm, 64.5 kHz"),
            ("D", "59.7um_64500hz", "D  60 µm, 64.4 kHz")]
MELTED = [(28.0, "28 µm, 34.8 kHz (melted)"), (25.0, "25 µm, 34.8 kHz (melted)")]
FIT_SUBSETS = ("all", "L + T(ka>1)", "T only")
THEORY = {"yL": lambda k: np.sqrt((9 + 3 * k) / 8),
          "yT": lambda k: np.sqrt(max((3 + 9 * k) / 8, 0.0)),
          "nu": lambda k: (1 - 5 * k) / (3 + k)}


def melted_rafts():
    """Rafts that melted at once: kappa from phonon_data.csv; c_T = 0 so nu = 1."""
    rows = list(csv.DictReader(open(os.path.join(config.DATA, "phonon_data.csv"))))
    out = []
    for size, label in MELTED:
        for r in rows:
            if r["file_name"] == "none" and abs(float(r["size"]) - size) < 0.05:
                ks = [float(r[c]) for c in ("k_phi/k_r_1", "k_phi/k_r_095", "k_phi/k_r_105")]
                cls = [float(r[f"c_l{i}"]) for i in (1, 2, 3) if r.get(f"c_l{i}")]
                cts = [float(r[f"c_t{i}"]) for i in (1, 2, 3) if r.get(f"c_t{i}")]
                cL = float(np.mean(cls)) if cls else np.nan
                cT = float(np.mean(cts)) if cts else np.nan
                out.append(dict(label=label, kb=ks[0], kb_lo=min(ks), kb_hi=max(ks),
                                yL=1.0 if cL == 1 else np.nan, yT=0.0 if cT == 0 else np.nan,
                                nu=1.0 if cT == 0 else 1 - 2 * (cT / cL) ** 2))
    return out


def crystal_records(fits):
    recs = []
    for raft, run, label in CRYSTALS:
        f = fits[raft]["fits"]
        k = f["all"]["k"]
        ksub = [f[s]["k"] for s in FIT_SUBSETS if f.get(s)]
        rec = dict(label=label, n=f["all"]["n"], **comsol.kappa_with_spread(run))
        for key, fn in THEORY.items():
            rec[key] = fn(k)
            vals = [fn(x) for x in ksub]
            rec[key + "_rng"] = (min(vals), max(vals))
        sp = fits[raft]["speeds"]
        print(f"{label:20s} kappa = {k:+.3f}  nu = {rec['nu']:.3f}  c_L = {sp['cL']:.0f} mm/s  c_T = {sp['cT']:.0f} mm/s  "
              f"(ell = {fits[raft]['ell_um']:.2f} um);  COMSOL kappa bond {rec['kb']:+.3f} "
              f"[{rec['kb_lo']:+.3f}, {rec['kb_hi']:+.3f}], effective {rec['ke']:+.3f} [{rec['ke_lo']:+.3f}, {rec['ke_hi']:+.3f}]")
        recs.append(rec)
    return recs


def panel_c(fits):
    recs, melted = crystal_records(fits), melted_rafts()
    for i, r in enumerate(recs + melted):
        r["color"] = f"C{i}"
    EB = dict(ls="none", elinewidth=1, capsize=2)
    xbar = lambda r, kk: [[r[kk] - r[kk + "_lo"]], [r[kk + "_hi"] - r[kk]]]

    def ybar(r, key):
        if key + "_rng" not in r:
            return None
        lo, hi = r[key + "_rng"]
        return [[max(r[key] - lo, 0)], [max(hi - r[key], 0)]]

    kk = np.linspace(-0.5, 0, 400)
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.5), dpi=144, constrained_layout=True)
    for ax, key, curve, label, ymax in (
            (axs[0], "yL", np.sqrt((9 + 3 * kk) / 8), r"$c_L/c_0=\sqrt{(9+3k)/8}$", 1.25),
            (axs[1], "yT", np.sqrt(np.clip((3 + 9 * kk) / 8, 0, None)), r"$c_T/c_0=\sqrt{(3+9k)/8}$", 0.5)):
        ax.plot(kk, curve, color="k", lw=2, zorder=1)
        ax.axvspan(-0.5, -1 / 3, color="FireBrick", zorder=-1, alpha=0.2)
        for kkey, marker, alpha in (("kb", "o", 0.8), ("ke", "s", 1.0)):
            for r in recs:
                ax.errorbar(r[kkey], r[key], xerr=xbar(r, kkey), yerr=ybar(r, key), marker=marker,
                            color=r["color"], mec="k", alpha=alpha, **EB)
        for r in melted:
            if np.isfinite(r[key]):
                ax.errorbar(r["kb"], r[key], xerr=xbar(r, "kb"), marker="o", color=r["color"], mec="k",
                            alpha=0.8, clip_on=False, zorder=4, **EB)
        ax.text(0.02, 0.02, "shear\nunstable", transform=ax.transAxes, color="FireBrick", fontsize=8)
        ax.text(0.95, 0.05, label, transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
        ax.set_ylabel(rf"$c_{key[1]}/c_0$ (measured)")
        ax.set_ylim(0, ymax)
    axs[0].set(yticks=[0, 0.5, 1.0])
    ax = axs[2]
    ax.plot(kk, (1 - 5 * kk) / (3 + kk), color="k", lw=2)
    ax.text(-0.48, 1 / 3 + 0.01, r"$\nu=1/3$", ha="left", va="bottom", fontsize=7, color="0.4")
    ax.axhline(1 / 3, ls="--", color="0.4", zorder=-1)
    ax.text(-1 / 3 - 0.003, 1.02, r"$\mu = 0$", ha="right", va="bottom", fontsize=8, color="0.35")
    ax.axhspan(1, 1.1, color="FireBrick", zorder=-1, alpha=0.2)
    ax.text(0.95, 0.05, r"$\nu=(1-5k)/(3+k)$", ha="right", va="bottom", transform=ax.transAxes, fontsize=8)
    for kkey, marker, alpha in (("kb", "o", 0.8), ("ke", "s", 1.0)):
        for r in recs:
            ax.errorbar(r[kkey], r["nu"], xerr=xbar(r, kkey), yerr=ybar(r, "nu"), marker=marker,
                        color=r["color"], mec="k", alpha=alpha, **EB)
    for r in melted:
        if np.isfinite(r["nu"]):
            ax.errorbar(r["kb"], r["nu"], xerr=xbar(r, "kb"), marker="o", color=r["color"], mec="k",
                        alpha=0.8, clip_on=False, zorder=4, **EB)
    ax.set(ylim=(0.2, 1.1), ylabel=r"Poisson ratio $\nu$ (measured)")
    for a in axs:
        a.set_xlim(-0.5, 0)
        a.set_xticks([0, -1 / 3, -0.5], ["0", "-1/3", "-1/2"])
        a.set_xlabel(r"$k_\phi/k_r$ (COMSOL)")
        a.set_box_aspect(1)
    handles = [Line2D([], [], ls="none", marker="o", ms=6, color=r["color"], label=r["label"]) for r in recs + melted]
    handles += [Line2D([], [], ls="none", marker="o", ms=6, color="0.45", label="COMSOL, nearest neighbours only"),
                Line2D([], [], ls="none", marker="s", ms=6, color="0.45", label="COMSOL, with 2nd neighbours")]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, fontsize=6.5, frameon=False)
    save_panel(fig, FIGURE, "c_speeds_poisson", bbox_inches=None, dpi=200)
    return {f"c_{r['label'].split()[0]}_{k}": r[k] for r in recs for k in ("yL", "yT", "nu", "kb", "ke")}


THETA0_DEG, SPREAD_DEG = -2.72, 11.2   # lab x relative to the lattice; spread of arg(psi6)/6 (interior)


def ridge(d, cut, fmin=dict(L=190.0, T=130.0), fmax=700.0, contrast=2.0, dfbin=18.0):
    """Constant-frequency ridge (peak in q ell at fixed f) of both branches: {"L": (n, 2), "T": ...}."""
    a = float(d["a_lat"])
    out = {}
    if cut:
        kx, f = d["kx_cut"], d["f_cut"]
        z = int(np.argmin(abs(f)))
        n = min(z, len(f) - z - 1)
        IP, IM = np.arange(z + 1, z + 1 + n), z - np.arange(1, 1 + n)
        fp, kq = f[IP], (kx * a > 0.15) & (kx * a < 4.15)
        qa = kx[kq] * a
        for br in ("L", "T"):
            S = 0.5 * (d[f"S{br}_cut"][:, IP] + d[f"S{br}_cut"][:, IM])     # +f and -f averaged
            out[br] = _peaks(S, fp, qa, kq, fmin[br], fmax, contrast, dfbin)
    else:
        kq = (d["k_az"] * a > 0.15) & (d["k_az"] * a < 4.15)
        for br in ("L", "T"):
            out[br] = _peaks(d[f"S{br}_az"], d["f_az"], d["k_az"][kq] * a, kq, fmin[br], fmax, contrast, dfbin)
    return out


def _peaks(S, f, qa, kq, fmin, fmax, contrast, dfbin):
    rows = []
    for lo in np.arange(fmin, fmax, dfbin):
        j = (f >= lo) & (f < lo + dfbin)
        if j.sum() < 3:
            continue
        p = np.convolve(S[:, j].mean(1)[kq], np.ones(3) / 3, mode="same")
        if p.max() / np.median(p) >= contrast:
            rows.append((qa[int(np.argmax(p))], lo + dfbin / 2))
    return np.array(rows) if rows else np.empty((0, 2))


def panel_b(d):
    a = float(d["a_lat"])
    qa = np.linspace(1e-3, 4.2, 90)
    res = {}
    for cut in (True, False):
        dirs = (np.radians(THETA0_DEG) + np.linspace(-1, 1, 16) * np.radians(SPREAD_DEG)) if cut else None
        mean, lo, hi = minimal_model.lattice_branches(qa, directions=dirs)
        r = ridge(d, cut)
        num = den = 0.0
        for key, br in (("L", 1), ("T", 0)):
            if len(r[key]):
                m = np.interp(r[key][:, 0], qa, mean[:, br])
                w = 1.0 / r[key][:, 1].mean() ** 2
                num += w * np.sum(m * r[key][:, 1])
                den += w * np.sum(m * m)
        scale = num / den
        small = qa < 0.3
        cT, cL = (np.polyfit(qa[small], mean[small, i], 1)[0] for i in (0, 1))
        tag = "cut" if cut else "azimuthal"
        print(f"panel b ({tag}): frequency scale {scale:.4e} Hz per model unit from {len(r['L'])} + {len(r['T'])} "
              f"ridge points; model c_L = {2 * np.pi * scale * cL * a * 1e-6:.4f} m/s, "
              f"c_T = {2 * np.pi * scale * cT * a * 1e-6:.4f} m/s, nu = {1 - 2 / (cL / cT) ** 2:.4f}")
        res[f"b_{tag}_scale"] = scale

        fig, axs = plt.subplots(1, 2, figsize=(6, 3), dpi=144, constrained_layout=True, sharey=True)
        for i, (br_name, col) in enumerate((("L", 1), ("T", 0))):
            ax = axs[i]
            if cut:
                k, f = d["kx_cut"] * a, d["f_cut"]
                m = np.abs(f) < 1000
            else:
                k, f = d["k_az"] * a, d["f_az"]
                m = (f > 2) & (f < 1000)
            S = np.log10(d[f"S{br_name}_{'cut' if cut else 'az'}"][:, m] + 1e-30)
            ax.pcolormesh(k, f[m], S.T, cmap=cmr.neutral, shading="auto", rasterized=True,
                          vmin=np.percentile(S, 20), vmax=np.percentile(S, 99.5))
            signs = (+1, -1) if cut else (+1,)
            for c2, ls in ((1, "-"), (0, "--")):              # longitudinal solid, transverse dashed
                alpha = 1.0 if c2 == col else 0.35
                for s in signs:
                    for xs in ((qa, -qa) if cut else (qa,)):
                        ax.fill_between(xs, s * lo[:, c2] * scale, s * hi[:, c2] * scale, color="FireBrick",
                                        alpha=0.2 * alpha, lw=0, zorder=3)
                        ax.plot(xs, s * mean[:, c2] * scale, ls, color="FireBrick", lw=2, alpha=alpha, zorder=4)
            for sk in signs:                                   # unresolved: finite raft, dq ell = 0.29
                ax.axvspan(0 if sk > 0 else -0.286, 0.286 if sk > 0 else 0, color="0.5", alpha=.18, lw=0, zorder=1)
            ax.set(xlim=(-4.0 if cut else 0, 4.0), ylim=(-1000 if cut else 0, 1000),
                   xlabel=r"$q_x\ell$" if cut else r"$q\ell$", title="longitudinal" if col else "transverse")
            ax.set_box_aspect(1)
        axs[0].set_ylabel("f (Hz)")
        save_panel(fig, FIGURE, f"b_{tag}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", nargs="+", default=["b", "c"])
    a = ap.parse_args()
    out = {}
    if "b" in a.panel:
        out.update(panel_b(spectra.raft_a_spectra()))
    if "c" in a.panel:
        out.update(panel_c(dispersion.fit_all()))
    save_derived(FIGURE, out)


if __name__ == "__main__":
    main()
