"""Fig 5 -- multi-body nonreciprocity emerges at defects and edges.

Panels written to OUT/figures/fig5/:
    a_perfect.pdf, a_dislocated.pdf   per-particle ||D-_i||/||D+_i|| of the minimal model on a
                                      free disc (radius 16 lattice spacings), without and with
                                      one edge dislocation at the centre
    b_vs_strain.pdf                   the dislocated disc's ratio against the local shear
                                      strain, with the uniform-shear prediction (dashed)
    c_experiment.pdf, c_model.pdf     the ratio on raft A: from the dynamical matrix inferred
                                      from the trajectories, and from the minimal model at the
                                      measured mean positions, both seen through the same 40 modes
    d_exp_vs_model.pdf                the two maps of (c) against each other
    e_vs_radius.pdf                   both against distance from the raft centroid
    f_vs_psi6.pdf                     both against 1 - |psi_6|

||D_i|| is the Frobenius norm of particle i's rows of D (inference.per_particle).

Row 1 (a, b).  The minimal model's D is the finite-difference derivative of its forces
(levitation.minimal_model), coefficients COEFFS_54 (Fig 3).  The discs are relaxed under unit
springs with the outer 1.5 spacings held fixed.  "Interior" means more than PAD = 2.5 spacings
inside the outermost particle.  Panel b uses six-coordinated interior particles more than 3
spacings from the dislocation core; the local shear strain |eps_dev| is the traceless part of
the neighbour-shell tensor <d d>/a^2, and the dashed line is ||D-||/||D+|| per unit strain of the
same lattice under uniform pure shear (not a fit to the points).

Row 2 (c-f).  D_exp is inferred from raft A's trajectories on its 40 highest-variance modes V
(levitation.inference).  The model is evaluated at the measured mean positions and pushed
through the same modes, D -> V V^T D V V^T, so that the two maps have the same resolution.
Each map is divided by ||D+||/sqrt(N) of its own matrix, so the RMS of the map over particles
is that matrix's raft-level ||D-||/||D+||.  The grey band in (e, f) is the mean over 40
reciprocal surrogates (the noise floor of the measurement).

Defects (pentagons: 5-fold, heptagons: 7-fold) are Delaunay coordinations, shown only for
particles whose nearest-neighbour shell closes (lattice.closed_shell); particles whose shell
does not close are the free edge (grey points in e, f).  Distances are in units of the lattice
spacing ell: for raft A its raft-average spacing (median Delaunay bond of the mean positions,
54.01 um), for the model discs the model's reference spacing a = 54.01 um.

    python fig5.py [--panel a b c d e f] [--n-surrogates 40]

Printed: the numbers quoted in the text (raft-level ratios, the exp-model correlation, the
noise floor and its control), and the slope of the dislocated disc's interior ratio against
alpha3 (1 means the ratio is proportional to alpha3).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from levitation import config, inference, lattice  # noqa: E402
from levitation import minimal_model as mm  # noqa: E402
from levitation.plotting import loess, plt, save_derived, save_panel  # noqa: E402
import matplotlib as mpl  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import RegularPolygon  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402
import cmasher as cmr  # noqa: E402

FIGURE = "fig5"
C = mm.COEFFS_54
A_UM = C["a"]              # model lattice spacing, um
R_MAP = 16                 # model disc radius, lattice spacings
PAD = 2.5                  # interior: r < r_max - PAD a
N_MODES = 40
N_SURR = 40

mpl.rcParams.update({"figure.dpi": 144, "savefig.dpi": 300, "axes.titlelocation": "left",
                     "axes.linewidth": 0.6, "xtick.direction": "out", "ytick.direction": "out",
                     "legend.frameon": False})
FLOOR_CLAMP = 1e-5
C_MEAS = mpl.colormaps["Blues"](0.90)[:3]
C_MODL = mpl.colormaps["Blues"](0.70)[:3]
LS_MEAS, LS_MODL = "-", (0, (4.5, 1.8))
C_FIVE, C_SEVEN = "magenta", "green"
R_POLY = 0.62              # defect marker radius, lattice units


# ================================================================== row 1: model discs
def interior_mask(pos, a=A_UM):
    rr = np.linalg.norm(pos, axis=1)
    return rr < rr.max() - PAD * a


def coordination(pos, a):
    """Number of nearest-neighbour bonds (Delaunay edges shorter than 1.45 a)."""
    return np.bincount(mm.short_bonds(pos, mm.RC_FACTOR * a).ravel(), minlength=len(pos))


def shell_strain(pos, a):
    """|eps_dev|: magnitude of the traceless part of the neighbour-shell tensor <d d>/a^2
    (a perfect six-fold shell gives I/2; uniform pure shear gamma gives gamma)."""
    t = cKDTree(pos)
    out = np.zeros(len(pos))
    for i, q in enumerate(pos):
        nb = [k for k in t.query_ball_point(q, mm.RC_FACTOR * a) if k != i]
        if len(nb) < 3:
            continue
        v = (pos[nb] - q) / a
        Sm = (v[:, :, None] * v[:, None, :]).mean(0)
        Dm = Sm - np.trace(Sm) / 2 * np.eye(2)
        out[i] = float(np.hypot(Dm[0, 0], Dm[0, 1]))
    return out


def ratio_map(D):
    return inference.per_particle(inference.anti(D)) / inference.per_particle(inference.sym(D))


def interior_ratio(D, inn):
    return float(np.linalg.norm(inference.per_particle(inference.anti(D))[inn])
                 / np.linalg.norm(inference.per_particle(inference.sym(D))[inn]))


def shear_coefficient(R=8, gammas=(0.002, 0.005, 0.01, 0.02)):
    """Interior ||D-||/||D+|| per unit strain of a perfect lattice under uniform pure shear."""
    G = np.array([[1.0, 0.0], [0.0, -1.0]])
    vals = []
    for g in gammas:
        lat = mm.tri_disk(R * A_UM, A_UM) @ (np.eye(2) + g * G).T
        vals.append(interior_ratio(mm.raft_dynamical_matrix(lat, C), interior_mask(lat)) / g)
    return float(np.median(vals))


def compute_model():
    d = {}
    for key, disl in (("perfect", False), ("dislocated", True)):
        pos = mm.build_raft(R_MAP, A_UM, disl)
        D0 = mm.raft_dynamical_matrix(pos, dict(C, alpha3=0.0))
        D1 = mm.raft_dynamical_matrix(pos, dict(C, alpha3=1.0)) - D0      # D is linear in alpha3
        D = mm.raft_dynamical_matrix(pos, C)
        inn = interior_mask(pos)
        alphas = C["k_a"] * np.logspace(-4, -1, 13)
        d[key] = dict(pos=pos, ratio=ratio_map(D), interior=inn, glob=interior_ratio(D, inn),
                      sweep=np.array([interior_ratio(D0 + al * D1, inn) for al in alphas]))
    pos = d["dislocated"]["pos"]
    co = coordination(pos, A_UM)
    rr = np.linalg.norm(pos, axis=1)
    core = pos[(co != 6) & (rr < rr.max() - 3.0 * A_UM)].mean(0)
    eps = shell_strain(pos, A_UM)
    rat = d["dislocated"]["ratio"]
    valid = (d["dislocated"]["interior"] & (eps > 0) & (co == 6)
             & (np.linalg.norm(pos - core, axis=1) / A_UM > 3.0))
    d["strain"] = dict(eps=eps, valid=valid, C_shear=shear_coefficient(),
                       C_raft=float(np.median(rat[valid] / eps[valid])),
                       r=float(pearsonr(rat[valid], eps[valid])[0]))
    d["alpha3_slope"] = float(np.polyfit(np.log10(alphas), np.log10(d["dislocated"]["sweep"]), 1)[0])
    return d


# ================================================================== row 2: raft A
def compute_experiment(n_surr=N_SURR, n_modes=N_MODES):
    U, pos = inference.raft_a_displacements()
    Q, _ = inference.mode_basis(U)
    V = Q[:, :n_modes]
    a = U @ V
    del U
    D_exp = inference.mode_dynamical_matrix(a)
    D_mod = inference.project(mm.raft_dynamical_matrix(pos, C), V)
    A_exp = inference.per_particle(inference.anti(V @ D_exp @ V.T))
    A_mod = inference.per_particle(inference.anti(D_mod))
    L = lattice.delaunay_lattice(pos)
    ell = lattice.median_bond(pos)             # raft-average spacing, 54.01 um
    closed = lattice.closed_shell(pos, ell)
    ex = dict(pos=pos, n_modes=n_modes,
              ell=ell, radius=np.linalg.norm(pos - pos.mean(0), axis=1) / ell,
              psi6=np.abs(lattice.psi6(pos, L.neighbors)),
              closed=closed, five=closed & (L.coordination == 5), seven=closed & (L.coordination == 7),
              A_exp=A_exp, A_mod=A_mod,
              nrm_exp=float(np.linalg.norm(inference.sym(D_exp)) / np.sqrt(len(pos))),
              nrm_mod=float(np.linalg.norm(inference.sym(D_mod)) / np.sqrt(len(pos))),
              ratio_exp=inference.ratio(D_exp), ratio_mod=inference.ratio(D_mod),
              r_pair=float(pearsonr(A_exp, A_mod)[0]), rs_pair=float(spearmanr(A_exp, A_mod)[0]))
    ex["floor"] = inference.null_and_control(a, ex["ratio_exp"], n_surr) if n_surr else None
    return ex


# ================================================================== drawing
def _defects(ax, pos, five, seven, a):
    for mask, nv, col in ((five, 5, C_FIVE), (seven, 7, C_SEVEN)):
        for p in pos[mask] / a:
            ax.add_patch(RegularPolygon(tuple(p), numVertices=nv, radius=R_POLY, orientation=0.0,
                                        facecolor=col, edgecolor="none", zorder=1.6))


def _map(ax, pos, val, a, title, five, seven, vmin=FLOOR_CLAMP, vmax=1e-2,
         label=r"$\|A_i\|/\|S_i\|$", cmap="Blues"):
    _defects(ax, pos, five, seven, a)
    im = ax.scatter(pos[:, 0] / a, pos[:, 1] / a, c=np.clip(val, vmin, None), s=5.5,
                    norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cmap, linewidths=0, zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left")
    cb = plt.colorbar(im, ax=ax, cax=ax.inset_axes([1.05, 0, 0.05, 1]))
    cb.set_label(label, labelpad=2)


def _model_defects(pos):
    closed = lattice.closed_shell(pos, A_UM)
    co = lattice.delaunay_lattice(pos).coordination
    return closed & (co == 5), closed & (co == 7)


def panel_a(md):
    for key, title in (("perfect", "Perfect lattice"), ("dislocated", "dislocation")):
        m = md[key]
        fig, ax = plt.subplots(figsize=(3, 3))
        _map(ax, m["pos"], m["ratio"], A_UM, title, *_model_defects(m["pos"]))
        save_panel(fig, FIGURE, f"a_{key}", dpi=300)


def panel_b(md):
    m, s = md["dislocated"], md["strain"]
    v = s["valid"]
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.loglog(s["eps"][v], m["ratio"][v], ".", color="SteelBlue", alpha=0.5,
              label="six-coordinated, $r>3a$")
    x = np.array([s["eps"][v].min() * 0.5, s["eps"][v].max() * 1.5])
    ax.loglog(x, s["C_shear"] * x, "--", lw=2, color="k", zorder=4, label="uniform-shear calibration")
    ax.text(0.05, 0.95, f"$r = {s['r']:+.2f}$\n", transform=ax.transAxes, va="top", ha="left",
            fontsize=plt.rcParams["legend.fontsize"])
    ax.set(xlim=[x.min(), x.max()], ylim=[s["C_shear"] * x.min(), s["C_shear"] * x.max()],
           xlabel=r"local deviatoric strain  $|\varepsilon_{\rm dev}|$", ylabel=r"$\|A_i\|/\|S_i\|$")
    ax.set_box_aspect(1)
    ax.legend(loc="lower right", frameon=False, handlelength=1.5, borderaxespad=0.15, labelspacing=0.25)
    save_panel(fig, FIGURE, "b_vs_strain", dpi=300)


def panel_c(ex):
    """Each map has its own colour bar (the two differ ~140x in magnitude) and is divided by its
    own ||D+||/sqrt(N), so its RMS is the raft-level ratio printed under it."""
    for which, key, nk, title, rk in (("experiment", "A_exp", "nrm_exp", "Measured: inferred from trajectories", "ratio_exp"),
                                      ("model", "A_mod", "nrm_mod", "Predicted: minimal model, same basis", "ratio_mod")):
        val = ex[key] / ex[nk]
        assert abs(np.sqrt((val ** 2).mean()) / ex[rk] - 1) < 1e-9, "RMS != raft ratio"
        lo, hi = np.percentile(val, [1, 99])
        fig, ax = plt.subplots(figsize=(3, 3))
        _map(ax, ex["pos"], val, ex["ell"], title, ex["five"], ex["seven"], vmin=lo, vmax=hi,
             label=r"local $\|A\|/\|S\|$")
        ax.text(0.0, -0.04, rf"raft $\|A\|/\|S\| = {ex[rk]:.4f}$", transform=ax.transAxes,
                ha="left", va="top", fontsize=plt.rcParams["legend.fontsize"])
        save_panel(fig, FIGURE, f"c_{which}", dpi=300)


def _thin_log_ticks(ax, which="xy"):
    """Label only 1, 2, 5 x 10^k on a log axis spanning less than a couple of decades."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    def fmt(v, _):
        k = int(np.floor(np.log10(v) + 1e-9))
        m = v / 10.0 ** k
        return rf"$10^{{{k}}}$" if abs(m - 1) < 1e-6 else rf"${m:g}\times10^{{{k}}}$"

    for nm in which:
        a, (lo, hi) = (ax.xaxis, ax.get_xlim()) if nm == "x" else (ax.yaxis, ax.get_ylim())
        a.set_minor_formatter(NullFormatter())
        a.set_major_locator(FixedLocator([m * 10.0 ** k for k in range(-12, 4) for m in (1, 2, 5)
                                          if lo <= m * 10.0 ** k <= hi]))
        a.set_major_formatter(FuncFormatter(fmt))


def panel_d(ex):
    """The dashed line is proportional (slope = ratio of the two raft-level values), not 1:1."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    x, y = ex["A_mod"] / ex["nrm_mod"], ex["A_exp"] / ex["nrm_exp"]
    fig, ax = plt.subplots(figsize=(3, 3))
    sc = ax.scatter(x, y, c=ex["radius"], s=10, cmap=cmr.ember, linewidths=0, alpha=0.85, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    fac = ex["ratio_exp"] / ex["ratio_mod"]
    gx = np.array([x.min() / 1.3, x.max() * 1.3])
    ax.plot(gx, fac * gx, "--", color="#0b0b0b", lw=0.8, zorder=2)
    ax.text(0.96, 0.08, rf"proportional, $\times{fac:.0f}$", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=plt.rcParams["legend.fontsize"])
    ax.set_xlim(*gx)
    ax.set_ylim(fac * gx[0], fac * gx[1])
    ax.text(0.04, 0.96, f"$r = {ex['r_pair']:+.2f}$", transform=ax.transAxes, ha="left", va="top",
            fontsize=plt.rcParams["legend.fontsize"])
    ax.set_box_aspect(1)
    cax = make_axes_locatable(ax).append_axes("right", size="4.5%", pad=0.08, axes_class=mpl.axes.Axes)
    cb = plt.colorbar(sc, cax=cax)
    cb.set_label(r"distance from raft centroid  ($\ell$)", labelpad=2)
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(length=2)
    _thin_log_ticks(ax, "xy")
    ax.set(xlabel=r"predicted local $\|A\|/\|S\|$", ylabel=r"measured local $\|A\|/\|S\|$")
    ax.set_title("Model predicts the measured map", loc="left")
    save_panel(fig, FIGURE, "d_exp_vs_model", dpi=300)


def _xvals(ex, xkey):
    return ex["radius"] if xkey == "radius" else 1.0 - ex["psi6"]


def _loess(x, y, logx, frac):
    return loess(np.log10(x), np.log10(y), frac=frac) if logx else loess(x, np.log10(y), frac=frac)


def profile_ylims(ex, frac=0.4):
    """Y-limits shared by panels e and f: from the union of the LOESS bands over both abscissae,
    the measured range widened to keep the noise floor in view, and both axes given the same
    number of decades (equal heights = equal fractional change)."""
    spans = {}
    for key, nk in (("A_exp", "nrm_exp"), ("A_mod", "nrm_mod")):
        y = ex[key] / ex[nk]
        los, his = [], []
        for xkey, logx in (("radius", False), ("psi6", True)):
            _, gy, gs = _loess(_xvals(ex, xkey), y, logx, frac)
            ok = np.isfinite(gy)
            los.append(float((10.0 ** (gy - gs)[ok]).min()))
            his.append(float((10.0 ** (gy + gs)[ok]).max()))
        spans[key] = [min(los), max(his)]
    if ex["floor"] is not None:
        spans["A_exp"][0] = min(spans["A_exp"][0], 0.92 * ex["floor"]["mean"])
    dec = max(hi / lo for lo, hi in spans.values()) * 1.6
    return {k: (np.sqrt(lo * hi) / np.sqrt(dec), np.sqrt(lo * hi) * np.sqrt(dec)) for k, (lo, hi) in spans.items()}


def _profile(ex, xkey, xlabel, title, name, logx, frac=0.4):
    """Measured (solid, left axis, with its points) and model (dashed, right axis); LOESS
    trends in log10(y) with +-1 local sd bands.  Free-edge particles are drawn grey (still
    included in the trends)."""
    xx = _xvals(ex, xkey)
    edge = ~ex["closed"]
    lims = profile_ylims(ex, frac)
    fig, ax = plt.subplots(figsize=(3, 3))
    axr = ax.twinx()
    handles = []
    for key, nk, col, lab, axn, ls, cloud in (("A_exp", "nrm_exp", C_MEAS, "measured", ax, LS_MEAS, True),
                                              ("A_mod", "nrm_mod", C_MODL, "minimal model", axr, LS_MODL, False)):
        y = ex[key] / ex[nk]
        if cloud:
            axn.plot(xx[~edge], y[~edge], ".", ms=5, color=col, alpha=0.18, zorder=1)
            axn.plot(xx[edge], y[edge], ".", ms=5, color="0.55", alpha=0.30, zorder=1)
        gx, gy, gs = _loess(xx, y, logx, frac)
        if logx:
            gx = 10.0 ** gx
        ok = np.isfinite(gy)
        axn.fill_between(gx[ok], 10.0 ** (gy - gs)[ok], 10.0 ** (gy + gs)[ok], color=col, alpha=0.15, lw=0, zorder=2)
        handles.append(axn.plot(gx[ok], 10.0 ** gy[ok], ls=ls, color=col, lw=2.2, zorder=3, label=lab)[0])
        axn.set_yscale("log")
        axn.set_ylim(*lims[key])
        axn.tick_params(axis="y", which="both", colors=col)
    ax.spines["left"].set_color(C_MEAS)
    axr.spines["right"].set_color(C_MODL)
    if ex["floor"] is not None:
        ax.axhspan(1e-12, ex["floor"]["mean"], color="0.88", zorder=0, lw=0)
        ax.text(0.02, ex["floor"]["mean"], " reciprocal surrogate null", transform=ax.get_yaxis_transform(),
                ha="left", va="top", color="0.45", fontsize=plt.rcParams["legend.fontsize"])
    handles.append(plt.Line2D([], [], ls="none", marker=".", ms=7, color="0.55", label="free edge"))
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"measured local $\|A\|/\|S\|$", color=C_MEAS)
    axr.set_ylabel(r"model local $\|A\|/\|S\|$", color=C_MODL)
    ax.margins(x=0.04)
    ax.set_box_aspect(1)
    axr.set_box_aspect(1)
    ax.legend(handles=handles, loc="upper left", frameon=False, handlelength=1.4, borderaxespad=0.15,
              labelspacing=0.25)
    ax.set_title(title, loc="left")
    save_panel(fig, FIGURE, name, dpi=300)


def panel_e(ex):
    _profile(ex, "radius", r"distance from raft centroid  ($\ell$)", "Activity rises towards the boundary",
             "e_vs_radius", logx=False)


def panel_f(ex):
    _profile(ex, "psi6", r"$1-|\psi_6|$", "Activity rises with local disorder", "f_vs_psi6", logx=True)


# ================================================================== main
def report(md, ex):
    if md is not None:
        s = md["strain"]
        print(f"model discs: interior ||D-||/||D+|| perfect {md['perfect']['glob']:.2e}, "
              f"dislocated {md['dislocated']['glob']:.2e}")
        print(f"  vs shear strain: r = {s['r']:+.3f}, median ratio/strain {s['C_raft']:.2e} "
              f"vs uniform-shear prediction {s['C_shear']:.2e}")
        print(f"  dislocated interior ratio vs alpha3: log-log slope {md['alpha3_slope']:.3f}")
    if ex is not None:
        print(f"raft A ({ex['n_modes']} modes): ||D-||/||D+|| measured {ex['ratio_exp']:.4f}, model {ex['ratio_mod']:.4f}; "
              f"per-particle r = {ex['r_pair']:+.3f} (Spearman {ex['rs_pair']:+.3f})")
        print(f"  defects (closed shells): {int(ex['five'].sum())} 5-fold, {int(ex['seven'].sum())} 7-fold; "
              f"{int((~ex['closed']).sum())} free-edge particles")
        fl = ex["floor"]
        if fl is not None:
            print(f"  noise floor {fl['mean']:.4f} +/- {fl['sd']:.4f} (95% {fl['lo']:.4f}-{fl['hi']:.4f}, "
                  f"{fl['n_seg']} segments, lost {fl['lost']:.3g}); control "
                  f"{'passes' if fl['control_passes'] else 'FAILS'}: {fl['control_mean']:.4f} +/- {fl['control_sd']:.4f} "
                  f"({len(fl['control'])} realizations)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", nargs="+", default=list("abcdef"))
    ap.add_argument("--n-surrogates", type=int, default=N_SURR)
    args = ap.parse_args()
    md = compute_model() if set(args.panel) & set("ab") else None
    ex = compute_experiment(args.n_surrogates) if set(args.panel) & set("cdef") else None
    report(md, ex)
    out = {}
    if md is not None:
        out.update({f"{k}_ratio": md[k]["ratio"] for k in ("perfect", "dislocated")})
        out.update(model_glob=np.array([md["perfect"]["glob"], md["dislocated"]["glob"]]),
                   strain=md["strain"]["eps"], strain_valid=md["strain"]["valid"],
                   strain_stats=np.array([md["strain"]["r"], md["strain"]["C_raft"], md["strain"]["C_shear"]]),
                   alpha3_slope=md["alpha3_slope"])
    if ex is not None:
        out.update({k: ex[k] for k in ("A_exp", "A_mod", "radius", "psi6", "closed", "five", "seven")})
        out.update(exp_stats=np.array([ex["ratio_exp"], ex["ratio_mod"], ex["r_pair"], ex["rs_pair"]]))
        if ex["floor"] is not None:
            out.update(null=ex["floor"]["null"], control=ex["floor"]["control"])
    save_derived(FIGURE, out)
    for p in args.panel:
        {"a": lambda: panel_a(md), "b": lambda: panel_b(md), "c": lambda: panel_c(ex),
         "d": lambda: panel_d(ex), "e": lambda: panel_e(ex), "f": lambda: panel_f(ex)}[p]()


if __name__ == "__main__":
    main()
