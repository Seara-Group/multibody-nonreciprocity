"""Fig 3 -- the minimal model's coefficients, measured from COMSOL (panel a is drawn by hand).

Panels written to OUT/figures/fig3/:
    b_isosceles.pdf    pinned-leg isosceles trimers (legs 79.57 um, base r varied; the network's
                       COMSOL training set): force along the apex direction on the apex (blue) and
                       on each base particle (red), with the minimal model whose two slopes are
                       fitted inside |r/leg - 1| <= 0.2 (grey band).  With the legs pinned k_a drops
                       out: F_apex = -(a3 - alpha3) u, F_base = (a3 + alpha3) u / 2,
                       u = leg (x - 1) sqrt(4 - x^2), x = r / leg.
    c_equilateral.pdf  equilateral trimers of the 950-configuration phase scan, the data the quoted
                       coefficients come from: outward radial force per particle (bars: sd over the
                       three particles) against side / r_ss, and the minimal model's tangent at the
                       reference spacing 54.01 um, F_r = -sqrt(3) [a0 + (k_a + 2 a3)(L - 54.01)].

Forces are in units of F0 = E0 d^6 q^4 (levitation.trimers).  r_ss is each dataset's own
force-free equilateral side (where the equilateral curve crosses zero).  No neural network.

Printed: the coefficients at 54.01 um with uncertainties and the ratios quoted in the main text
(levitation.trimers.coefficients); a check that they equal minimal_model.COEFFS_54, which Figs 4b
and 5 use.

    python fig3.py [--panel b c]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from levitation import minimal_model as mm, trimers as T  # noqa: E402
from levitation.plotting import plt, save_derived, save_panel  # noqa: E402

FIGURE = "fig3"
WINDOW = 0.20
APEX, BASE, EQC, MUT = "SteelBlue", "FireBrick", "Purple", "0.5"


def _style(ax):
    ax.axhline(0, color=MUT, lw=1, zorder=0)
    ax.set_box_aspect(1)
    ax.tick_params(direction="out")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)


def panel_b():
    leg, x, Fy = T.isosceles_family()
    Fy = Fy / T.F0
    a3, al, ea3, eal, n = T.isosceles_slopes(x, Fy, leg, WINDOW)
    P, F = T.load_training_set()
    s, a0, _ = T.equilateral_curve(P, F, max_asym=0.002)
    rss = T.force_free_side(s, a0)
    print(f"panel b: {len(x)} COMSOL configurations, legs {leg:.2f} um (training-set force-free "
          f"equilateral side r_ss = {rss:.2f} um); fit in |x-1| <= {WINDOW} ({n} configurations): "
          f"a3 = {a3:+.4g} +- {ea3:.2g}, alpha3 = {al:+.4g} +- {eal:.2g} (F0/um), alpha3/a3 = {al / a3:+.3f}")
    X = np.linspace(x.min(), x.max(), 300)
    g = leg * (X - 1) * np.sqrt(np.clip(4 - X ** 2, 0, None))
    fig, ax = plt.subplots(figsize=(3, 3), dpi=144)
    _style(ax)
    ax.axvspan(1 - WINDOW, 1 + WINDOW, color="0.92", zorder=0)
    ax.axvline(1, color=MUT, lw=1, zorder=0)
    ax.plot(X, -(a3 - al) * g, color=APEX, lw=1.2, label="model", zorder=3)
    ax.plot(X, 0.5 * (a3 + al) * g, color=BASE, lw=1.2, zorder=3)
    ax.plot(x, Fy[:, 0], "o", ms=3.5, mfc="w", mec=APEX, label="COMSOL", zorder=4)
    ax.plot(x, Fy[:, 1], "o", ms=3.5, mfc="w", mec=BASE, zorder=4)
    ax.text(0.04, 0.11, r"apex $\propto a_3-\alpha_3$", color=APEX, transform=ax.transAxes, fontsize=7)
    ax.text(0.04, 0.04, r"base $\propto a_3+\alpha_3$", color=BASE, transform=ax.transAxes, fontsize=7)
    ax.set(xlabel=r"$r/r_{ss}$", ylabel=r"$F_y / (E_0 d^6 q^4)$")
    ax.legend(frameon=False, loc="upper right", fontsize=7, handlelength=1.6)
    save_panel(fig, FIGURE, "b_isosceles")
    return dict(iso_x=x, iso_F=Fy, iso_coeffs=np.array([a3, al, ea3, eal]), iso_leg=leg, rss_training=rss)


def panel_c(c):
    P, F = T.load_phase_scan()
    s, a0, sd = T.equilateral_curve(P, F)
    rss = T.force_free_side(s, a0)
    to_F0 = 1e-19 / T.F0                       # file units (N x 1e19) -> F0
    Fr, eFr = -np.sqrt(3) * a0 * to_F0, np.sqrt(3) * sd * to_F0
    A = c["a"]
    Fref, slope = -np.sqrt(3) * c["a0"] * to_F0, -np.sqrt(3) * c["S"] * to_F0
    print(f"panel c: {len(s)} equilateral phase-scan configurations, sides {s.min():.1f}-{s.max():.1f} um; "
          f"force-free side r_ss = {rss:.2f} um; reference spacing {A} um = {A / rss:.3f} r_ss")
    fig, ax = plt.subplots(figsize=(3, 3), dpi=144)
    _style(ax)
    ss = np.array([A - 8.0, A + 8.0])
    ax.plot(ss / rss, Fref + slope * (ss - A), color=EQC, lw=1.2, label="model", zorder=3)
    ax.errorbar(s / rss, Fr, yerr=eFr, fmt="o", ms=3.5, mfc="w", mec=EQC, ecolor=EQC, elinewidth=0.8,
                capsize=1.5, label="COMSOL", zorder=4)
    ax.axvline(A / rss, color=MUT, lw=0.8, ls=":", zorder=0)
    ax.text(A / rss, 0.5 * Fref, " reference\n spacing", fontsize=6.5, color="0.35", va="center")
    ax.set(xlim=(0.55, 1.35), xlabel=r"$L/r_{ss}$", ylabel=r"$F_r / (E_0 d^6 q^4)$")
    ax.set_ylim(-0.05 * Fr.max(), 1.1 * Fr[(s / rss) >= 0.55].max())
    ax.legend(frameon=False, loc="upper right", fontsize=7, handlelength=1.6, numpoints=1)
    save_panel(fig, FIGURE, "c_equilateral")
    return dict(eq_side=s, eq_Fr=Fr, eq_Fr_sd=eFr, rss_phase_scan=rss)


def report(c):
    sd, r = c["sd"], c["ratios"]
    print(f"coefficients at a = {c['a']} um (phase scan; N x 1e19, um):")
    print(f"  a0 = {c['a0']:+.5f} +- {sd['a0']:.5f}; S = k_a + 2a3 = {c['S']:+.5f} +- {sd['S']:.5f}")
    print(f"  a3 = {c['a3']:+.4e} +- {sd['a3']:.1e} ({c['n_a3']} configurations); k_a = {c['k_a']:+.5f} +- {sd['k_a']:.5f}")
    print(f"  alpha3 = {c['alpha3']:+.4e} +- {sd['alpha3']:.1e} ({c['n_alpha3']} trimers); "
          f"beta3 = {c['beta3']:+.1e} +- {c['beta3_sd']:.1e} (the achiral null)")
    print("  quoted in the text:")
    print(f"    a0/(ell k_a) = {r['a0_over_aka'][0]:+.3f} +- {r['a0_over_aka'][1]:.3f}")
    print(f"    a3/k_a       = {r['a3_over_ka'][0]:+.3f} +- {r['a3_over_ka'][1]:.3f}")
    print(f"    alpha3/k_a   = {r['alpha3_over_ka'][0]:+.4f} +- {r['alpha3_over_ka'][1]:.4f}")
    same = all(np.isclose(c[k], mm.COEFFS_54[k], rtol=1e-9, atol=0) for k in ("a0", "k_a", "a3", "alpha3"))
    print(f"  {'equal to' if same else 'DIFFERENT FROM'} minimal_model.COEFFS_54 (used by Figs 4b, 5)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", nargs="+", default=list("bc"))
    args = ap.parse_args()
    c = T.coefficients()
    report(c)
    out = dict(coeffs=np.array([c[k] for k in ("a0", "k_a", "a3", "alpha3", "S")]),
               coeffs_sd=np.array([c["sd"][k] for k in ("a0", "k_a", "a3", "alpha3", "S")]),
               ratios=np.array([c["ratios"][k] for k in ("a0_over_aka", "a3_over_ka", "alpha3_over_ka")]))
    if "b" in args.panel:
        out.update(panel_b())
    if "c" in args.panel:
        out.update(panel_c(c))
    save_derived(FIGURE, out)


if __name__ == "__main__":
    main()
