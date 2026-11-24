"""Compare refactored outputs with the frozen reference (2026-11-23).

    python tools/compare.py fig1 si_rdf ...          # or: all

Reference: $LEVITATION_OUT/reference_2026-11-23/{figures,derived}, i.e. what the original scripts
produced.  New: $LEVITATION_OUT/derived/<figure>.npz written by figures/<figure>.py.

Each check compares one new array with one reference array.  "same" means equal to the stated
tolerance; anything else is reported with its largest difference.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from levitation import config  # noqa: E402

REF = os.path.join(config.OUT, "reference_2026-11-23")

# figure -> list of (new key, reference file (relative to REF), reference key, rtol, atol, note
#                    [, reduction applied to both before comparing: "mean0" / "std0" over axis 0])
CHECKS = {
    "fig1": [
        ("psi6", "derived/hexatic_order.npz", "psi6", 1e-12, 1e-12, "panel c"),
        ("coordination", "derived/hexatic_order.npz", "coordination", 0, 0, "Delaunay coordination"),
        ("mean_pos", "derived/hexatic_order.npz", "mean_pos", 0, 0, "mean positions"),
        ("lags", "figures/notebook_panels_fig1_fig2/msd.npz", "lags", 0, 0, "panel e lags"),
        ("msd", "figures/notebook_panels_fig1_fig2/msd.npz", "msd", 1e-8, 0,
         "panel e, mean over particles (the notebook listed particles in another order)", "mean0"),
        ("msd", "figures/notebook_panels_fig1_fig2/msd.npz", "msd", 1e-8, 0,
         "panel e, std over particles", "std0"),
    ],
    "si_rdf": [
        ("r", "figures/radial_dist_func/radial_dist_func_by_annulus.npz", "r", 0, 0, ""),
        ("R_edges", "figures/radial_dist_func/radial_dist_func_by_annulus.npz", "R_edges", 0, 0, ""),
        ("gr", "figures/radial_dist_func/radial_dist_func_by_annulus.npz", "gr", 1e-12, 1e-12, "panel b"),
        ("r_peak", "figures/radial_dist_func/radial_dist_func_by_annulus.npz", "r_peak", 1e-12, 1e-12, "panel c"),
        ("ell", "derived/hexatic_order.npz", "lspacing", 1e-12, 0, "raft-average spacing (dashed line)"),
    ],
}

CHECKS["fig2"] = [
    ("winding_rate", "figures/particle_winding/particle_winding.npz", "rate", 1e-9, 1e-12, "panel a, winding rate per particle"),
    ("winding_turns", "figures/particle_winding/particle_winding.npz", "turns", 1e-5, 1e-6, "panel a, curves"),
    ("winding_null_rate", "figures/particle_winding/particle_winding.npz", "null_rate", 1e-5, 1e-6, "panel a, surrogates (same random draws)"),
    ("winding_highlighted", "figures/particle_winding/particle_winding.npz", "highlighted", 0, 0, "panel a, highlighted particles"),
    ("L", "figures/notebook_panels_fig1_fig2/angular_momentum.npz", "L", 1e-5, 1e-6, "panel c, L_i (cache vs pickle precision)"),
    ("L_null", "figures/notebook_panels_fig1_fig2/angular_momentum.npz", "null", 0.03, 0,
     "panel c, spread of the noise floor", "stdall"),
    ("V", "figures/pca_circulation/pca_circulation.npz", "V", 1e-6, 1e-9, "panel d, mode shapes"),
    ("variances", "figures/pca_circulation/pca_circulation.npz", "variances", 1e-9, 0, "panel d, variance fractions"),
    ("omega", "figures/pca_circulation/pca_circulation.npz", "omega", 1e-6, 1e-6, "panel f, cycling matrix"),
    ("omega_z", "figures/pca_circulation/pca_circulation.npz", "omega_z", 1e-5, 1e-6, "z-scores (choose panel e's control)"),
    ("lam", "figures/pca_circulation/pca_circulation.npz", "lam", 1e-6, 1e-6, "panel g, spectrum"),
    ("pairs_highlighted", "figures/pca_circulation/pca_circulation.npz", "cycle_pairs", 0, 0, "panel e, highlighted pairs"),
]

_W = "figures/dispersion_welch/dispersion_welch.npz"
CHECKS["raft_a_spectra"] = [
    (k, _W, k, 1e-6, 0, "Fig 4b / SI stripes spectra") for k in
    ("kx_cut", "k_az", "f_cut", "f_az", "SL_cut", "ST_cut", "SL_az", "ST_az", "a_lat", "counts_per_bin")
]

_S = "figures/stripe_modes/stripe_modes.npz"
CHECKS["si_stripes"] = [
    ("f_modes", _S, "f_modes", 0, 0, "panels a-c frequency bins"),
    ("share", _S, "share", 1e-9, 1e-12, "panel a, leading-mode share"),
    ("pr", _S, "pr", 1e-9, 1e-12, "panel b, participation ratio"),
    ("stripe_lines", _S, "L_stripe", 0, 0, "detected lines"),
]

# Outputs compared by their PDF drawing content.
PDF_SAME = {}


def _streams(fn):
    import re
    import zlib
    b = open(fn, "rb").read()
    out = []
    for m in re.finditer(rb"stream\r?\n", b):
        st = m.end()
        en = b.find(b"endstream", st)
        try:
            out.append(zlib.decompress(b[st:en]))
        except Exception:
            pass
    return out


def check_arrays(fig):
    new_f = os.path.join(config.DERIVED, f"{fig}.npz")
    if not os.path.exists(new_f):
        print(f"  MISSING  {new_f} (run figures/{fig}.py first)")
        return False
    new = np.load(new_f, allow_pickle=True)
    ok = True
    reduce = {"mean0": lambda x: x.mean(0), "std0": lambda x: x.std(0),
              "stdall": lambda x: np.array(x.std()), None: lambda x: x}
    for key, rf, rkey, rtol, atol, note, *red in CHECKS[fig]:
        ref = np.load(os.path.join(REF, rf), allow_pickle=True)
        f = reduce[red[0] if red else None]
        a, b = f(np.asarray(new[key])), f(np.asarray(ref[rkey]))
        if a.shape != b.shape:
            print(f"  DIFFERENT {key:14s} shape {a.shape} vs reference {b.shape}   {note}")
            ok = False
            continue
        same = np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.nanmax(np.abs(a - b)) if a.size else 0.0
            rel = np.nanmax(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)) if a.size else 0.0
        tag = "same     " if same else "DIFFERENT"
        print(f"  {tag} {key:14s} max |diff| {float(d):.3g} (relative {float(rel):.3g})   {note}")
        ok &= bool(same)
    return ok


def check_pdfs(fig):
    ok = True
    for new_rel, ref_rel in PDF_SAME[fig]:
        a = _streams(os.path.join(config.OUT, new_rel))
        b = _streams(os.path.join(REF, ref_rel))
        n = sum(x in b for x in a)
        same = n == len(a) == len(b)
        print(f"  {'same     ' if same else 'DIFFERENT'} {new_rel}: {n} of {len(a)} drawing streams "
              f"identical to the reference ({len(b)})")
        ok &= same
    return ok


def report_dispersion():
    """Dispersion fits (every subset) and spacings against the reference run."""
    import csv
    import json
    new_f = os.path.join(config.DERIVED, "dispersion_fits.json")
    if not os.path.exists(new_f):
        print("  MISSING  run figures/tables.py dispersion first")
        return
    new = json.load(open(new_f))
    old = json.load(open(os.path.join(REF, "figures/raft_spectra/kr_kphi_fits.json")))
    sp = {r["raft"]: float(r["a_raft_average_um"])
          for r in csv.DictReader(open(os.path.join(REF, "figures/raft_spectra/raft_average_spacing.csv")))}
    sp["A"] = float(np.load(os.path.join(REF, "derived/hexatic_order.npz"))["lspacing"])
    for t in sorted(new):
        ok = np.isclose(new[t]["boot_sd_all"], old[t]["boot_sd_all"], rtol=1e-9, atol=0)
        for nm, fo in old[t]["fits"].items():
            fn = new[t]["fits"].get(nm)
            if fo is None or fn is None:
                ok &= fo is None and fn is None
                continue
            ok &= fo["n"] == fn["n"] and np.isclose(fo["k"], fn["k"], rtol=1e-9, atol=1e-12) \
                and np.isclose(fo["scale_Hz"], fn["scale_Hz"], rtol=1e-9, atol=0)
        ok_sp = np.isclose(new[t]["ell_um"], sp[t], rtol=1e-6, atol=0)
        fa = new[t]["fits"]["all"]
        print(f"  {'same     ' if ok and ok_sp else 'DIFFERENT'} raft {t}: kappa {fa['k']:+.3f}, {fa['n']} ridge points, "
              f"s {fa['scale_Hz']:.0f} Hz, spacing {new[t]['ell_um']:.2f} um (reference {sp[t]:.2f})"
              f"{'' if ok else '   <-- fits differ'}{'' if ok_sp else '   <-- spacing differs'}")


# Fig 5 and the inference table: the reference run printed these numbers; no arrays were saved.
# A value is "same" if it rounds to the printed one.
FIG5_PRINTED = [  # (label, derived key, index or None, printed value as a string)
    ("perfect disc, interior ratio (round-off level)", "model_glob", 0, "3e-11"),
    ("dislocated disc, interior ratio", "model_glob", 1, "5.18e-05"),
    ("ratio vs strain, Pearson r", "strain_stats", 0, "0.975"),
    ("median ratio / strain", "strain_stats", 1, "1.74e-03"),
    ("uniform-shear prediction", "strain_stats", 2, "1.67e-03"),
    ("alpha3 sweep slope", "alpha3_slope", None, "0.998"),
    ("raft A measured ratio", "exp_stats", 0, "0.2036"),
    ("raft A model ratio", "exp_stats", 1, "0.0015"),
    ("exp vs model Pearson r", "exp_stats", 2, "0.821"),
    ("exp vs model Spearman", "exp_stats", 3, "0.823"),
    ("5-fold defects", "five", "sum", "15"),
    ("7-fold defects", "seven", "sum", "9"),
    ("noise floor mean", "null", "mean", "0.1208"),
    ("noise floor sd", "null", "std", "0.0010"),
    ("control mean", "control", "mean", "0.2039"),
    ("control sd", "control", "std", "0.0014"),
]
INFERENCE_PRINTED = {10: ("0.1545", "0.0761"), 20: ("0.1768", "0.1199"), 40: ("0.2036", "0.1208"),
                     80: ("0.1922", "0.1571"), 160: ("0.2007", "0.1858")}


def _rounds_to(x, printed):
    dec = printed.split("e")[0]
    exp = int(printed.split("e")[1]) if "e" in printed else 0
    nd = len(dec.split(".")[1]) if "." in dec else 0
    return abs(x - float(printed)) <= 0.5 * 10.0 ** (exp - nd) * (1 + 1e-9)


def report_fig5():
    f = os.path.join(config.DERIVED, "fig5.npz")
    if not os.path.exists(f):
        print("  MISSING  run figures/fig5.py first")
        return
    n = np.load(f, allow_pickle=True)
    for label, key, idx, printed in FIG5_PRINTED:
        if key not in n:
            print(f"  missing   {label} (panel not run)")
            continue
        v = np.asarray(n[key])
        x = float(v.sum() if idx == "sum" else v.mean() if idx == "mean" else v.std() if idx == "std"
                  else v if idx is None else v[idx])
        print(f"  {'same     ' if _rounds_to(x, printed) else 'DIFFERENT'} {label:32s} {x:.6g}  (printed {printed})")


def report_inference():
    import json
    f = os.path.join(config.DERIVED, "inference_table.json")
    if not os.path.exists(f):
        print("  MISSING  run figures/tables.py inference first")
        return
    rows = json.load(open(f))
    for M, (meas, null) in INFERENCE_PRINTED.items():
        r = rows[str(M)]
        ok = _rounds_to(r["measured"], meas) and _rounds_to(r["null_mean"], null)
        print(f"  {'same     ' if ok else 'DIFFERENT'} n = {M:3d}: measured {r['measured']:.4f} ({meas}), "
              f"null {r['null_mean']:.4f} ({null}), control {r['control_mean']:.4f} +/- {r['control_sd']:.4f}"
              f"{'' if r['control_passes'] else '  (control outside 3 sd)'}")


def main(figs):
    if figs == ["all"]:
        figs = sorted(set(CHECKS) | set(PDF_SAME)) + ["dispersion", "fig5", "inference"]
    if "dispersion" in figs:
        print("== dispersion fits (SI table, SI raft spectra, Fig 4c)")
        report_dispersion()
        figs = [f for f in figs if f != "dispersion"]
    for name, fn in (("fig5", report_fig5), ("inference", report_inference)):
        if name in figs:
            print(f"== {name} (against the printed reference numbers)")
            fn()
            figs = [f for f in figs if f != name]
    for f in figs:
        print(f"== {f}")
        if f in CHECKS:
            check_arrays(f)
        if f in PDF_SAME:
            check_pdfs(f)


if __name__ == "__main__":
    main(sys.argv[1:] or ["all"])
