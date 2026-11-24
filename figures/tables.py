"""SI tables, printed as LaTeX rows.

    python tables.py dispersion      # tab:dispersion_fits
    python tables.py inference       # tab:inference_M (added with Fig 5)

dispersion: for each raft, particle diameter d, drive frequency f_d, lattice spacing ell (first
peak of g(r)), fitted kappa = k_phiphi/k_rr, frequency scale s = sqrt(k_rr/m)/2pi, the
long-wavelength sound speeds c_L, c_T = 2 pi s ell sqrt((9+3k)/8), sqrt((3+9k)/8), the Poisson ratio
nu = (1-5k)/(3+k), the number N of ridge points in the fit and the bootstrap sd of kappa.
Rows A, B, D are the raft-forming conditions; C and E rearrange during the record.

inference: raft A's ||D-||/||D+|| inferred on the n highest-variance modes, for n = 10, 20, 40, 80,
160, against 40 reciprocal surrogates (mean +- sd), and the control (surrogates from the full
cross spectrum, mean over max(8, 40/4) = 10 realizations).  Same machinery as Fig 5
(levitation.inference); n = 40 is the value used there.  The numbers are also written to
OUT/derived/inference_table.json.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from levitation import dispersion, inference  # noqa: E402
from levitation.plotting import derived_path  # noqa: E402

CONDITIONS = {"A": (41.1, 34.7), "B": (41.1, 64.5), "D": (59.7, 64.4), "C": (59.7, 34.7), "E": (32.0, 34.8)}


def dispersion_table():
    fits = dispersion.fit_all()
    lines = []
    for i, tag in enumerate(("A", "B", "D", "C", "E")):
        r = fits[tag]
        fa, sp = r["fits"]["all"], r["speeds"]
        d, fd = CONDITIONS[tag]
        if i == 3:
            lines.append(r"\hline")
        lines.append(f"{tag} & {d:.1f} & {fd:.1f} & {r['ell_um']:.1f} & ${fa['k']:+.3f}$ & {fa['scale_Hz']:.0f} & "
                     f"{sp['cL']:.0f} & {sp['cT']:.0f} & {sp['nu']:.3f} & {fa['n']} & {r['boot_sd_all']:.4f} \\\\")
    ks = np.array([fits[t]["fits"]["all"]["k"] for t in "ABCDE"])
    print("\n".join(lines))
    print(f"% kappa over the five conditions: mean {ks.mean():+.3f}, sd {ks.std():.3f}; "
          f"raft-forming A, B, D: mean {np.mean([fits[t]['fits']['all']['k'] for t in 'ABD']):+.3f}")


def inference_table(mode_counts=(10, 20, 40, 80, 160), n_null=40):
    import json
    U, _ = inference.raft_a_displacements()
    Q, _ = inference.mode_basis(U)
    rows, lines = {}, []
    for M in mode_counts:
        a = U @ Q[:, :M]
        meas = inference.ratio(inference.mode_dynamical_matrix(a))
        fl = inference.null_and_control(a, meas, n_null)
        rows[M] = dict(measured=meas, null_mean=fl["mean"], null_sd=fl["sd"], control_mean=fl["control_mean"],
                       control_sd=fl["control_sd"], n_control=len(fl["control"]), control_passes=fl["control_passes"])
        lines.append(f"{M:<3d} & {meas:.3f} & ${fl['mean']:.3f} \\pm {fl['sd']:.3f}$ & {meas / fl['mean']:.2f} & "
                     f"{fl['control_mean']:.3f} \\\\")
        print(f"% n = {M}: measured {meas:.4f}, null {fl['mean']:.4f} +/- {fl['sd']:.4f}, control "
              f"{fl['control_mean']:.4f} +/- {fl['control_sd']:.4f} ({'passes' if fl['control_passes'] else 'FAILS'} "
              f"the 3-sd check)", flush=True)
    print("\n".join(lines))
    json.dump(rows, open(derived_path("inference_table.json"), "w"), indent=1)


if __name__ == "__main__":
    which = sys.argv[1:] or ["dispersion"]
    if "dispersion" in which:
        dispersion_table()
    if "inference" in which:
        inference_table()
