#!/usr/bin/env python3
"""
prep.py -- build the prestressed starting configurations.

For every design, starting from the stress-free alpha1 = 0 equilibrium
(data/init_raw/<design>.npz, the September burn-in endpoint):

  trap      lab-fixed harmonic well about the raft centroid.  kappa is chosen so
            that the local spacing in the OUTER half of the raft over the INNER
            half (split at the median radius, equal particle counts -- the same
            split as in the raft A analysis) equals raft A's measured
            57.4 / 52.1 = 1.102.
  pressure  uniform pressure on the raft's own boundary (co-moving).  P is chosen
            so the MEAN bond length equals the trap configuration's, i.e. the same
            average compression but no gradient.

Each is relaxed to force balance with alpha1 = 0 (FIRE, frozen topology) and
written to data/init/<design>_<prestress>.npz.  A summary goes to
data/init/calibration.json.

    python prep.py                    # all designs
    python prep.py --designs rotor wall4
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import eadyn as E

TARGET_RATIO = 57.4 / 52.1      # raft A, outer / inner local spacing


def spacing_profile(P, tris):
    """(inner median bond, outer median bond, mean bond, edge-ring/centre ratio)."""
    b = E.bonds_from_tris(tris)
    L = np.linalg.norm(P[b[:, 0]] - P[b[:, 1]], axis=1)
    r = np.linalg.norm(P - P.mean(0), axis=1)
    rm = np.median(r)
    inner = (r[b[:, 0]] < rm) & (r[b[:, 1]] < rm)
    outer = (r[b[:, 0]] >= rm) & (r[b[:, 1]] >= rm)
    return float(np.median(L[inner])), float(np.median(L[outer])), float(L.mean())


def relaxed(kind, val, P_start, P_ref, tris):
    """Relax under prestress `val` starting from P_start; trap centre and the
    boundary-edge list come from the stress-free P_ref."""
    ps = E.make_prestress(kind, val, P_ref, tris, c0=P_ref.mean(0))
    f = E.make_force(tris, dict(alpha1=0.0), ps, n=len(P_ref))
    P, fmax = E.relax_fire(f, P_start, ftol=1e-10, maxit=400000, dt0=0.01)
    flips = int(E.n_flipped(P, P_ref, tris))
    return P, fmax, flips


def ramp(kind, vals, P0, tris, stop, label):
    """Quasi-static ramp: relax at each value starting from the previous state.
    Stops at the first value where `stop(profile)` is True or the network folds.
    Returns the list of unfolded states [(val, P, inner, outer, mean, fmax)]."""
    out, P = [], P0.copy()
    for v in vals:
        Pn, fm, fl = relaxed(kind, v, P, P0, tris)
        i, o, mn = spacing_profile(Pn, tris)
        print("   %-8s %.4e  inner=%.4f outer=%.4f ratio=%.4f mean=%.4f  fmax=%.1e  flips=%d"
              % (label, v, i, o, o / i, mn, fm, fl), flush=True)
        if fl:
            print("   %s: network FOLDS at %.4e -- ramp stopped" % (label, v))
            return out, v
        P = Pn
        out.append((v, P, i, o, mn, fm))
        if stop(i, o, mn):
            return out, None
    return out, None


def interp_pick(states, key, target):
    """Bracket `target` in key(state) and re-relax at the interpolated value."""
    ks = np.array([key(s) for s in states])
    j = int(np.searchsorted(ks, target))
    if j == 0 or j >= len(ks):
        return None
    v0, v1 = states[j - 1][0], states[j][0]
    w = (target - ks[j - 1]) / (ks[j] - ks[j - 1])
    return np.exp((1 - w) * np.log(v0) + w * np.log(v1)), states[j - 1][1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--designs", nargs="*", default=["rotor", "wall1", "wall2", "wall3", "wall4"])
    ap.add_argument("--target", type=float, default=1.05,
                    help="outer/inner spacing ratio to reach (raft A measured: %.3f; the linear "
                         "model folds before reaching it, see README)" % TARGET_RATIO)
    a = ap.parse_args()
    out = os.path.join(ROOT, "data", "init")
    os.makedirs(out, exist_ok=True)
    cal_fn = os.path.join(out, "calibration.json")
    cal = json.load(open(cal_fn)) if os.path.exists(cal_fn) else {}

    for des in a.designs:
        d = np.load(os.path.join(ROOT, "data", "init_raw", des + ".npz"))
        P0, tris = d["pos"], d["tris"]
        i0, o0, m0 = spacing_profile(P0, tris)
        print("%s: N=%d  stress-free inner/outer/mean bond = %.4f %.4f %.4f"
              % (des, len(P0), i0, o0, m0))

        # ---- trap ---------------------------------------------------------- #
        st, kfold = ramp("trap", np.logspace(-4.5, -1, 71), P0, tris,
                         lambda i, o, mn: o / i > a.target + 0.005, "kappa")
        best_ratio = max(s[3] / s[2] for s in st)
        pick = interp_pick(st, lambda s: s[3] / s[2], a.target)
        if pick is None:
            raise SystemExit("%s: target ratio %.3f not reachable before folding (max %.4f)"
                             % (des, a.target, best_ratio))
        kap, Pstart = pick
        Pt, ft_, fl = relaxed("trap", kap, Pstart, P0, tris)
        it_, ot_, mt_ = spacing_profile(Pt, tris)
        print("   -> trap kappa=%.4e  ratio=%.4f (target %.3f)  fold at %s  max unfolded ratio %.4f"
              % (kap, ot_ / it_, a.target, kfold, best_ratio))
        np.savez(os.path.join(out, des + "_trap.npz"), pos=Pt, tris=tris, pval=kap,
                 c0=P0.mean(0), fmax=ft_)

        # ---- pressure ------------------------------------------------------ #
        sp, pfold = ramp("pressure", np.linspace(0.005, 0.6, 120), P0, tris,
                         lambda i, o, mn: mn < mt_ - 0.002, "P")
        pick = interp_pick(sp, lambda s: -s[4], -mt_)     # mean bond falls as P rises
        if pick is None:
            raise SystemExit("%s: pressure cannot reach mean bond %.4f" % (des, mt_))
        Pv = pick[0]
        Pp, fp_, flp = relaxed("pressure", Pv, pick[1], P0, tris)
        ip_, op_, mp_ = spacing_profile(Pp, tris)
        print("   -> pressure P=%.5f  mean=%.4f (target %.4f)  ratio=%.4f  fold at %s"
              % (Pv, mp_, mt_, op_ / ip_, pfold))
        np.savez(os.path.join(out, des + "_pressure.npz"), pos=Pp, tris=tris, pval=Pv,
                 c0=Pp.mean(0), fmax=fp_)

        cal[des] = dict(
            raftA_outer_over_inner=TARGET_RATIO, target_used=a.target,
            trap=dict(kappa=kap, inner=it_, outer=ot_, mean=mt_, ratio=ot_ / it_, fmax=ft_,
                      kappa_fold=kfold, max_unfolded_ratio=best_ratio,
                      centre_strain=it_ - 1.0, edge_strain=ot_ - 1.0),
            pressure=dict(P=Pv, inner=ip_, outer=op_, mean=mp_, ratio=op_ / ip_, fmax=fp_,
                          P_fold=pfold))
        json.dump(cal, open(cal_fn, "w"), indent=1)


if __name__ == "__main__":
    main()
