#!/usr/bin/env python3
"""
stability.py -- linear prediction for question 2 (what does inertia do?).

Linearise about the alpha1 = 0 equilibrium (with or without prestress), keeping
alpha1 in the Jacobian:  D = -dF/dR.  For an eigenvalue lambda of D the mode obeys

    m s^2 + gamma s + lambda = 0 .

* overdamped (m -> 0):  s = -lambda/gamma, unstable iff Re(lambda) < 0 -- at ANY gamma.
* underdamped:          for Re(lambda) > 0 the mode grows iff
                            gamma < |Im lambda| / sqrt(m Re lambda)
                        i.e. below a critical damping ratio
                            zeta_c = |Im lambda| / (2 sqrt(m k_a Re lambda)).

Nonreciprocity (alpha1) is what makes Im(lambda) nonzero, so zeta_c grows with
alpha1 (linearly, for small alpha1).  A raft with zeta < zeta_c should heat up and
eventually melt; above zeta_c it should behave like the overdamped one.

Rigid-body zero modes (|lambda| < tol) are excluded.

    python stability.py                          # all designs x prestress x alpha
    python stability.py --designs rotor --alphas 0.1
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--designs", nargs="*", default=["rotor", "wall4"])
    ap.add_argument("--prestress", nargs="*", default=["none", "trap", "pressure"])
    ap.add_argument("--alphas", nargs="*", type=float, default=[0.003, 0.033, 0.1])
    ap.add_argument("--zetas", nargs="*", type=float, default=[0.003, 0.01, 0.03, 0.1, 0.3])
    ap.add_argument("--tol", type=float, default=1e-7)
    a = ap.parse_args()
    import jax, jax.numpy as jnp
    import eadyn as E
    from run import load_init

    out = {}
    fn = os.path.join(ROOT, "data", "stability.json")
    if os.path.exists(fn):
        out = json.load(open(fn))
    for des in a.designs:
        for ps in a.prestress:
            ini = load_init(des, ps)
            P = jnp.asarray(ini["pos"]).ravel()
            for al in a.alphas:
                pre = E.make_prestress(ps, ini["pval"], ini["pos"], ini["tris"], c0=ini["c0"])
                f = E.make_force(ini["tris"], dict(alpha1=al), pre, n=len(ini["pos"]))
                D = -np.asarray(jax.jacfwd(lambda x: f(x.reshape(-1, 2)).ravel())(P))
                lam = np.linalg.eigvals(D)
                nz = np.abs(lam) > a.tol
                l = lam[nz]
                neg = l[l.real < 0]
                pos = l[l.real > 0]
                zc = np.abs(pos.imag) / (2 * np.sqrt(pos.real))
                j = int(np.argmax(zc))
                rate = {}
                for z in a.zetas:
                    g = 2 * z
                    s = (-g + np.sqrt(g * g - 4 * l.astype(complex))) / 2
                    s2 = (-g - np.sqrt(g * g - 4 * l.astype(complex))) / 2
                    rate[str(z)] = float(max(s.real.max(), s2.real.max()))
                key = "%s_%s_a%g" % (des, ps, al)
                out[key] = dict(design=des, prestress=ps, alpha=al,
                                n_modes=int(nz.sum()), n_negative_real=int(len(neg)),
                                min_real=float(l.real.min()),
                                overdamped_max_growth=float(max(0.0, -l.real.min())),
                                zeta_c=float(zc.max()), zeta_c_mode_lambda=[float(pos[j].real), float(pos[j].imag)],
                                n_modes_unstable_at={z: int((zc > float(z)).sum()) for z in map(str, a.zetas)},
                                max_growth_rate_at=rate, max_abs_imag=float(np.abs(l.imag).max()))
                r = out[key]
                print("%-24s Re<0: %3d  min Re %+.2e   zeta_c = %.4f   unstable modes at zeta=%s: %s"
                      % (key, r["n_negative_real"], r["min_real"], r["zeta_c"],
                         ",".join(a_ for a_ in map(str, a.zetas)),
                         ",".join(str(r["n_modes_unstable_at"][z]) for z in map(str, a.zetas))),
                      flush=True)
                json.dump(out, open(fn, "w"), indent=1)


if __name__ == "__main__":
    main()
