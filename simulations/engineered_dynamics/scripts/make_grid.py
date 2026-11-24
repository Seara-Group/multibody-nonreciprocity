#!/usr/bin/env python3
"""
make_grid.py -- write configs/grid.csv, one row per run, with a cost estimate.

Main grid  : design {rotor, wall4} x prestress {none, trap, pressure}
             x damping {overdamped, zeta = 0.3, 0.1, 0.03, 0.01, 0.003}
             x alpha1/k_a {0 (control), 0.003 (physical), 0.033, 0.1}
Noise set  : kT = 1e-4 for zeta in {0.01, 0.003}, alpha in {0.003, 0.1},
             prestress in {none, trap}: does a thermal kick trigger the
             underdamped instability that a T = 0 run might sit on?

Goals: rotor -> 1 full turn; swimmer -> 10 lattice spacings; swimmer in the
lab-fixed trap -> a fixed time long enough for its offset to settle.  If a goal
would need more than --max_steps, the row becomes a fixed-time run of
max_steps (the rate is then fitted from the trajectory, still no extrapolation).
alpha = 0 controls run for the same time as the alpha = 0.1 row they control.

    python make_grid.py [--max_steps 1.2e8] [--us_per_step 300]
"""
import argparse, csv, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

DAMP = [("overdamped", None), ("underdamped", 0.3), ("underdamped", 0.1),
        ("underdamped", 0.03), ("underdamped", 0.01), ("underdamped", 0.003)]
ALPHAS = [0.1, 0.033, 0.003, 0.0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max_steps", type=float, default=1.2e8)
    ap.add_argument("--us_per_step", type=float, default=300.0,
                    help="for the printed cost estimate only (300 = 2 CPU cores, N=1759)")
    ap.add_argument("--designs", nargs="*", default=["rotor", "wall4"])
    a = ap.parse_args()
    import jax.numpy as jnp
    import eadyn as E
    from run import load_init

    # static response per unit alpha (exactly linear: gate G2) and dt per mode
    base = {}
    for des in a.designs:
        for ps in ("none", "trap", "pressure"):
            ini = load_init(des, ps)
            P = jnp.asarray(ini["pos"])
            pre = E.make_prestress(ps, ini["pval"], ini["pos"], ini["tris"], c0=ini["c0"])
            f1 = E.make_force(ini["tris"], dict(alpha1=1.0), pre, n=len(P))
            f0 = E.make_force(ini["tris"], dict(alpha1=0.0), pre, n=len(P))
            dF = f1(P) - f0(P)
            Fn, tau = E.totals(P, dF)
            zR = float(jnp.sum((P - P.mean(0)) ** 2))
            lam = E.max_stiffness(f0, P, iters=150)
            base[(des, ps)] = dict(tau=float(tau), F=float(jnp.linalg.norm(Fn)), zR=zR,
                                   N=len(P), lam=lam, pval=ini["pval"])
            print("%-6s %-8s tau/alpha=%+.4e |F|/alpha=%.4e  lam_max=%.3f  pval=%.4g"
                  % (des, ps, tau, float(jnp.linalg.norm(Fn)), lam, ini["pval"]))

    rows = []

    def add(des, ps, mode, zeta, al, kT=0.0):
        b = base[(des, ps)]
        gamma = 1.0 if mode == "overdamped" else 2 * zeta
        dt = E.auto_dt(b["lam"], mode, gamma, 1.0, 0.5 if mode == "overdamped" else 0.2)
        al_eff = al if al else 0.1                      # controls: time of the 0.1 run
        Om = abs(b["tau"] * al_eff) / (gamma * b["zR"])
        v = b["F"] * al_eff / (b["N"] * gamma)
        if des == "rotor":
            goal, T = "turns:1", 2 * np.pi / Om
        elif ps == "trap":
            hold = max(20 * gamma / b["pval"], (20 / zeta) if zeta else 0.0, 2e4)
            goal, T = "time:%.6g" % hold, hold
        else:
            goal, T = "dist:10", 10.0 / v
        if mode == "underdamped":
            T = max(T, 10.0 / gamma)          # run.py never stops before 10 m/gamma
        if al == 0:
            goal = "time:%.6g" % min(T, a.max_steps * dt)
            T = min(T, a.max_steps * dt)
        if T / dt > a.max_steps:
            T = a.max_steps * dt
            goal = "time:%.6g" % T
        rows.append(dict(design=des, prestress=ps, mode=mode,
                         zeta="" if zeta is None else zeta, alpha=al, kT=kT, goal=goal,
                         max_time=1e12, est_steps=int(T / dt),
                         est_hours=round(T / dt * a.us_per_step * 1e-6 / 3600, 2)))

    for des in a.designs:
        for ps in ("none", "trap", "pressure"):
            for mode, z in DAMP:
                for al in ALPHAS:
                    add(des, ps, mode, z, al)
    for des in a.designs:
        for ps in ("none", "trap"):
            for z in (0.01, 0.003):
                for al in (0.1, 0.003):
                    add(des, ps, "underdamped", z, al, kT=1e-4)

    fn = os.path.join(ROOT, "configs", "grid.csv")
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=["row"] + list(rows[0].keys()))
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow(dict(row=i, **r))
    h = np.array([r["est_hours"] for r in rows])
    print("\n%d rows -> %s" % (len(rows), fn))
    print("estimated hours at %g us/step: total %.0f, max %.1f, rows > 12 h: %d"
          % (a.us_per_step, h.sum(), h.max(), (h > 12).sum()))


if __name__ == "__main__":
    main()
