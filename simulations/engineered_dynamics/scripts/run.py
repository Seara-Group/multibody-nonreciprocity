#!/usr/bin/env python3
"""
run.py -- one forward simulation of an engineered-activity raft.

    python run.py --design rotor --prestress none --mode overdamped --alpha 0.1 --goal turns:1
    python run.py --row 17                       # row 17 of configs/grid.csv
    python run.py --row 17 --walltime 21000      # checkpoint and exit before 5h50m

Everything is in lattice units (a = 1, k_a = 1).  Overdamped: gamma = 1.
Underdamped: m = 1, gamma = 2*zeta.

The run length is set from the STATIC prediction at the start
    Omega_pred = tau / (gamma * sum r^2),   v_pred = F / (N gamma)
and the goal ('turns:1' -> one full turn, 'dist:10' -> 10 lattice spacings,
'time:5e4' -> fixed time).  The run continues until the MEASURED goal is reached
or t > slack * T_pred (default 2), so a renormalised (slower) rate still
completes.  Nothing about the motion is extrapolated: every saved frame is an
integrated configuration.

Restartable: a checkpoint is written every --ckpt_every seconds and on exit at
--walltime; rerunning the same command resumes.  A finished run is skipped.
"""
import argparse, csv, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)


def parse():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--row", type=int, default=None, help="row of configs/grid.csv (0-based)")
    ap.add_argument("--grid", default=os.path.join(ROOT, "configs", "grid.csv"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--design", default="rotor", help="rotor | wall1..wall4")
    ap.add_argument("--prestress", default="none", choices=["none", "trap", "pressure"])
    ap.add_argument("--mode", default="overdamped", choices=["overdamped", "underdamped"])
    ap.add_argument("--zeta", type=float, default=None, help="damping ratio (underdamped)")
    ap.add_argument("--alpha", type=float, default=0.1, help="alpha1 / k_a")
    ap.add_argument("--kT", type=float, default=0.0, help="k_B T / (k_a a^2)")
    ap.add_argument("--goal", default="turns:1")
    ap.add_argument("--slack", type=float, default=2.0)
    ap.add_argument("--max_time", type=float, default=2e7)
    ap.add_argument("--frames", type=int, default=1500)
    ap.add_argument("--nobs", type=int, default=6000, help="observable samples over T_pred")
    ap.add_argument("--dt_safety", type=float, default=None)
    ap.add_argument("--retri", type=float, default=0.0,
                    help="retriangulate every this many TIME units (0 = frozen topology)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--walltime", type=float, default=1e12, help="seconds before checkpoint+exit")
    ap.add_argument("--ckpt_every", type=float, default=900.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "runs"))
    ap.add_argument("--smoke", action="store_true",
                    help="200 time units into data/smoke/, whatever the row says")
    a = ap.parse_args()
    if a.row is not None:
        with open(a.grid) as f:
            rows = list(csv.DictReader(f))
        r = rows[a.row]
        for k, v in r.items():
            if k == "row" or v == "":
                continue
            ty = type(getattr(a, k)) if getattr(a, k, None) is not None else str
            if k in ("zeta", "alpha", "kT", "slack", "max_time", "retri", "dt_safety"):
                ty = float
            if k in ("frames", "nobs", "seed"):
                ty = int
            setattr(a, k, ty(v))
    if a.smoke:
        a.goal, a.frames, a.nobs = "time:200", 20, 200
        a.out = os.path.join(ROOT, "data", "smoke")
    if a.mode == "underdamped" and a.zeta is None:
        raise SystemExit("--zeta is required for underdamped runs")
    if a.name is None:
        z = "od" if a.mode == "overdamped" else "z%g" % a.zeta
        a.name = "%s_%s_%s_a%g%s" % (a.design, a.prestress, z, a.alpha,
                                     "_kT%g" % a.kT if a.kT else "")
    return a


def load_init(design, prestress):
    """Start configuration: alpha1 = 0 equilibrium WITH the prestress applied."""
    if prestress == "none":
        d = np.load(os.path.join(ROOT, "data", "init_raw", design + ".npz"))
        return dict(pos=d["pos"], tris=d["tris"], pval=0.0, c0=d["pos"].mean(0))
    fn = os.path.join(ROOT, "data", "init", "%s_%s.npz" % (design, prestress))
    if not os.path.exists(fn):
        raise SystemExit("%s missing -- run scripts/prep.py first" % fn)
    d = np.load(fn)
    return dict(pos=d["pos"], tris=d["tris"], pval=float(d["pval"]), c0=d["c0"])


def main():
    a = parse()
    import jax, jax.numpy as jnp
    import eadyn as E

    od = os.path.join(a.out, a.name)
    os.makedirs(od, exist_ok=True)
    final = os.path.join(od, a.name + ".npz")
    if os.path.exists(final):
        print("done already:", final)
        return
    ck = os.path.join(od, "checkpoint.npz")
    t_wall0 = time.time()

    ini = load_init(a.design, a.prestress)
    tris = ini["tris"]
    N = len(ini["pos"])
    m = 1.0
    gamma = 1.0 if a.mode == "overdamped" else 2.0 * a.zeta
    coeffs = dict(alpha1=a.alpha)

    def build(P, tris):
        ps = E.make_prestress(a.prestress, ini["pval"], P, tris, c0=ini["c0"])
        f = E.make_force(tris, coeffs, ps, n=N)
        return f

    f = build(ini["pos"], tris)
    P0 = jnp.asarray(ini["pos"])
    lam = E.max_stiffness(f, P0, iters=150)
    saf = a.dt_safety if a.dt_safety else (0.5 if a.mode == "overdamped" else 0.2)
    dt = E.auto_dt(lam, a.mode, gamma, m, saf)

    # ---- static prediction and run length --------------------------------- #
    F0 = f(P0)
    Fn, tau = E.totals(P0, F0)
    zR = float(jnp.sum((P0 - P0.mean(0)) ** 2))
    Om_pred = float(tau) / (gamma * zR)
    v_pred = np.asarray(Fn) / (N * gamma)
    kind, val = a.goal.split(":")
    val = float(val)
    if kind == "turns":
        T_pred = val * 2 * np.pi / max(abs(Om_pred), 1e-300)
    elif kind == "dist":
        T_pred = val / max(np.linalg.norm(v_pred), 1e-300)
    else:
        T_pred = val
    # underdamped: never stop before the inertial start-up transient (m/gamma) is over
    T_min = 10.0 * m / gamma if a.mode == "underdamped" else 0.0
    T_pred = min(max(T_pred, T_min), a.max_time)
    T_max = min(a.slack * T_pred, a.max_time) if kind != "time" else T_pred
    n_obs_steps = max(1, int(round(T_pred / a.nobs / dt)))     # steps per chunk
    frame_every = max(1, int(round(T_pred / a.frames / (n_obs_steps * dt))))
    print("run %s" % a.name)
    print("   N=%d  tris=%d  gamma=%g  dt=%.4g (lam_max=%.3f, safety %.2f)"
          % (N, len(tris), gamma, dt, lam, saf))
    print("   static prediction: Omega=%.4e  v=(%.3e,%.3e)  -> T_pred=%.4g  (%d steps)"
          % (Om_pred, v_pred[0], v_pred[1], T_pred, T_pred / dt))

    obs = E.make_observe(f, P0, tris, m, a.mode == "underdamped")
    chunk = E.make_chunk(f, a.mode, dt, gamma, m, a.kT)

    # ---- state / resume --------------------------------------------------- #
    if os.path.exists(ck):
        c = np.load(ck, allow_pickle=True)
        P, V = jnp.asarray(c["P"]), jnp.asarray(c["V"])
        key = jnp.asarray(c["key"])
        step = int(c["step"])
        log = {k: list(v) for k, v in c["log"].item().items()}
        frames, ftimes = list(c["frames"]), list(c["ftimes"])
        tris = c["tris"]
        if a.retri:
            f = build(np.asarray(P), tris)
            obs = E.make_observe(f, P0, tris, m, a.mode == "underdamped")
            chunk = E.make_chunk(f, a.mode, dt, gamma, m, a.kT)
        print("   resumed at step %d (t=%.4g)" % (step, step * dt))
    else:
        P, V = P0, jnp.zeros_like(P0)
        key = jax.random.PRNGKey(a.seed)
        step = 0
        log = {}
        frames, ftimes = [np.asarray(P0, np.float32)], [0.0]

    def record(P, V, step):
        o = obs(P, V)
        for k, v in o.items():
            v = np.asarray(v)
            if v.ndim:
                for i, x in enumerate(v):
                    log.setdefault("%s%d" % (k, i), []).append(float(x))
            else:
                log.setdefault(k, []).append(float(v))
        log.setdefault("t", []).append(step * dt)

    if step == 0:
        record(P, V, 0)

    def theta_unwrapped():
        return np.unwrap(np.array(log["theta"]))

    def goal_met():
        if step * dt < T_min:
            return False
        if kind == "turns":
            return abs(theta_unwrapped()[-1]) >= val * 2 * np.pi
        if kind == "dist":
            c = np.array([log["com0"][-1] - log["com0"][0], log["com1"][-1] - log["com1"][0]])
            return np.linalg.norm(c) >= val
        return step * dt >= T_pred

    def save_ckpt():
        np.savez(ck, P=np.asarray(P), V=np.asarray(V), key=np.asarray(key), step=step,
                 log=np.array(log, dtype=object), frames=np.array(frames),
                 ftimes=np.array(ftimes), tris=tris)

    status = "running"
    t_ck = time.time()
    nchunk = 0
    retri_steps = int(round(a.retri / dt)) if a.retri else 0
    last_retri = step
    while True:
        P, V, key = chunk(P, V, key, n_obs_steps)
        step += n_obs_steps
        nchunk += 1
        record(P, V, step)
        if nchunk % frame_every == 0:
            frames.append(np.asarray(P, np.float32))
            ftimes.append(step * dt)
        if not np.isfinite(log["fmax"][-1]) or log["lind"][-1] > 5.0:
            status = "blew_up"
            break
        if log["flips"][-1] > 0 and not a.retri:
            status = "folded"          # unphysical: the network passed through itself
            break
        if goal_met():
            status = "goal_met"
            break
        if step * dt >= T_max:
            status = "time_limit"
            break
        if retri_steps and step - last_retri >= retri_steps:
            tris = E.triangles(np.asarray(P))
            f = build(np.asarray(P), tris)
            obs = E.make_observe(f, P0, tris, m, a.mode == "underdamped")
            chunk = E.make_chunk(f, a.mode, dt, gamma, m, a.kT)
            last_retri = step
        if time.time() - t_ck > a.ckpt_every:
            save_ckpt()
            t_ck = time.time()
            print("   t=%.4g (%.1f%% of T_pred)  theta=%.4f  com=(%.3f,%.3f)  lind=%.3e"
                  % (step * dt, 100 * step * dt / T_pred, theta_unwrapped()[-1],
                     log["com0"][-1], log["com1"][-1], log["lind"][-1]), flush=True)
        if time.time() - t_wall0 > a.walltime:
            save_ckpt()
            print("   walltime reached at step %d; resubmit to continue" % step)
            return
    if frames and ftimes[-1] != step * dt:
        frames.append(np.asarray(P, np.float32))
        ftimes.append(step * dt)

    # ---- measured rates (late 60% of the run, from the trajectory) -------- #
    L = {k: np.array(v) for k, v in log.items()}
    L["theta"] = theta_unwrapped()
    t = L["t"]
    msk = t >= 0.4 * t[-1]
    Om = float(np.polyfit(t[msk], L["theta"][msk], 1)[0]) if msk.sum() > 3 else np.nan
    vx = float(np.polyfit(t[msk], L["com0"][msk], 1)[0]) if msk.sum() > 3 else np.nan
    vy = float(np.polyfit(t[msk], L["com1"][msk], 1)[0]) if msk.sum() > 3 else np.nan
    res = dict(status=status, t_end=float(t[-1]), steps=int(step), dt=float(dt),
               gamma=gamma, Omega=Om, v=[vx, vy], Omega_pred=Om_pred,
               v_pred=[float(v_pred[0]), float(v_pred[1])],
               Omega_gamma=Om * gamma, v_gamma=[vx * gamma, vy * gamma],
               theta_total=float(L["theta"][-1]),
               com_disp=[float(L["com0"][-1] - L["com0"][0]), float(L["com1"][-1] - L["com1"][0])],
               lind_late=float(np.mean(L["lind"][msk])),
               psi6_start=float(L["psi6"][0]), psi6_late=float(np.mean(L["psi6"][msk])),
               ke_int_late=float(np.mean(L["ke_int"][msk])),
               T_eff_late=float(np.mean(L["ke_int"][msk]) / max(N - 1.5, 1)),
               flips_max=int(np.max(L["flips"])),
               wall_seconds=float(time.time() - t_wall0))
    np.savez_compressed(final, trajectory_R=np.array(frames, np.float32),
                        times=np.array(ftimes), triangles_final=tris,
                        triangles0=ini["tris"], pos0=ini["pos"],
                        **{"log_" + k: v for k, v in L.items()})
    import yaml
    meta = {k: getattr(a, k) for k in ("name", "design", "prestress", "mode", "zeta", "alpha",
                                         "kT", "goal", "slack", "retri", "seed")}
    meta.update(N=N, n_tri=int(len(ini["tris"])), pval=ini["pval"], lam_max=lam,
                coeffs=dict(a=1.0, k_a=1.0, alpha1=a.alpha, m=m, gamma=gamma),
                synthetic_rigid_extrapolation=False, result=res)
    with open(os.path.join(od, a.name + "_params.yaml"), "w") as fo:
        yaml.safe_dump(meta, fo, sort_keys=False)
    if os.path.exists(ck):
        os.remove(ck)
    print("   %s: Omega=%.4e (pred %.4e)  v=(%.3e,%.3e)  theta=%.3f  lind=%.3e  psi6 %.3f->%.3f"
          % (status, Om, Om_pred, vx, vy, res["theta_total"], res["lind_late"],
             res["psi6_start"], res["psi6_late"]))


if __name__ == "__main__":
    main()
