#!/usr/bin/env python3
"""
analyze.py -- collect every finished run into figs/summary.csv and four figures.

    python analyze.py

Q1  fig_q1_traj_vs_rigid    theta(t) / com(t) of the overdamped, unprestressed runs
                            against the rigid prediction Omega_pred * t.
Q2  fig_q2_damping          Omega*gamma and |v|*gamma (normalised by alpha and by the
                            overdamped value) vs zeta; effective temperature and
                            Lindemann ratio vs zeta, with zeta_c from stability.py.
Q3  fig_q3_prestress        the same rates, prestress none / trap / pressure side by side;
                            for the swimmer in the lab-fixed trap, the settled centroid
                            offset instead of a speed.
All rates are in lattice units; multiplying by gamma removes the trivial 1/gamma.
"""
import csv, glob, json, os
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GRAYS = {"none": "0.0", "trap": "0.45", "pressure": "0.72"}
MARK = {"none": "o", "trap": "s", "pressure": "^"}


def load():
    rows = []
    for pf in sorted(glob.glob(os.path.join(ROOT, "data", "runs", "*", "*_params.yaml"))):
        m = yaml.safe_load(open(pf))
        r = m["result"]
        rows.append(dict(name=m["name"], design=m["design"], prestress=m["prestress"],
                         mode=m["mode"], zeta=m["zeta"] if m["zeta"] is not None else np.inf,
                         alpha=m["alpha"], kT=m["kT"], status=r["status"], gamma=r["gamma"],
                         Omega=r["Omega"], Omega_pred=r["Omega_pred"],
                         vx=r["v"][0], vy=r["v"][1], vpx=r["v_pred"][0], vpy=r["v_pred"][1],
                         Om_g=r["Omega_gamma"], v_g=float(np.hypot(*r["v_gamma"])),
                         theta=r["theta_total"], dx=r["com_disp"][0], dy=r["com_disp"][1],
                         lind=r["lind_late"], psi6_0=r["psi6_start"], psi6=r["psi6_late"],
                         T_eff=r["T_eff_late"], flips=r.get("flips_max", 0),
                         t_end=r["t_end"], steps=r["steps"], wall_h=r["wall_seconds"] / 3600))
    return rows


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = load()
    os.makedirs(os.path.join(ROOT, "figs"), exist_ok=True)
    if not rows:
        print("no finished runs yet")
        return
    with open(os.path.join(ROOT, "figs", "summary.csv"), "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("%d runs -> figs/summary.csv" % len(rows))
    stab = {}
    sf = os.path.join(ROOT, "data", "stability.json")
    if os.path.exists(sf):
        stab = json.load(open(sf))

    def sel(**kw):
        return [r for r in rows if all(r[k] == v for k, v in kw.items())]

    # ---------------- Q1 --------------------------------------------------- #
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, des in zip(axs, ("rotor", "wall4")):
        for r in sel(design=des, prestress="none", mode="overdamped", kT=0.0):
            if r["alpha"] == 0:
                continue
            d = np.load(os.path.join(ROOT, "data", "runs", r["name"], r["name"] + ".npz"))
            t = d["log_t"]
            if des == "rotor":
                ax.plot(t, d["log_theta"] / (2 * np.pi), c="k", lw=1.4,
                        alpha=0.4 + 6 * r["alpha"], label="alpha=%g  integrated" % r["alpha"])
                ax.plot(t, r["Omega_pred"] * t / (2 * np.pi), "--", c="0.5", lw=1)
                ax.set_ylabel("turns")
            else:
                c = np.hypot(d["log_com0"] - d["log_com0"][0], d["log_com1"] - d["log_com1"][0])
                ax.plot(t, c, c="k", lw=1.4, alpha=0.4 + 6 * r["alpha"],
                        label="alpha=%g  integrated" % r["alpha"])
                ax.plot(t, np.hypot(r["vpx"], r["vpy"]) * t, "--", c="0.5", lw=1)
                ax.set_ylabel("centroid distance / a")
        ax.set_xlabel("time (gamma/k_a)")
        ax.set_title("%s, overdamped, no prestress (dashed: static prediction)" % des, fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figs", "fig_q1_traj_vs_rigid.pdf"))

    # ---------------- Q2 + Q3 --------------------------------------------- #
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.4))
    for row, des in enumerate(("rotor", "wall4")):
        for ps in ("none", "trap", "pressure"):
            for al, ls in ((0.1, "-"), (0.033, "--"), (0.003, ":")):
                R = [r for r in sel(design=des, prestress=ps, alpha=al, kT=0.0)]
                if not R:
                    continue
                R = sorted(R, key=lambda r: r["zeta"])
                # reference: overdamped run, else the most strongly damped run available
                od = [r for r in R if r["mode"] == "overdamped"] or [R[-1]]
                z = np.array([r["zeta"] if np.isfinite(r["zeta"]) else 3.0 for r in R])
                if des == "rotor":
                    y = np.array([r["Om_g"] for r in R]) / al
                    ref = od[0]["Om_g"] / al if od else np.nan
                else:
                    if ps == "trap":
                        y = np.array([np.hypot(r["dx"], r["dy"]) for r in R]) / al
                        ref = np.hypot(od[0]["dx"], od[0]["dy"]) / al if od else np.nan
                    else:
                        y = np.array([r["v_g"] for r in R]) / al
                        ref = od[0]["v_g"] / al if od else np.nan
                bad = np.array([r["status"] in ("blew_up", "folded") for r in R])
                axs[row, 0].plot(z, y / ref, ls, marker=MARK[ps], c=GRAYS[ps], ms=4,
                                 label="%s a=%g" % (ps, al) if ls == "-" else None)
                if bad.any():
                    axs[row, 0].plot(z[bad], (y / ref)[bad], "x", c="r", ms=7)
                axs[row, 1].plot(z, [max(r["T_eff"], 1e-16) for r in R], ls, marker=MARK[ps],
                                 c=GRAYS[ps], ms=4)
                axs[row, 2].plot(z, [r["lind"] for r in R], ls, marker=MARK[ps], c=GRAYS[ps], ms=4)
                k = "%s_%s_a%g" % (des, ps, al)
                if k in stab and ls == "-":
                    for ax in axs[row]:
                        ax.axvline(stab[k]["zeta_c"], c=GRAYS[ps], lw=0.6, ls="-.")
        axs[row, 0].set_ylabel(("Omega*gamma" if des == "rotor" else "|v|*gamma  (trap: offset)")
                               + " / most-damped run", fontsize=8)
        axs[row, 1].set_ylabel("internal kinetic T_eff (floor 1e-16; overdamped = 0)", fontsize=8)
        axs[row, 2].set_ylabel("bond Lindemann (co-rotating)", fontsize=8)
        axs[row, 0].set_title(des, fontsize=9)
        for ax in axs[row]:
            ax.set_xscale("log"); ax.set_xlabel("zeta  (3 = overdamped)", fontsize=8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axs[row, 1].set_yscale("log")
        axs[row, 2].set_yscale("log")
    axs[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("solid a=0.1, dashed 0.033, dotted 0.003; dash-dot verticals: linear zeta_c "
                 "(stability.py); red x: blew up / folded", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figs", "fig_q2q3_damping_prestress.pdf"))
    bad = [r["name"] for r in rows if r["status"] in ("blew_up", "folded")]
    print("blew up / folded:", bad if bad else "none")


if __name__ == "__main__":
    main()
