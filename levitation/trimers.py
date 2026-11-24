"""COMSOL three-particle simulations and the minimal-model coefficients measured from them
(Fig 3, the coefficients quoted in the main text, and the coefficient set Figs 4b and 5 use).

Two COMSOL datasets:
  * the phase scan (DATA/phase_scan_result2.mat): 950 three-particle configurations, total force
    on each particle.  The coefficients come from it, expanded about the reference spacing
    A_REF = 54.01 um (the median nearest-neighbour bond of raft A):
      a0      value of the equilateral curve a0(s) = -F_radial(s)/sqrt(3) at A_REF
      S       its slope there, S = k_a + 2 a3                (cubic spline through all equilateral
                                                             configurations; no model fit)
      alpha3  from the net force on each trimer, which only alpha3 can produce:
              F_net = C1 Q + C2 e Q + C3 conj(Q)^2,  alpha3 = Re C1 / (2 (w^2 - 1)),
              Q = s1 + w s2 + w^2 s3 the shear of the triangle (w = exp(2 pi i/3), s_i the side
              opposite particle i), e = mean side - A_REF; trimers with mean side 48-64 um and
              asymmetry 0.05-0.45
      a3      one-parameter least squares on the full forces with a0, S, alpha3 fixed
              (k_a = S - 2 a3), configurations within 25% strain of A_REF
      k_a     = S - 2 a3
    Forces in file units x 1e19 (newtons x 1e19), lengths in um.
  * the isosceles family of the network's training set (DATA/41um_particle/
    three_body_data_41um.pkl): two legs pinned at 79.57 um, the base r varied.  With the legs
    pinned k_a drops out:  F_apex = -(a3 - alpha3) u,  F_base = (a3 + alpha3) u / 2,
    u = leg (x - 1) sqrt(4 - x^2), x = r / leg.  Fig 3b; these a3, alpha3 belong to the 79.6 um
    expansion point and are not the quoted ones.

Uncertainties: a0 and S by leave-one-out over the equilateral configurations, alpha3 by bootstrap
over trimers (600 resamples), a3 from the least-squares covariance, k_a = S - 2 a3 combining S
and a3; ratios by linear propagation treating numerator and k_a as independent (for a3/k_a they
are not, since k_a contains a3; the correlation is ignored).
"""
import os

import numpy as np
from scipy.interpolate import UnivariateSpline

from . import config

A_REF = 54.01                       # reference spacing of the coefficients, um
W = np.exp(2j * np.pi / 3)
SQ3 = np.sqrt(3.0)

# acoustic force scale F0 = E0 d^6 k^4 of the COMSOL runs (the normalization of the COMSOL runs)
DIAM_COMSOL = 41.11e-6              # m
FREQ = 34700.0                      # Hz
C_AIR, RHO_AIR, V0 = 343.0, 1.293, 1e-5
F0 = 0.5 * RHO_AIR * V0 ** 2 * DIAM_COMSOL ** 6 * (2 * np.pi * FREQ / C_AIR) ** 4     # N


# ---- data -----------------------------------------------------------------------------------
def load_phase_scan(force="total"):
    """Positions (n, 3, 2) um and forces (n, 3, 2) in N x 1e19 ("total", "viscous" or
    "scattering" = total - viscous)."""
    from scipy.io import loadmat
    d = loadmat(os.path.join(config.DATA, "phase_scan_result2.mat"))
    P = np.array([np.array(p, float) for p in d["positions"][0]])
    F6 = -np.array([np.array(f, float) for f in d["fs"][0]]) * 1e19
    tot, vis = F6[:, :, :2], F6[:, :, 3:5]
    return P, {"total": tot, "viscous": vis, "scattering": tot - vis}[force]


def load_training_set():
    """Positions (n, 3, 2) um and forces (n, 3, 2) N of the network's COMSOL training set."""
    import pandas as pd
    df = pd.read_pickle(os.path.join(config.DATA, "41um_particle", "three_body_data_41um.pkl"))
    P = np.stack([df[[f"x{i}", f"y{i}"]].to_numpy() for i in (1, 2, 3)], 1)
    F = np.stack([df[[f"f{i}x", f"f{i}y"]].to_numpy() for i in (1, 2, 3)], 1)
    return P, F


# ---- equilateral family: a0 and S -----------------------------------------------------------
def equilateral_curve(P, F, max_asym=0.08):
    """(side, a0, sd of a0 over the three particles) of the near-equilateral configurations
    (max side - min side < max_asym x mean side), sorted by side; a0 = -F_radial / sqrt(3)."""
    L = np.linalg.norm(P[:, [1, 2, 0]] - P[:, [2, 0, 1]], axis=-1)
    side = L.mean(1)
    m = (L.max(1) - L.min(1)) / side < max_asym
    out = P[m] - P[m].mean(1)[:, None, :]
    out = out / np.linalg.norm(out, axis=-1, keepdims=True)
    a0p = -(F[m] * out).sum(-1) / SQ3                 # per particle
    o = np.argsort(side[m])
    return side[m][o], a0p.mean(1)[o], a0p.std(1)[o]


def _unique_mean(s, y):
    su, idx = np.unique(np.round(s, 6), return_inverse=True)
    return su, np.array([y[idx == i].mean() for i in range(len(su))])


def a0_and_slope(P, F, a=A_REF):
    """a0(a), S = da0/ds (a) from a cubic interpolating spline, and leave-one-out sds."""
    su, au = _unique_mean(*equilateral_curve(P, F)[:2])
    sp = UnivariateSpline(su, au, k=3, s=0)
    loo = []
    for i in range(len(su)):
        m = np.arange(len(su)) != i
        q = UnivariateSpline(su[m], au[m], k=3, s=0)
        loo.append((float(q(a)), float(q.derivative()(a))))
    loo = np.array(loo)
    return float(sp(a)), float(sp.derivative()(a)), float(loo[:, 0].std()), float(loo[:, 1].std())


def force_free_side(s, a0):
    """Side at which the equilateral a0(s) first crosses zero (linear interpolation)."""
    q = np.where(np.diff(np.sign(a0)) != 0)[0][0]
    return float(s[q] - a0[q] * (s[q + 1] - s[q]) / (a0[q + 1] - a0[q]))


# ---- net force: alpha3 ----------------------------------------------------------------------
def canonical(P, F):
    """Sides L[:, i] (opposite particle i, labelled counter-clockwise) and the complex net force
    in each trimer's own frame (particle 0 at the origin, particle 1 on +x)."""
    P = np.array(P, float)
    F = np.array(F, float)
    u, v = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
    flip = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0] < 0
    P[flip] = P[flip][:, [0, 2, 1], :]
    F[flip] = F[flip][:, [0, 2, 1], :]
    d = P[:, 1] - P[:, 0]
    th = np.arctan2(d[:, 1], d[:, 0])
    c, s = np.cos(-th), np.sin(-th)
    R = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
    Fn = np.einsum("nab,nb->na", R, F.sum(1))
    L = np.linalg.norm(P[:, [1, 2, 0]] - P[:, [2, 0, 1]], axis=-1)
    return L, Fn[:, 0] + 1j * Fn[:, 1]


def net_force_alpha3(P, F, a=A_REF, side=(48, 64), asym=(0.05, 0.45), nboot=600, seed=0):
    """alpha3 (and its chiral partner beta3) from the trimer net force, with bootstrap sds."""
    L, Fc = canonical(P, F)
    sd = L.mean(1)
    ay = (L.max(1) - L.min(1)) / sd
    m = (sd >= side[0]) & (sd < side[1]) & (ay >= asym[0]) & (ay <= asym[1])
    L, Fc = L[m], Fc[m]
    Q = L[:, 0] + W * L[:, 1] + W ** 2 * L[:, 2]
    X = np.stack([Q, (L.mean(1) - a) * Q, np.conj(Q) ** 2], 1)
    solve = lambda idx: np.linalg.lstsq(X[idx], Fc[idx], rcond=None)[0][0]
    rng = np.random.default_rng(seed)
    c1 = solve(np.arange(len(Fc)))
    bs = np.array([solve(rng.integers(0, len(Fc), len(Fc))) for _ in range(nboot)])
    ab, abs_ = c1 / (2 * (W ** 2 - 1)), bs / (2 * (W ** 2 - 1))
    return dict(alpha3=float(ab.real), beta3=float(ab.imag), sd=float(abs_.real.std()),
                sd_beta=float(abs_.imag.std()), n=int(m.sum()))


# ---- full forces: a3 ------------------------------------------------------------------------
def design(P, a):
    """M with F.ravel() = M @ [a0, k_a, a3, alpha3, b0, k_b, b3, beta3] for each trimer."""
    n = len(P)
    R = P[:, None, :, :] - P[:, :, None, :]              # R[n, i, j] = r_j - r_i
    Ln = np.linalg.norm(R, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        Rhat = R / Ln[..., None]
    dr = Ln - a
    M = np.zeros((n, 3, 2, 8))
    for i in range(3):
        j, k = [x for x in range(3) if x != i]
        for jj, kk in ((j, k), (k, j)):
            u = Rhat[:, i, jj]
            up = np.stack([-u[:, 1], u[:, 0]], -1)
            cs = (np.ones(n), dr[:, i, jj], dr[:, i, kk] + dr[:, jj, kk], dr[:, i, kk] - dr[:, jj, kk])
            for col in range(8):
                M[:, i, :, col] += cs[col % 4][:, None] * (u if col < 4 else up)
    return M.reshape(-1, 8)


def fit_a3(P, F, a0, S, alpha3, a=A_REF, max_strain=0.25, max_asym=0.45):
    """a3 with a0, S = k_a + 2 a3 and alpha3 fixed (b0, k_b, b3 fitted alongside as a null)."""
    Ln = np.linalg.norm(P[:, [0, 0, 1]] - P[:, [1, 2, 2]], axis=-1)
    m = (np.abs(Ln / a - 1).max(1) <= max_strain) & ((Ln.max(1) - Ln.min(1)) / Ln.mean(1) <= max_asym)
    M = design(P[m], a)
    y = F[m].reshape(-1) - M[:, 0] * a0 - M[:, 1] * S - M[:, 3] * alpha3
    X = np.stack([-2 * M[:, 1] + M[:, 2], M[:, 4], M[:, 5], M[:, 6]], 1)
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ c
    err = np.sqrt(np.diag(res @ res / max(len(y) - 4, 1) * np.linalg.pinv(X.T @ X)))
    return dict(a3=float(c[0]), sd=float(err[0]), b_over_sd=np.abs(c[1:] / err[1:]), n=int(m.sum()))


# ---- everything -----------------------------------------------------------------------------
def coefficients(a=A_REF):
    """The minimal-model coefficients at reference spacing a, with uncertainties and the ratios
    quoted in the text."""
    P, F = load_phase_scan()
    a0, S, ea0, eS = a0_and_slope(P, F, a)
    al = net_force_alpha3(P, F, a)
    a3 = fit_a3(P, F, a0, S, al["alpha3"], a)
    ka = S - 2 * a3["a3"]
    eka = np.hypot(eS, 2 * a3["sd"])
    ratio = lambda num, enum, den: (num / den, abs(num / den) * np.hypot(enum / num, eka / ka))
    return dict(a=a, a0=a0, k_a=ka, a3=a3["a3"], alpha3=al["alpha3"], S=S,
                sd=dict(a0=ea0, S=eS, k_a=eka, a3=a3["sd"], alpha3=al["sd"]),
                beta3=al["beta3"], beta3_sd=al["sd_beta"], n_alpha3=al["n"], n_a3=a3["n"],
                ratios=dict(a0_over_aka=ratio(a0 / a, ea0 / a, ka),
                            a3_over_ka=ratio(a3["a3"], a3["sd"], ka),
                            alpha3_over_ka=ratio(al["alpha3"], al["sd"], ka)))


def isosceles_family(leg_target=79.57, tol=0.05):
    """The pinned-leg isosceles scan (apex = particle 3): leg, x = r / leg, and the force along
    the apex direction on the apex, on the base particles (mean) and in total, in N."""
    P, G = load_training_set()
    i, j, k = 2, 0, 1
    lj = np.linalg.norm(P[:, j] - P[:, i], axis=1)
    lk = np.linalg.norm(P[:, k] - P[:, i], axis=1)
    base = np.linalg.norm(P[:, j] - P[:, k], axis=1)
    leg = 0.5 * (lj + lk)
    m = (np.abs(lj - lk) / leg < 1e-9) & (np.abs(leg - leg_target) < tol) & (base / leg < 1.999)
    ax = -(0.5 * (P[m][:, j] + P[m][:, k]) - P[m][:, i])
    ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    x = base[m] / leg[m]
    Fy = np.column_stack([(G[m][:, i] * ax).sum(1),
                          0.5 * ((G[m][:, j] * ax).sum(1) + (G[m][:, k] * ax).sum(1)),
                          (G[m].sum(1) * ax).sum(1)])
    o = np.argsort(x)
    return float(np.median(leg[m])), x[o], Fy[o]


def isosceles_slopes(x, Fy, leg, window=0.20):
    """(a3, alpha3, sd a3, sd alpha3, n) from the apex and base forces with |x - 1| <= window
    (forces in whatever unit Fy is in, per um)."""
    m = np.abs(x - 1.0) <= window
    u = leg * (x[m] - 1.0) * np.sqrt(np.clip(4 - x[m] ** 2, 0, None))
    M = np.vstack([np.column_stack([-u, u]), np.column_stack([u / 2, u / 2])])
    y = np.concatenate([Fy[m, 0], Fy[m, 1]])
    c, *_ = np.linalg.lstsq(M, y, rcond=None)
    r = y - M @ c
    e = np.sqrt(np.diag((r @ r / max(len(y) - 2, 1)) * np.linalg.pinv(M.T @ M)))
    return float(c[0]), float(c[1]), float(e[0]), float(e[1]), int(m.sum())
