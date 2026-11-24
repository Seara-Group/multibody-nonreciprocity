"""
eadyn.py -- self-contained dynamics of the triangle minimal model (engineered activity).

Nothing here imports from the rest of the repository.  The force law is written
out below and is the same one as levitation/minimal_model.py (checks in scripts/validate.py).

FORCE LAW (lattice units: rest spacing a = 1, k_a = 1)
------------------------------------------------------
    F_i = sum over nearest-neighbour TRIANGLES containing i, over both ordered
          pairs (j,k) of the other two vertices, of

          A(d_ij; d_ik, d_jk) rhat_ij + B(d_ij; d_ik, d_jk) rhat_ij^perp

    A = a0 + k_a d_ij + a1 (d_ik + d_jk) + alpha1 (d_ik - d_jk)
    B = b0 + k_b d_ij + b1 (d_ik + d_jk) + beta1  (d_ik - d_jk)

with d = |r| - a and rhat_ij pointing from i to j.  alpha1 is the only
nonreciprocal (momentum-violating) coefficient used here; everything else is
zero except k_a = 1, exactly as in the September engineered-activity runs.

Topology (the triangle list) is FROZEN unless a run asks for retriangulation.

EQUATIONS OF MOTION
-------------------
  overdamped    gamma dr = F dt + sqrt(2 gamma kT dt) xi
  underdamped   m dv = (F - gamma v) dt + sqrt(2 gamma kT dt) xi     (BAOAB)

Underdamped runs use m = 1, gamma = 2 zeta, so zeta = gamma / (2 sqrt(m k_a)) is
the damping ratio of a single bond.  Steady rigid-body rates scale as 1/gamma in
BOTH cases, so compare Omega*gamma and v*gamma across damping.

PRESTRESS
---------
  'trap'      lab-fixed harmonic well about c0:  F_i = -kappa (r_i - c0).
              Graded pressure, highest at the centre (p ~ R^2 - r^2).
  'pressure'  uniform pressure P on the raft's own boundary, moving with it:
              each boundary edge (p,q) receives -P * (outward normal * length),
              split equally between its two ends.  Sums to EXACTLY zero net force
              and zero torque for every closed boundary, so it cannot make the
              raft swim or spin by itself.
"""
import os
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)    # torques are ~1% residues: keep f64

RC = 1.45   # triangle edge cutoff, units of a (same as the numpy code)


# =========================================================================== #
# topology (host side, numpy/scipy)
# =========================================================================== #
def triangles(pos, rc=RC):
    from scipy.spatial import Delaunay
    P = np.asarray(pos)
    sm = Delaunay(P).simplices
    e = [np.linalg.norm(P[sm[:, u]] - P[sm[:, v]], axis=1)
         for u, v in ((0, 1), (1, 2), (2, 0))]
    keep = np.maximum(np.maximum(e[0], e[1]), e[2]) < rc
    return np.ascontiguousarray(sm[keep]).astype(np.int64)


def ordered_terms(tris):
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    ijk = [(v0, v1, v2), (v0, v2, v1), (v1, v2, v0),
           (v1, v0, v2), (v2, v0, v1), (v2, v1, v0)]
    return tuple(np.concatenate([t[n] for t in ijk]) for n in range(3))


def bonds_from_tris(tris):
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.sort(e, 1)
    return np.unique(e, axis=0)


def coordination(tris, n):
    b = bonds_from_tris(tris)
    c = np.zeros(n, int)
    np.add.at(c, b[:, 0], 1)
    np.add.at(c, b[:, 1], 1)
    return c


def incidence(tris, n):
    """Padded (particle -> [(triangle, slot)]) table so forces are GATHERED, not
    scattered (much faster in XLA).  Returns (tri_idx, slot, mask), each (n, kmax)."""
    tris = np.asarray(tris)
    lists = [[] for _ in range(n)]
    for t, tri in enumerate(tris):
        for s, v in enumerate(tri):
            lists[v].append((t, s))
    kmax = max(1, max(len(l) for l in lists))
    ti = np.zeros((n, kmax), np.int64); si = np.zeros((n, kmax), np.int64)
    mk = np.zeros((n, kmax))
    for v, l in enumerate(lists):
        for q, (t, s) in enumerate(l):
            ti[v, q], si[v, q], mk[v, q] = t, s, 1.0
    return jnp.asarray(ti), jnp.asarray(si), jnp.asarray(mk)


def boundary_edges(pos, tris):
    """Edges belonging to exactly one triangle, ORIENTED so that the triangle's
    third vertex lies to the left of p->q.  The outward normal is then the
    right-hand perpendicular (e_y, -e_x)/|e|."""
    P = np.asarray(pos)
    cnt = {}
    for t in tris:
        for u, v, w in ((t[0], t[1], t[2]), (t[1], t[2], t[0]), (t[2], t[0], t[1])):
            key = (min(u, v), max(u, v))
            cnt.setdefault(key, []).append(w)
    out = []
    for (p, q), ws in cnt.items():
        if len(ws) != 1:
            continue
        r = ws[0]
        e, f = P[q] - P[p], P[r] - P[p]
        if e[0] * f[1] - e[1] * f[0] < 0:
            p, q = q, p
        out.append((p, q))
    return np.array(out, np.int64)


# =========================================================================== #
# forces (jax)
# =========================================================================== #
DEFAULT_COEFFS = dict(a=1.0, a0=0.0, k_a=1.0, a1=0.0, alpha1=0.0,
                      b0=0.0, k_b=0.0, b1=0.0, beta1=0.0)


def make_force(tris, coeffs, prestress=None, n=None):
    """Returns F(P) -> (N,2) as a jit-able closure.

    prestress: None, ('trap', kappa, c0) or ('pressure', P, boundary_edges)."""
    c = {**DEFAULT_COEFFS, **coeffs}
    tris = np.asarray(tris)
    T = jnp.asarray(tris)
    inc_t, inc_s, inc_m = incidence(tris, int(tris.max()) + 1 if n is None else n)

    def term(ei, li, ek, lk, ej_k, lj_k):
        """Force on vertex i from ordered pair (j,k): ei = r_j - r_i (len li),
        lk = |r_k - r_i|, lj_k = |r_k - r_j|."""
        dij, dik, djk = li - c["a"], lk - c["a"], lj_k - c["a"]
        A = c["a0"] + c["k_a"] * dij + c["a1"] * (dik + djk) + c["alpha1"] * (dik - djk)
        B = c["b0"] + c["k_b"] * dij + c["b1"] * (dik + djk) + c["beta1"] * (dik - djk)
        hat = ei / li[:, None]
        perp = jnp.stack([-hat[:, 1], hat[:, 0]], 1)
        return A[:, None] * hat + B[:, None] * perp

    def f_internal(P):
        p0, p1, p2 = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
        e01, e02, e12 = p1 - p0, p2 - p0, p2 - p1
        l01, l02, l12 = (jnp.linalg.norm(e, axis=1) for e in (e01, e02, e12))
        # vertex 0: pairs (1,2) and (2,1);  vertex 1: (2,0),(0,2);  vertex 2: (0,1),(1,0)
        f0 = term(e01, l01, e02, l02, e12, l12) + term(e02, l02, e01, l01, -e12, l12)
        f1 = term(e12, l12, -e01, l01, -e02, l02) + term(-e01, l01, e12, l12, e02, l02)
        f2 = term(-e02, l02, -e12, l12, e01, l01) + term(-e12, l12, -e02, l02, -e01, l01)
        Ftv = jnp.stack([f0, f1, f2], 1)                     # (T, 3, 2)
        return jnp.sum(Ftv[inc_t, inc_s] * inc_m[..., None], axis=1)

    if prestress is None or prestress[0] == "none":
        return f_internal
    kind = prestress[0]
    if kind == "trap":
        kappa, c0 = float(prestress[1]), jnp.asarray(prestress[2])
        return lambda P: f_internal(P) - kappa * (P - c0)
    if kind == "pressure":
        Pr, be = float(prestress[1]), np.asarray(prestress[2])
        bp, bq = jnp.asarray(be[:, 0]), jnp.asarray(be[:, 1])

        def f(P):
            e = P[bq] - P[bp]
            fe = -Pr * jnp.stack([e[:, 1], -e[:, 0]], 1)     # -P * length * n_out
            return f_internal(P).at[bp].add(0.5 * fe).at[bq].add(0.5 * fe)
        return f
    raise ValueError(kind)


def make_prestress(kind, value, pos, tris, c0=None):
    if kind in (None, "none") or value == 0:
        return None
    if kind == "trap":
        return ("trap", value, np.asarray(pos).mean(0) if c0 is None else c0)
    if kind == "pressure":
        return ("pressure", value, boundary_edges(pos, tris))
    raise ValueError(kind)


# =========================================================================== #
# rigid-body bookkeeping and observables (jax)
# =========================================================================== #
def totals(P, F):
    """Net force and torque about the centroid."""
    d = P - P.mean(0)
    return F.sum(0), jnp.sum(d[:, 0] * F[:, 1] - d[:, 1] * F[:, 0])


def kabsch_angle(R0, R):
    X = R0 - R0.mean(0)
    Y = R - R.mean(0)
    return jnp.arctan2(jnp.sum(X[:, 0] * Y[:, 1] - X[:, 1] * Y[:, 0]),
                       jnp.sum(X * Y))


def internal_ke(P, V, m=1.0):
    """Kinetic energy left after removing COM translation and rigid rotation."""
    d = P - P.mean(0)
    Vc = V - V.mean(0)
    Iz = jnp.sum(d ** 2)
    om = jnp.sum(d[:, 0] * Vc[:, 1] - d[:, 1] * Vc[:, 0]) / Iz
    Vr = Vc - om * jnp.stack([-d[:, 1], d[:, 0]], 1)
    return 0.5 * m * jnp.sum(Vr ** 2), om


def corotated(R0, R):
    """R mapped back onto R0's frame (remove translation and best rotation)."""
    th = kabsch_angle(R0, R)
    c, s = jnp.cos(-th), jnp.sin(-th)
    Rm = jnp.array([[c, -s], [s, c]])
    return (R - R.mean(0)) @ Rm.T + R0.mean(0)


def bond_lindemann(R0, R, bi, bj):
    """sqrt< |u_i - u_j|^2 > over bonds, u measured in the co-rotating frame."""
    U = corotated(R0, R) - R0
    du = U[bi] - U[bj]
    return jnp.sqrt(jnp.mean(jnp.sum(du ** 2, 1)))


def psi6_mean(R, bi, bj, n):
    """|psi_6| averaged over particles, on the FROZEN bond network."""
    v = R[bj] - R[bi]
    z = jnp.exp(6j * jnp.arctan2(v[:, 1], v[:, 0]))
    s = jnp.zeros(n, complex).at[bi].add(z).at[bj].add(z)   # z(-v) = z(v) for 6-fold
    c = jnp.zeros(n).at[bi].add(1.0).at[bj].add(1.0)
    return jnp.mean(jnp.abs(s) / jnp.maximum(c, 1.0))


# =========================================================================== #
# step-size from the stiffest mode
# =========================================================================== #
def max_stiffness(force, P, iters=60, seed=0):
    """Largest |eigenvalue| of D = -dF/dR by power iteration on JVPs."""
    key = jax.random.PRNGKey(seed)
    x = jax.random.normal(key, P.shape)
    x = x / jnp.linalg.norm(x)
    jvp = jax.jit(lambda v: jax.jvp(force, (P,), (v,))[1])
    lam = 0.0
    for _ in range(iters):
        y = -jvp(x)
        lam = float(jnp.linalg.norm(y))
        x = y / lam
    return lam


def auto_dt(lam_max, mode, gamma=1.0, m=1.0, safety=0.25):
    if mode == "overdamped":
        return safety * 2.0 * gamma / lam_max           # Euler limit 2 gamma/lam
    w = np.sqrt(lam_max / m)
    return safety * 2.0 / w                             # Verlet limit 2/omega_max


# =========================================================================== #
# relaxation (alpha1 = 0, used for burn-in under prestress)
# =========================================================================== #
def relax_fire(force, P, ftol=1e-10, maxit=200000, dt0=0.05, verbose=False):
    """FIRE minimisation of the force (no energy needed).  Returns (P, max|F|)."""
    dtmax = 10 * dt0

    def cond(s):
        P, V, dt, al, npos, it, fmax = s
        return (fmax > ftol) & (it < maxit)

    def body(s):
        P, V, dt, al, npos, it, _ = s
        F = force(P)
        pw = jnp.sum(F * V)
        fn = jnp.linalg.norm(F) + 1e-300
        vn = jnp.linalg.norm(V)
        V = (1 - al) * V + al * vn * F / fn
        up = pw > 0
        npos = jnp.where(up, npos + 1, 0)
        grow = up & (npos > 5)
        dt = jnp.where(grow, jnp.minimum(dt * 1.1, dtmax), jnp.where(up, dt, dt * 0.5))
        al = jnp.where(grow, al * 0.99, jnp.where(up, al, 0.1))
        V = jnp.where(up, V, 0.0 * V)
        V = V + dt * F
        P = P + dt * V
        return P, V, dt, al, npos, it + 1, jnp.abs(F).max()

    run = jax.jit(lambda P: jax.lax.while_loop(
        cond, body, (P, jnp.zeros_like(P), dt0, 0.1, 0, 0, jnp.inf)))
    P, V, dt, al, npos, it, fmax = run(jnp.asarray(P))
    fmax = float(jnp.abs(force(P)).max())
    if verbose:
        print("   FIRE: %d iterations, max|F| = %.3e" % (int(it), fmax))
    return np.asarray(P), fmax


# =========================================================================== #
# integrators: one jitted chunk of `n` steps, plus observables at its end
# =========================================================================== #
def make_chunk(force, mode, dt, gamma=1.0, m=1.0, kT=0.0):
    """chunk(state, key, n) -> state.  state = (P, V).

    For overdamped runs V is the DRIFT velocity F/gamma (no inertia)."""
    if mode == "overdamped":
        sig = np.sqrt(2 * kT * dt / gamma)

        def body(i, s):
            P, V, key = s
            F = force(P)
            V = F / gamma
            key, sub = jax.random.split(key)
            P = P + V * dt + (sig * jax.random.normal(sub, P.shape) if kT > 0 else 0.0)
            return P, V, key
    else:
        c1 = np.exp(-gamma * dt / m)
        c2 = np.sqrt((1 - c1 ** 2) * kT / m)

        def body(i, s):
            P, V, key = s
            F = force(P)
            V = V + 0.5 * dt * F / m                         # B
            P = P + 0.5 * dt * V                             # A
            key, sub = jax.random.split(key)
            V = c1 * V + (c2 * jax.random.normal(sub, P.shape) if kT > 0 else 0.0)  # O
            P = P + 0.5 * dt * V                             # A
            V = V + 0.5 * dt * force(P) / m                  # B
            return P, V, key

    @jax.jit
    def chunk(P, V, key, n):
        return jax.lax.fori_loop(0, n, body, (P, V, key))
    return chunk


def signed_areas(P, tris):
    a, b, c = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
    return 0.5 * ((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])


def n_flipped(P, P_ref, tris):
    """Triangles whose orientation has reversed relative to P_ref = FOLDING.
    The linear force law has no hard core, so a compressed network can fold
    over itself; real particles cannot.  Any nonzero count invalidates a run."""
    t = jnp.asarray(tris)
    return jnp.sum(jnp.sign(signed_areas(P, t)) != jnp.sign(signed_areas(P_ref, t)))


def make_observe(force, R0, tris, m=1.0, underdamped=True):
    """Cheap per-chunk observables.  Returns dict of scalars."""
    tri_j = jnp.asarray(np.asarray(tris))
    b = bonds_from_tris(np.asarray(tris))
    bi, bj = jnp.asarray(b[:, 0]), jnp.asarray(b[:, 1])
    n = R0.shape[0]
    R0 = jnp.asarray(R0)

    @jax.jit
    def obs(P, V):
        F = force(P)
        Fn, tau = totals(P, F)
        ke, om = internal_ke(P, V, m)
        return dict(com=P.mean(0), theta=kabsch_angle(R0, P), Fx=Fn[0], Fy=Fn[1],
                    tau=tau, ke_int=ke if underdamped else 0.0 * ke, omega_inst=om,
                    vcom=V.mean(0), lind=bond_lindemann(R0, P, bi, bj),
                    psi6=psi6_mean(P, bi, bj, n), fmax=jnp.abs(F).max(),
                    zeta_rot=jnp.sum((P - P.mean(0)) ** 2),
                    flips=jnp.sum(jnp.sign(signed_areas(P, tri_j))
                                  != jnp.sign(signed_areas(R0, tri_j))))
    return obs
