"""The minimal three-body force law (Eq. minimal_model of the paper).

    F_i = sum_{(j,k)} a_ijk rhat_ij,
    a_ijk = a0 + k_a dr_ij + a3 (dr_ik + dr_jk) + alpha3 (dr_ik - dr_jk),

with dr = |r| - a and rhat_ij the unit vector from i to j.  The sum runs over ORDERED pairs
(j, k) such that {i, j, k} is a nearest-neighbour triangle: a Delaunay triangle whose three
sides are all shorter than rc = rc_factor * a.  The rule depends on the triple only, so it is
the same for all three particles of a triangle.  (a3 and alpha3 are called a1 and alpha1 in
older code.)  alpha3 is the only nonreciprocal coefficient: with alpha3 = 0 every triangle's
forces sum to zero.

Coefficients are dicts with keys a, a0, k_a, a3, alpha3, in network force units (N -> 1e-19)
and microns.

Two uses:
  * a finite raft (Fig 5): `make_force` freezes the triangles of a reference configuration,
    `dynamical_matrix` differentiates the forces, `build_raft` makes a free disc with or
    without an edge dislocation;
  * the infinite lattice (Fig 4b): `lattice_branches`, from the force constants of the
    central particle, D(q) = sum_R Phi(R) exp(i q.R).  D(q) is not Hermitian when alpha3 != 0.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

SQ3 = np.sqrt(3.0)

# Expanded about a = 54.01 um (the median nearest-neighbour bond of raft A), measured from the
# COMSOL phase scan by levitation.trimers.coefficients (figures/fig3.py checks they agree).
# Force units N x 1e19, lengths um.  The reference spacing stays 54.01 um (decided 2026-11-24): the
# coefficients are a first-order expansion and their predictions depend on where it is taken.
COEFFS_54 = dict(a=54.01, a0=-0.318436313801337, k_a=0.021335790461839085,
                 a3=0.001074403581232237, alpha3=8.117636048464799e-05)

RC_FACTOR = 1.45          # nearest-neighbour triangles: all sides < 1.45 a


# ---- the force law -------------------------------------------------------------------------
def triangles(pos, rc):
    """Delaunay triangles whose three sides are all shorter than rc."""
    P = np.asarray(pos)
    sm = Delaunay(P).simplices
    e01 = np.linalg.norm(P[sm[:, 0]] - P[sm[:, 1]], axis=1)
    e12 = np.linalg.norm(P[sm[:, 1]] - P[sm[:, 2]], axis=1)
    e20 = np.linalg.norm(P[sm[:, 2]] - P[sm[:, 0]], axis=1)
    return np.ascontiguousarray(sm[np.maximum(np.maximum(e01, e12), e20) < rc])


def ordered_terms(tris):
    """(I, J, K): every vertex of every triangle as i, with both orders of the other two."""
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    ijk = [(v0, v1, v2), (v0, v2, v1), (v1, v2, v0), (v1, v0, v2), (v2, v0, v1), (v2, v1, v0)]
    return tuple(np.concatenate([t[n] for t in ijk]) for n in range(3))


def forces(pos, terms, c):
    """(N, 2) forces of the minimal model for the frozen triangle terms (I, J, K)."""
    P = np.asarray(pos, dtype=float)
    I, J, K = terms
    rij = P[J] - P[I]
    nij = np.linalg.norm(rij, axis=1)
    dij = nij - c["a"]
    dik = np.linalg.norm(P[K] - P[I], axis=1) - c["a"]
    djk = np.linalg.norm(P[K] - P[J], axis=1) - c["a"]
    A = c["a0"] + c["k_a"] * dij + c["a3"] * (dik + djk) + c["alpha3"] * (dik - djk)
    F = np.zeros_like(P)
    np.add.at(F, I, A[:, None] * (rij / nij[:, None]))
    return F


def make_force(pos_ref, c, rc_factor=RC_FACTOR):
    """Force function of position, with the triangles frozen from pos_ref (so the force stays
    smooth when a displacement would make a triangle cross the cutoff)."""
    terms = ordered_terms(triangles(pos_ref, rc_factor * c["a"]))
    return lambda p: forces(p, terms, c)


def dynamical_matrix(pos, force_fn, h):
    """D[2i+a, 2j+b] = -dF_i^a / du_j^b by central differences (step h)."""
    P = np.asarray(pos, dtype=float)
    N = len(P)
    D = np.zeros((2 * N, 2 * N))
    for j in range(N):
        for b in range(2):
            p = P.copy(); p[j, b] += h
            Fp = force_fn(p)
            p = P.copy(); p[j, b] -= h
            Fm = force_fn(p)
            D[:, 2 * j + b] = -((Fp - Fm) / (2 * h)).ravel()
    return D


def raft_dynamical_matrix(pos, c, rc_factor=RC_FACTOR):
    return dynamical_matrix(pos, make_force(pos, c, rc_factor), h=1e-5 * c["a"])


# ---- finite rafts --------------------------------------------------------------------------
def tri_disk(R, a):
    """Triangular-lattice disc of radius R (length units), centred on the origin."""
    n = int(R / a) + 3
    pos = np.array([[i * a + (a / 2 if j % 2 else 0.0), j * (SQ3 / 2) * a]
                    for j in range(-n, n + 1) for i in range(-n, n + 1)])
    pos = pos - pos.mean(0)
    return pos[np.linalg.norm(pos, axis=1) < R]


def edge_disp(d, b, nu=1.0 / 3):
    """Continuum displacement of an edge dislocation at the origin, Burgers vector b along +x.
    The log is taken of r / |b|, so the construction does not depend on the length unit."""
    core = abs(b)
    x, y = d[:, 0], d[:, 1]
    r2 = x * x + y * y + 1e-9 * core * core
    th = np.arctan2(y, x)
    ux = (b / (2 * np.pi)) * (th + x * y / (2 * (1 - nu) * r2))
    uy = -(b / (2 * np.pi)) * ((1 - 2 * nu) / (4 * (1 - nu)) * np.log(r2 / core ** 2)
                               + (x * x - y * y) / (4 * (1 - nu) * r2))
    return np.stack([ux, uy], 1)


def short_bonds(pos, rc):
    """Delaunay edges shorter than rc, sorted (E, 2) with i < j."""
    s = set()
    for sm in Delaunay(pos).simplices:
        for u in range(3):
            for v in range(u + 1, 3):
                p, q = sm[u], sm[v]
                if np.linalg.norm(pos[p] - pos[q]) < rc:
                    s.add((min(p, q), max(p, q)))
    return np.array(sorted(s))


def relax_springs(pos, a, pin, iters=5, rc_factor=RC_FACTOR):
    """Relax under unit springs of rest length a on the nearest-neighbour bonds, with the
    particles in `pin` held fixed.  Done in units of a, so the result does not depend on the
    length unit (L-BFGS-B's stopping tests are absolute)."""
    a = float(a)
    P = np.asarray(pos, dtype=float) / a
    B = short_bonds(P, rc_factor)
    free = np.ones(len(P), bool)
    free[pin] = False
    fi = np.where(free)[0]
    base = P.copy()

    def EG(xf):
        p = base.copy()
        p[fi] = xf.reshape(-1, 2)
        d = p[B[:, 0]] - p[B[:, 1]]
        r = np.linalg.norm(d, axis=1)
        dr = r - 1.0
        g = np.zeros_like(p)
        fv = (2 * dr / r)[:, None] * d
        np.add.at(g, B[:, 0], fv)
        np.add.at(g, B[:, 1], -fv)
        return float(np.sum(dr ** 2)), g[fi].ravel()

    for _ in range(iters):
        r = minimize(EG, base[fi].ravel(), jac=True, method="L-BFGS-B",
                     options={"maxiter": 400, "ftol": 1e-15, "gtol": 1e-12})
        base[fi] = r.x.reshape(-1, 2)
        B = short_bonds(base, rc_factor)
    return base * a


def build_raft(R, a, dislocation):
    """Free disc of radius R (lattice units), optionally with one edge dislocation at the
    centre (Burgers vector a along +x), relaxed under springs with the outer 1.5 a held."""
    lat = tri_disk(R * a, a)
    rr = np.linalg.norm(lat, axis=1)
    pin = np.where(rr > rr.max() - 1.5 * a)[0]
    start = lat + edge_disp(lat, +a) if dislocation else lat.copy()
    return relax_springs(start, a, pin)


# ---- the infinite lattice ------------------------------------------------------------------
def lattice(a, n):
    """Triangular lattice with site c at the origin, n cells each way."""
    a1, a2 = np.array([a, 0.0]), np.array([a / 2, a * SQ3 / 2])
    R = np.array([i * a1 + j * a2 for i in range(-n, n + 1) for j in range(-n, n + 1)])
    c = int(np.argmin(np.linalg.norm(R, axis=1)))
    return R - R[c], c


def closed_triples(R, c, rcut):
    """Pairs (j, k) such that {c, j, k} has every side shorter than rcut."""
    d = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
    nb = [m for m in np.where(d[c] < rcut)[0] if m != c]
    return [(nb[x], nb[y]) for x in range(len(nb)) for y in range(x + 1, len(nb)) if d[nb[x], nb[y]] < rcut]


def force_constants(c, rcut, n=4, h=1e-5):
    """Phi(R) = -dF_0/du_R for the central particle of a perfect lattice: {m: 2x2 block}."""
    R, ic = lattice(c["a"], n)
    trips = closed_triples(R, ic, rcut)
    I = np.array([ic] * (2 * len(trips)))
    J = np.array([x for j, k in trips for x in (j, k)])
    K = np.array([x for j, k in trips for x in (k, j)])
    f0 = lambda P: forces(P, (I, J, K), c)[ic]
    Phi = {}
    for m in sorted({ic} | set(J.tolist())):
        blk = np.zeros((2, 2))
        for v in range(2):
            P = R.copy(); P[m, v] += h
            Fp = f0(P)
            P = R.copy(); P[m, v] -= h
            blk[:, v] = -(Fp - f0(P)) / (2 * h)
        Phi[m] = blk
    return R, ic, Phi


def Dq(q, R, c, Phi):
    D = np.zeros((2, 2), dtype=complex)
    for m, blk in Phi.items():
        D += blk * np.exp(1j * np.dot(q, R[m] - R[c]))
    return D


def lattice_branches(qa, coeffs=COEFFS_54, rcut_a=1.05, directions=None):
    """omega(|q| a) of the two branches (model units), mean / min / max over `directions`
    (angles relative to the lattice; default: 48 directions across the 60-degree wedge).
    Columns: 0 transverse, 1 longitudinal (eigenvalues sorted)."""
    a = coeffs["a"]
    R, c, Phi = force_constants(coeffs, rcut_a * a)
    angs = np.linspace(0, np.pi / 3, 48) if directions is None else np.asarray(directions)
    out = np.zeros((len(qa), len(angs), 2))
    for i, x in enumerate(qa):
        for j, th in enumerate(angs):
            ev = np.sort(np.linalg.eigvals(Dq(x / a * np.array([np.cos(th), np.sin(th)]), R, c, Phi)).real)
            out[i, j] = np.sqrt(np.abs(ev)) * np.sign(ev)
    return out.mean(1), out.min(1), out.max(1)
