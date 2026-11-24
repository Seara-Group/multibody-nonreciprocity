"""COMSOL force Jacobians of a perfect triangular lattice, and the spring-constant ratio from them.

DATA/force_jacobian/force_jacobian<run>_.mat holds the forces on every particle of a perfect
lattice with the central particle at rest and displaced in x and in y.  Translational invariance
turns that one column into the full dynamical matrix:  D_n = -(F_n(displaced) - F_n(static)) / h.

kappa = k_phiphi / k_rr two ways:
  bond level  ratio of the transverse to longitudinal stiffness of the first shell (symmetric part)
  effective   D(q) = sum_{|x_n| < 1.85 a} D_n (exp(i q.x_n) - 1), averaged over 12 directions at
              q a = 0.2, 0.35, 0.5, with lambda_T / lambda_L = (1 + 3 kappa)/(3 + kappa) inverted
              -- the nearest-neighbour-equivalent kappa including the 2nd shell (sqrt(3) a).
"""
import os

import numpy as np
from scipy.io import loadmat

from . import config


def jacobian_column(fn):
    d = loadmat(fn)
    P, F = d["poss"], d["fforce"]
    pos0, F0 = P[0][0][:, :2].astype(float), F[0][0][:, :2].astype(float)
    c = int(np.argmin(np.linalg.norm(pos0 - pos0.mean(0), axis=1)))
    cols = []
    for b in (1, 2):
        h = (P[b][0][:, :2].astype(float) - pos0)[c, b - 1]
        cols.append(-(F[b][0][:, :2].astype(float) - F0) / h)
    return pos0 - pos0[c], np.stack(cols, axis=-1)          # x_n, D_n (N, 2, 2)


def kappa_values(fn, rc_eff=1.85):
    """(kappa bond level, kappa effective, lattice spacing a in um) of one Jacobian file."""
    x, Dn = jacobian_column(fn)
    r = np.linalg.norm(x, axis=1)
    a = np.sort(r)[1]
    num = den = 0.0
    for n in np.where((r > 0.5 * a) & (r < 1.3 * a))[0]:
        e = x[n] / r[n]
        t = np.array([-e[1], e[0]])
        S = 0.5 * (Dn[n] + Dn[n].T)
        num += -(t @ S @ t)
        den += -(e @ S @ e)
    m = r < rc_eff * a
    ks = []
    for qa in (0.2, 0.35, 0.5):
        lL = lT = 0.0
        for th in np.arange(12) * np.pi / 6 + 0.05:
            nn = np.array([np.cos(th), np.sin(th)])
            tt = np.array([-nn[1], nn[0]])
            Dq = np.einsum("n,nab->ab", np.exp(1j * (x[m] @ (qa / a * nn))) - 1, Dn[m])
            Dq = 0.5 * (Dq + Dq.conj().T).real
            lL += nn @ Dq @ nn
            lT += tt @ Dq @ tt
        rat = lT / lL
        ks.append((3 * rat - 1) / (3 - rat))
    return num / den, float(np.mean(ks)), a


def kappa_with_spread(run):
    """kappa (bond, effective) for the base run and its _095 / _105 variants (numerical spread)."""
    jd = os.path.join(config.DATA, "force_jacobian")
    base = kappa_values(os.path.join(jd, f"force_jacobian{run}_.mat"))
    var = [kappa_values(os.path.join(jd, f"force_jacobian_{v}_{run}_.mat")) for v in ("095", "105")]
    kb = [base[0]] + [v[0] for v in var]
    ke = [base[1]] + [v[1] for v in var]
    return dict(kb=base[0], kb_lo=min(kb), kb_hi=max(kb), ke=base[1], ke_lo=min(ke), ke_hi=max(ke), a=base[2])


# ---- the COMSOL dynamical matrix at the experimental configuration (SI stripes d-f) ------------
HZ = 1.65e8          # Hz per sqrt(eigenvalue) unit (COMSOL forces x 1e12, particle mass)


def experimental_dynamical_matrix(neighbours=50):
    """COMSOL dynamical matrix of raft A's measured configuration, in the convention of the COMSOL export.

    DATA/comsol_dynamical_matrix.mat: forces on every particle at rest (fforce[0]) and with
    particle p displaced in x (fforce[1, p]) or y (fforce[2, p]).  Row 2p + dim holds minus the
    force change on every particle when p is displaced along dim, i.e. the TRANSPOSE of
    D_{2i+mu, 2j+nu} = -dF_i^mu/dx_j^nu.  Eigenvalues (frequencies, growth rates) are those of D;
    eigenvectors are D's left eigenvectors -- the ones that project measured motion onto modes.
    `neighbours`: keep only the force on each displaced particle's nearest 50 (as in the COMSOL analysis;
    reproduces the COMSOL analysis plots).  Returns (positions in COMSOL order, matrix).
    """
    d = loadmat(os.path.join(config.DATA, "comsol_dynamical_matrix.mat"))
    lat, ff = d["pos"], d["fforce"]
    N = len(lat)
    D = np.zeros((2 * N, 2 * N))
    for p in range(N):
        for dim in (1, 2):
            f = (ff[dim, p] - ff[0, p]).copy()
            if neighbours:
                dis = np.linalg.norm(lat - lat[p], axis=1)
                f[~(dis < sorted(dis)[neighbours])] = 0
            D[2 * p + dim - 1, 0::2] = -f[:, 0]
            D[2 * p + dim - 1, 1::2] = -f[:, 1]
    return lat, D


def displacements_comsol_order(tr):
    """Raft A displacements (T, 2N) in the particle order of the COMSOL file, mean removed."""
    perm = np.load(os.path.join(config.DATA, "comsol_exp_permutation.npz"), allow_pickle=True)["perm_exp_to_comsol"]
    U = np.zeros((tr.n_frames, 2 * tr.n_particles), np.float32)
    U[:, 2 * perm] = tr.dx[:, :, 0].T
    U[:, 2 * perm + 1] = tr.dx[:, :, 1].T
    return U - U.mean(0)


def eigenmodes(D):
    """Signed frequencies (sqrt of |eigenvalue|, negative for negative real part), eigenvectors,
    eigenvalues and growth rates, sorted by frequency."""
    ev, vec = np.linalg.eig(D)
    re, im = ev.real, ev.imag
    freq = np.sqrt((abs(re) + np.sqrt(re ** 2 + im ** 2)) / 2)
    freq[re < 0] = -freq[re < 0]
    o = np.argsort(freq)
    return freq[o], vec[:, o], ev[o], np.imag(ev[o]) / (-2 * freq[o])


def mode_power(U, B, dt, nper=4096, hop=1024, fmax=1400.0):
    """Power of the mode amplitudes U(t) @ B (Hann windows of nper frames every hop), +f and -f
    folded.  Returns f (Hz) and P (n_modes, n_f)."""
    w = np.hanning(nper).astype(np.float32)
    f = np.fft.fftfreq(nper, dt)
    B = B.astype(np.complex64)
    P = 0
    for s in range(0, U.shape[0] - nper + 1, hop):
        P = P + np.abs(np.fft.fft((U[s:s + nper] * w[:, None]) @ B, axis=0)) ** 2
    pos, neg = (f > 0) & (f <= fmax), (f < 0) & (f >= -fmax)
    return f[pos], (P[pos] + P[neg][::-1][:pos.sum()]).T
