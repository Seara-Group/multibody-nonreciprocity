"""Collective motion of a raft: displacement matrix, rigid-motion removal, PCA modes,
winding numbers, cycling rates and phase-randomised surrogates.

Displacements are stored as a (T, 2N) matrix U with columns [x0, y0, x1, y1, ...].
"""
import numpy as np


# ---- displacement matrix and rigid motion -----------------------------------------------
def displacement_matrix(dx):
    """(N, T, 2) displacements -> (T, 2N) float32, columns [x0, y0, x1, y1, ...]."""
    N, T, _ = dx.shape
    return np.nan_to_num(dx).transpose(1, 0, 2).reshape(T, 2 * N).astype(np.float32)


def rigid_basis(mean_pos):
    """Orthonormal (2N, 3) basis of rigid motions: x and y translation, rotation about the centroid."""
    N = len(mean_pos)
    c = mean_pos - mean_pos.mean(0)
    R = np.zeros((2 * N, 3))
    R[0::2, 0] = 1.0
    R[1::2, 1] = 1.0
    R[0::2, 2] = -c[:, 1]
    R[1::2, 2] = c[:, 0]
    return np.linalg.qr(R)[0]


def remove_rigid_motion(U, mean_pos):
    """U with the raft's overall translation and rotation projected out, float32."""
    R = rigid_basis(mean_pos)
    return (U - (U @ R) @ R.T).astype(np.float32)


def internal_displacements(tr):
    """Raft displacements with rigid motion removed and each column's mean subtracted."""
    U = remove_rigid_motion(displacement_matrix(tr.dx), tr.mean_pos)
    U -= U.mean(0, keepdims=True)
    return U


# ---- PCA ----------------------------------------------------------------------------------
def pca(U, n_modes):
    """Top-variance eigenvectors of the equal-time covariance of (centred) U.

    Returns V (2N, n_modes), the variance of each mode, and the total variance.  The
    covariance is accumulated in single precision and then converted, as in the published
    analysis, so the modes (and their signs) are reproduced exactly.
    """
    T = len(U)
    Sig = (U.T @ U).astype(np.float64) / T
    w, Q = np.linalg.eigh(Sig)
    idx = np.argsort(w)[::-1][:n_modes]
    return Q[:, idx], w[idx], float(w.sum())


def mode_amplitudes(U, V, dt):
    """Mode amplitudes a (T, M) and their time derivatives (central differences)."""
    a = (U @ V.astype(np.float32)).astype(np.float64)
    return a, np.gradient(a, dt, axis=0)


# ---- winding numbers ----------------------------------------------------------------------
def winding(X, Y, dt, sub=60, chunk=64):
    """Winding number of the point (X, Y) about the origin, column by column.

    The phase atan2(Y, X) is unwrapped at the full sampling rate (the step between frames must
    stay below pi); only the finished curves are decimated by `sub`.
    Returns t[::sub], turns (n_columns, n_t) float32, and the mean winding rate in Hz.
    """
    T, K = X.shape
    t = np.arange(T) * dt
    turns = np.empty((K, len(t[::sub])), dtype=np.float32)
    rate = np.empty(K)
    for s in range(0, K, chunk):
        ph = np.unwrap(np.arctan2(Y[:, s:s + chunk], X[:, s:s + chunk]), axis=0)
        ph -= ph[0]
        rate[s:s + chunk] = np.polyfit(t, ph, 1)[0] / (2 * np.pi)
        turns[s:s + chunk] = (ph[::sub] / (2 * np.pi)).T
    return t[::sub], turns, rate


def particle_winding(Uc, dt, sub=60):
    """Each particle's winding about its mean position, x and y scaled to unit variance."""
    ux = Uc[:, 0::2] / Uc[:, 0::2].std(0)
    uy = Uc[:, 1::2] / Uc[:, 1::2].std(0)
    return winding(ux, uy, dt, sub, chunk=ux.shape[1])


def pair_winding(a, dt, sub=60, chunk=64):
    """Winding of every mode pair (m < n) in the plane (a_m, a_n), amplitudes scaled to unit variance."""
    ah = a / a.std(0)
    M = ah.shape[1]
    pairs = [(m, n) for m in range(M) for n in range(m + 1, M)]
    turns, rate = [], []
    for s in range(0, len(pairs), chunk):
        blk = pairs[s:s + chunk]
        t, w, r = winding(ah[:, [m for m, _ in blk]], ah[:, [n for _, n in blk]], dt, sub, chunk)
        turns.append(w)
        rate.append(r)
    return pairs, t, np.concatenate(turns), np.concatenate(rate)


# ---- cycling rates ---------------------------------------------------------------------------
def cycling_matrix(a, adot):
    """omega_mn = <a_m adot_n - a_n adot_m> / (2 sigma_m sigma_n), rad/s."""
    T = len(a)
    s = a.std(0)
    C = (a.T @ adot) / T
    return (C - C.T) / (2.0 * np.outer(s, s))


def cycling_spectrum(om):
    """Positive members of the +- i lambda eigenvalue pairs of the antisymmetric matrix, sorted."""
    ev = np.linalg.eigvals(0.5 * (om - om.T))
    return np.sort(np.abs(ev.imag))[::-1][::2]


# ---- phase-randomised surrogates -----------------------------------------------------------
def phase_randomized(F, n, rng, dtype=np.complex128):
    """One surrogate from the Fourier transform F = rfft(X, axis=0) of a (n, K) signal.

    Every column gets its own random phase at every frequency (zero frequency untouched): each
    column keeps its power spectrum exactly, and the timing between columns becomes random.
    """
    ph = np.exp(1j * rng.uniform(0, 2 * np.pi, F.shape)).astype(dtype)
    ph[0] = 1.0
    return np.fft.irfft(F * ph, n=n, axis=0)


def particle_winding_null(Uc, dt, n_null, sub=60, seed=0):
    """Winding curves and rates of phase-randomised particles (x and y randomised independently)."""
    T = len(Uc)
    F = np.fft.rfft(Uc.astype(np.float64), axis=0).astype(np.complex64)
    rng = np.random.default_rng(seed)
    curves, rates = [], []
    for _ in range(n_null):
        S = phase_randomized(F, T, rng, np.complex64).astype(np.float32)
        _, w, r = particle_winding(S, dt, sub)
        curves.append(w)
        rates.append(r)
    return np.concatenate(curves), np.concatenate(rates)


def cycling_null(a, dt, n_null, seed=0):
    """Cycling matrices and their spectra for phase-randomised mode amplitudes."""
    T, M = a.shape
    F = np.fft.rfft(a / a.std(0), axis=0)
    rng = np.random.default_rng(seed)
    om = np.empty((n_null, M, M), dtype=np.float32)
    lam = np.empty((n_null, M // 2))
    for b in range(n_null):
        s = phase_randomized(F, T, rng)
        O = cycling_matrix(s, np.gradient(s, dt, axis=0))
        om[b] = O.astype(np.float32)
        lam[b] = cycling_spectrum(O)[:M // 2]
    return om, lam


# ---- specific angular momentum of each particle ---------------------------------------------
def angular_momentum(dx, dt, frames=None):
    """Time-averaged L_i = <u_x v_y - u_y v_x> about each particle's mean position, um^2/s.

    u is the displacement, v its central-difference derivative; `frames` limits the average
    (e.g. slice(0, 29999)).
    """
    u = dx.astype(np.float64)
    v = np.gradient(u, dt, axis=1)
    L = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    return (L if frames is None else L[:, frames]).mean(1)


def angular_momentum_null(dx, dt, n_null=20, seed=0):
    """L_i of phase-randomised copies of each particle's x and y (raw displacements), (n_null, N)."""
    U = displacement_matrix(dx).astype(np.float64)
    U -= U.mean(0)
    T = len(U)
    F = np.fft.rfft(U, axis=0)
    rng = np.random.default_rng(seed)
    out = np.empty((n_null, U.shape[1] // 2))
    for b in range(n_null):
        S = phase_randomized(F, T, rng)
        Sd = np.gradient(S, dt, axis=0)
        out[b] = (S[:, 0::2] * Sd[:, 1::2] - S[:, 1::2] * Sd[:, 0::2]).mean(0)
    return out
