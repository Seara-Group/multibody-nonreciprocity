"""The dynamical matrix inferred from trajectories, its nonreciprocal part, and the surrogate
null (Fig 5c-f and the SI table of ||D-||/||D+|| against the number of modes).

Inference (Methods).  Project the rigid-motion-free displacements onto the M highest-variance
modes V (the softest normal modes), a(t) = U(t) V, take velocities by a Savitzky-Golay
derivative, and fit the phase-space process z = (a, da/dt) as a linear AR(1):
    F = C1 C0^-1 = exp(-G dt),   G = -logm(F) / dt.
The velocity rows of G acting on the positions are the mode-space dynamical matrix,
D = G[M:, :M].  D+ and D- are its symmetric and antisymmetric parts, and the raft-level
nonreciprocity is ||D-|| / ||D+|| (Frobenius norms).

Surrogate null.  Detailed balance makes the cross-spectral matrix S(f) of the mode amplitudes
real.  Surrogate series drawn with S_null(f) = Re S(f) keep every power spectrum and every
reciprocal coupling and remove the directional part; pushed through the same inference they give
the value the estimator returns on reciprocal data with the same spectrum.  The control:
surrogates drawn from the full complex S(f) must give back the measured value.
"""
import warnings

import numpy as np
from scipy.linalg import logm
from scipy.signal import savgol_filter

from . import config, io, modes

SG_WINDOW, SG_ORDER = 9, 3            # Savitzky-Golay velocity
NPERSEG, NOVERLAP = 4096, 3072        # cross spectrum for the surrogates


# ---- norms ---------------------------------------------------------------------------------
def sym(D):
    return 0.5 * (D + D.T)


def anti(D):
    return 0.5 * (D - D.T)


def ratio(D):
    """||D-|| / ||D+||."""
    return float(np.linalg.norm(anti(D)) / np.linalg.norm(sym(D)))


def per_particle(D):
    """Frobenius norm of each particle's rows of a (2N, 2N) matrix (the paper's ||D_i||)."""
    N = D.shape[0] // 2
    return np.sqrt((D.reshape(N, 2, N, 2) ** 2).sum(axis=(1, 2, 3)))


# ---- inference -----------------------------------------------------------------------------
def raft_a_displacements():
    """(T, 2N) float64 displacements of raft A with rigid translation and rotation removed,
    and the mean positions."""
    tr = io.load_raft_a()
    U = modes.displacement_matrix(tr.dx)
    R = modes.rigid_basis(tr.mean_pos)
    return U - (U @ R) @ R.T, tr.mean_pos


def mode_basis(U):
    """Eigenvectors of the equal-time covariance, highest variance (softest) first."""
    w, Q = np.linalg.eigh(np.cov(U, rowvar=False))
    idx = np.argsort(w)[::-1]
    return Q[:, idx], w[idx]


def mode_dynamical_matrix(a, dt=config.DT):
    """M x M mode-space dynamical matrix from mode amplitudes a (T, M)."""
    M = a.shape[1]
    v = savgol_filter(a, SG_WINDOW, SG_ORDER, deriv=1, delta=dt, axis=0)
    z = np.concatenate([a, v], 1)
    z = z - z.mean(0, keepdims=True)
    T = len(z)
    C0 = (z.T @ z) / T
    C1 = (z[1:].T @ z[:-1]) / (T - 1)
    with warnings.catch_warnings():      # logm flags errors of ~1e-13 at large M; harmless
        warnings.filterwarnings("ignore", message="logm result may be inaccurate")
        G = -np.real(logm(C1 @ np.linalg.inv(C0))) / dt
    return G[M:, :M]


def project(D, V):
    """A full (2N, 2N) matrix seen through the modes V: V (V^T D V) V^T."""
    return V @ (V.T @ D @ V) @ V.T


# ---- surrogate null ------------------------------------------------------------------------
def cross_spectrum(a, nperseg=NPERSEG, noverlap=NOVERLAP):
    """Welch-averaged cross-spectral matrix S(f), (nf, M, M), and the number of segments."""
    T, M = a.shape
    w = np.hanning(nperseg)
    S = np.zeros((nperseg // 2 + 1, M, M), dtype=complex)
    ns = 0
    for s in range(0, T - nperseg + 1, nperseg - noverlap):
        X = np.fft.rfft(a[s:s + nperseg] * w[:, None], axis=0)
        S += X[:, :, None] * np.conj(X[:, None, :])
        ns += 1
    return S / ns, ns


def _psd_factor(S):
    """L with L L^H = nearest positive semi-definite matrix to S, and the fraction of the trace
    lost to negative eigenvalues."""
    ev, U = np.linalg.eigh(S)
    lost = np.clip(-ev, 0, None).sum() / np.clip(ev, 0, None).sum()
    S = (U * np.clip(ev, 0, None)[..., None, :]) @ np.conj(np.swapaxes(U, -1, -2))
    ev, U = np.linalg.eigh(S)
    return U * np.sqrt(np.clip(ev, 0, None))[..., None, :], lost


def _surrogate(L, T, rng):
    """A length-T real series whose cross spectrum is L L^H."""
    nfull = T // 2 + 1
    idx = np.linspace(0, L.shape[0] - 1, nfull).round().astype(int)
    M = L.shape[1]
    zr = rng.standard_normal((nfull, M)) + 1j * rng.standard_normal((nfull, M))
    X = np.einsum("fij,fj->fi", L[idx], zr) / np.sqrt(2)
    X[0] = X[0].real
    if T % 2 == 0:
        X[-1] = X[-1].real
    return np.fft.irfft(X, n=T, axis=0)


def surrogate_ratios(a, n, seed, reciprocal=True, dt=config.DT):
    """||D-||/||D+|| of n surrogates of the mode amplitudes a: from Re S(f) (reciprocal=True,
    the null) or from the full S(f) (the control).  Also returns the number of Welch segments
    and the trace fraction lost to the positive-semi-definite projection."""
    S, ns = cross_spectrum(a)
    L, lost = _psd_factor(S.real if reciprocal else S)
    rng = np.random.default_rng(seed)
    out = np.array([ratio(mode_dynamical_matrix(_surrogate(L, len(a), rng), dt)) for _ in range(n)])
    return out, ns, lost


def null_and_control(a, measured, n_null=40, dt=config.DT):
    """The noise floor (n_null reciprocal surrogates, seed 0) and the control (max(8, n_null/4)
    full-spectrum surrogates, seed 1; passes if within 3 sd of the measured value)."""
    null, ns, lost = surrogate_ratios(a, n_null, seed=0, dt=dt)
    ctrl, _, _ = surrogate_ratios(a, max(8, n_null // 4), seed=1, reciprocal=False, dt=dt)
    return dict(null=null, mean=float(null.mean()), sd=float(null.std()),
                lo=float(np.percentile(null, 2.5)), hi=float(np.percentile(null, 97.5)),
                n_seg=int(ns), lost=float(lost), control=ctrl,
                control_mean=float(ctrl.mean()), control_sd=float(ctrl.std()),
                control_passes=bool(abs(ctrl.mean() - measured) < 3 * ctrl.std()))
