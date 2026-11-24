"""Velocity-flux power spectra of a raft.

The collective current at wavevector k is J(k, t) = sum_j v_j(t) exp(-i k . R_j), with R_j the
particles' mean positions and v_j their velocities (central differences).  Its longitudinal and
transverse parts are J_L = khat . J and J_T = khat_perp . J, and their Welch power spectra give
S_L(k, f) and S_T(k, f).  A line that appears at every k is a raft-wide oscillation; a ridge that
moves with k is a phonon branch.
"""
import numpy as np
from scipy.signal import welch


def velocities(dx, dt):
    """(N, T, 2) central-difference velocities, float32."""
    return np.gradient(dx, dt, axis=1).astype(np.float32)


def two_sided_welch(x, dt, nperseg, noverlap=None, axis=-1):
    """Two-sided Welch PSD (Hann window, mean removed per segment), frequencies sorted."""
    f, P = welch(x, fs=1.0 / dt, window="hann", nperseg=nperseg,
                 noverlap=nperseg // 2 if noverlap is None else noverlap, detrend="constant",
                 return_onesided=False, scaling="density", axis=axis)
    order = np.argsort(f)
    return f[order], np.take(P, order, axis=axis)


def flux_spectra(mean_pos, v, dt, kvecs, nperseg, noverlap=None, kchunk=96):
    """S_L(k, f), S_T(k, f) (two-sided) for a list of wavevectors (1/um)."""
    nk = len(kvecs)
    khat = kvecs / np.maximum(np.linalg.norm(kvecs, axis=1, keepdims=True), 1e-12)
    kperp = np.stack([-khat[:, 1], khat[:, 0]], axis=1)
    fs = SL = ST = None
    for c0 in range(0, nk, kchunk):
        kk = kvecs[c0:c0 + kchunk]
        ph = np.exp(-1j * (kk @ mean_pos.T)).astype(np.complex64)
        Jx, Jy = ph @ v[:, :, 0], ph @ v[:, :, 1]
        sl = slice(c0, c0 + kchunk)
        JL = khat[sl, 0:1] * Jx + khat[sl, 1:2] * Jy
        JT = kperp[sl, 0:1] * Jx + kperp[sl, 1:2] * Jy
        f, pL = two_sided_welch(JL, dt, nperseg, noverlap)
        _, pT = two_sided_welch(JT, dt, nperseg, noverlap)
        if SL is None:
            fs, SL, ST = f, np.empty((nk, len(f))), np.empty((nk, len(f)))
        SL[sl], ST[sl] = pL.real, pT.real
    return fs, SL, ST


def raft_spectra(tr, ell, nperseg=4096, kmax=5.0, nk1d=101, nkxy=41, nkbin=30):
    """Spectra of raft A along k_y = 0 and averaged around rings of constant |k|.

    The k-grid for the ring average is built exactly symmetric about zero (a ring must contain
    -k for every k).  The |k| = 0 bin is left out: the k = 0 current is the total momentum, which
    was removed from the trajectories.  Returns a dict of arrays (k in 1/um, f in Hz).
    """
    v = velocities(np.ascontiguousarray(tr.dx, dtype=np.float32), tr.dt)
    mean_pos = np.asarray(tr.mean_pos, dtype=np.float64)
    kx = np.linspace(-kmax, kmax, nk1d) / ell
    f_cut, SL_cut, ST_cut = flux_spectra(mean_pos, v, tr.dt, np.stack([kx, 0 * kx], 1), nperseg)
    nhalf = nkxy // 2
    half = np.arange(1, nhalf + 1) * (kmax / max(nhalf, 1))
    kv1 = np.concatenate([-half[::-1], [0.0], half]) / ell
    KX, KY = np.meshgrid(kv1, kv1)
    kgrid = np.stack([KX.ravel(), KY.ravel()], axis=1)
    kmag = np.linalg.norm(kgrid, axis=1)
    f_az, SL_g, ST_g = flux_spectra(mean_pos, v, tr.dt, kgrid, nperseg)
    dk = (kmax / ell) / nkbin
    kbin = (np.arange(nkbin) + 0.5) * dk
    SL_az = np.full((nkbin, SL_g.shape[1]), np.nan)
    ST_az = np.full_like(SL_az, np.nan)
    counts = np.zeros(nkbin, dtype=int)
    for i, km in enumerate(kbin):
        m = (kmag >= km - 0.5 * dk) & (kmag < km + 0.5 * dk)
        counts[i] = int(m.sum())
        if m.any():
            SL_az[i], ST_az[i] = SL_g[m].mean(0), ST_g[m].mean(0)
    nseg = max(1, tr.n_frames // (nperseg // 2) - 1)
    return dict(kx_cut=kx, k_az=kbin, f_cut=f_cut, f_az=f_az, SL_cut=SL_cut, ST_cut=ST_cut,
                SL_az=SL_az, ST_az=ST_az, a_lat=ell, dt=tr.dt, nperseg=nperseg, n_segments=nseg,
                counts_per_bin=counts)


def raft_a_spectra(refresh=False):
    """raft_spectra() of raft A on the ell grid, cached in OUT/derived/raft_a_spectra.npz."""
    import os
    from . import config, io, lattice
    path = os.path.join(config.DERIVED, "raft_a_spectra.npz")
    if os.path.exists(path) and not refresh:
        return dict(np.load(path))
    tr = io.load_raft_a()
    out = raft_spectra(tr, lattice.lattice_spacing(tr.mean_pos))
    config.ensure_dirs(config.DERIVED)
    np.savez(path, **out)
    return out
