"""Measured dispersion of each raft and its fit by the nearest-neighbour spring lattice.

Spectra.  For each raft, velocity-flux power spectra S_L(k, f), S_T(k, f) are computed in
Welch segments of SEG frames, in the raft's co-rotating lattice frame, along the Gamma-K and
Gamma-M directions (12 rays, averaged into the two classes), plus a null with the particle
positions permuted within each frame.  Segments are cached in
DATA/_raft_spectra_cache/<raft>/seg_*.npz.  The published figures and table use exactly the
segments cached on 2026-10-05; `compute_segments` adds new ones (and so changes the averages).

Wavevector units.  The cached spectra sit on the grid k = KA / a_closest, where a_closest is
the raft's median closest-neighbour distance (pixels); ridge points and fits are in units of
a_closest, as published.  The sound speeds use the raft-average spacing (rafts.spacings).

Ridge points (peak in k at fixed f, 20 Hz bands from 30 to 900 Hz) enter the fit if
  * the peak is not at the zone edge,
  * the first-half and second-half averages of the segments agree within 0.15 in k a,
  * its contrast (peak / median over k) is above 1.3 x that of the permuted null,
  * for the longitudinal branch, f >= 170 Hz (below that the peak is inside the finite-size
    aperture).
Fit: f = s sqrt(lambda(k a; kappa)) with D(q) = 2 sum_n [n n + kappa (I - n n)](1 - cos q.n),
a grid search over kappa in (-1/3, 0.3] with s solved in closed form (log-space least squares).
"""
import glob
import json
import os

import numpy as np

from . import config, rafts

SEG = 4096
KA = np.arange(0.10, 4 * np.pi / 3 + 1e-9, 0.05)           # cached grid, units of a_closest
EDGE = {"K": 4 * np.pi / 3, "M": 2 * np.pi / np.sqrt(3)}    # zone edges, units of 1/a_closest
DIRS = {"K": 0.0, "M": np.pi / 6}
NV = np.array([[1, 0], [0.5, np.sqrt(3) / 2], [-0.5, np.sqrt(3) / 2]])
BANDS = [(lo, lo + 20) for lo in range(30, 900, 20)]
SUBSETS = ("all", "L + T(ka>1)", "T only", "T, ka>1")


def freqs():
    return np.fft.fftfreq(SEG, config.DT)[1:SEG // 2]


def cache_dir(tag):
    return os.path.join(config.DATA, "_raft_spectra_cache", tag)


# ---- spring lattice ---------------------------------------------------------------------------
def lam(q, th, kappa, branch):
    """Eigenvalue of the NN spring-lattice dynamical matrix along direction th, at q = |k| a_closest."""
    qv = q * np.array([np.cos(th), np.sin(th)])
    D = np.zeros((2, 2))
    for n in NV:
        nn = np.outer(n, n)
        D += 2 * (nn + kappa * (np.eye(2) - nn)) * (1 - np.cos(qv @ n))
    e = np.array([np.cos(th), np.sin(th)]) if branch == "L" else np.array([-np.sin(th), np.cos(th)])
    return e @ D @ e


def _lam_parts(points):
    """lambda = A + kappa B for each ridge point: e.D.e with D linear in kappa."""
    A, B = np.empty(len(points)), np.empty(len(points))
    for i, (br, c, q, _) in enumerate(points):
        th = DIRS[c]
        e = np.array([np.cos(th), np.sin(th)]) if br == "L" else np.array([-np.sin(th), np.cos(th)])
        w = 2 * (1 - np.cos(q * (NV @ np.array([np.cos(th), np.sin(th)]))))   # per neighbour vector
        en2 = (NV @ e) ** 2
        A[i], B[i] = (w * en2).sum(), (w * (1 - en2)).sum()
    return A, B


def fit(points, kappas=np.linspace(-0.333, 0.3, 634)):
    """Best kappa and frequency scale s (Hz) for ridge points [(branch, dir, k a, f)].

    Grid search over kappa; for each, s is the geometric-mean solution of
    ln f = ln s + 0.5 ln lambda(k a; kappa), and the cost is the variance of the residual.
    """
    A, B = _lam_parts(points)
    lnf = np.log([p[3] for p in points])
    r = lnf[None, :] - 0.5 * np.log(np.maximum(A[None, :] + kappas[:, None] * B[None, :], 1e-12))
    cost = ((r - r.mean(1, keepdims=True)) ** 2).mean(1)
    j = int(np.argmin(cost))
    return dict(k=float(kappas[j]), scale_Hz=float(np.exp(r[j].mean())),
                rms_lnf=float(np.sqrt(cost[j])), n=len(points))


def speeds(kappa, scale_Hz, ell_um):
    """Long-wavelength sound speeds (mm/s) and Poisson ratio of the fitted lattice."""
    c0 = 2 * np.pi * scale_Hz * ell_um * 1e-3                  # ell sqrt(k_r/m)
    cL = c0 * np.sqrt((9 + 3 * kappa) / 8)
    cT = c0 * np.sqrt(max((3 + 9 * kappa) / 8, 0.0))
    return dict(c0=c0, cL=cL, cT=cT, nu=(1 - 5 * kappa) / (3 + kappa))


# ---- ridge points -----------------------------------------------------------------------------
def ridge(q, S, f):
    """Peak position in q at fixed frequency, one per band (parabolic refinement in log S)."""
    m = q >= 0.15
    kk = q[m]
    out = []
    for lo, hi in BANDS:
        prof = S[m][:, (f >= lo) & (f < hi)].mean(1)
        i = int(np.argmax(prof))
        if 0 < i < len(prof) - 1:
            y0, y1, y2 = np.log(prof[i - 1:i + 2])
            den = y0 - 2 * y1 + y2
            kp, edge = kk[i] + (0.5 * (y0 - y2) / den if den else 0) * (kk[1] - kk[0]), False
        else:
            kp, edge = kk[i], True
        out.append(dict(lo=lo, hi=hi, ka=float(kp), contrast=float(prof.max() / np.median(prof)),
                        at_edge=edge))
    return out


def load_segments(tag):
    fs = sorted(glob.glob(os.path.join(cache_dir(tag), "seg_*.npz")))
    return [np.load(x) for x in fs]


def analyse(tag, ell_over_closest, nboot=200):
    """Ridge points and fits of one raft; wavevectors are the cached k a_closest times ell_over_closest."""
    D = load_segments(tag)
    if not D:
        return None
    q = D[0]["ka"] * ell_over_closest
    f = freqs()
    h = len(D) // 2
    avg = lambda key, sl: np.mean([d[key] for d in D[sl]], 0)
    pts, rows = [], {}
    for br in ("L", "T"):
        full, h1, h2 = avg("S" + br, slice(None)), avg("S" + br, slice(0, h)), avg("S" + br, slice(h, None))
        nul = avg("S" + br + "n", slice(None))
        for ci, c in enumerate(("K", "M")):
            m = q <= EDGE[c] + 1e-9
            rf, r1, r2, rn = (ridge(q[m], X[ci][m], f) for X in (full, h1, h2, nul))
            for i, r in enumerate(rf):
                fc = 0.5 * (r["lo"] + r["hi"])
                r.update(ka_h1=r1[i]["ka"], ka_h2=r2[i]["ka"], null_contrast=rn[i]["contrast"])
                r["pass"] = bool(not r["at_edge"] and abs(r["ka_h1"] - r["ka_h2"]) < 0.15
                                 and r["contrast"] > 1.3 * r["null_contrast"] and not (br == "L" and fc < 170))
                if r["pass"]:
                    pts.append((br, c, r["ka"], fc))
            rows[f"{br}_{c}"] = rf
    sub = {"all": pts, "L + T(ka>1)": [p for p in pts if p[0] == "L" or p[2] > 1],
           "T only": [p for p in pts if p[0] == "T"], "T, ka>1": [p for p in pts if p[0] == "T" and p[2] > 1]}
    fits = {nm: (fit(P) if len(P) >= 5 else None) for nm, P in sub.items()}
    rng = np.random.default_rng(0)
    boot = []
    if len(pts) >= 5:
        coarse = np.linspace(-0.333, 0.3, 128)
        boot = [fit([pts[i] for i in rng.integers(0, len(pts), len(pts))], coarse)["k"] for _ in range(nboot)]
    return dict(nseg=len(D), points=pts, rows=rows, fits=fits,
                boot_sd_all=float(np.std(boot)) if boot else None,
                counts={k: sum(1 for p in pts if p[0] + "_" + p[1] == k) for k in rows},
                rotation_per_segment_deg=[float(d["rot_deg"]) for d in D])


def fit_all(refresh=False):
    """Fits of every raft (closest-neighbour grid); cached in OUT/derived/dispersion_fits.json."""
    path = os.path.join(config.DERIVED, "dispersion_fits.json")
    if os.path.exists(path) and not refresh:
        return json.load(open(path))
    sp = rafts.spacings()
    out = {}
    for tag in config.RAFTS:
        s = sp[tag]
        D = load_segments(tag)
        if not D:
            continue
        ratio = 1.0                       # fit on the cached closest-neighbour grid, as published
        r = analyse(tag, ratio)
        if r is None:
            continue
        r.update(ell_um=s["ell_um"], ell_over_closest=ratio)
        r["speeds"] = speeds(r["fits"]["all"]["k"], r["fits"]["all"]["scale_Hz"], s["ell_um"])
        out[tag] = r
        fa = r["fits"]["all"]
        print(f"raft {tag}: {r['nseg']} segments, {fa['n']} ridge points; kappa = {fa['k']:+.3f} "
              f"(bootstrap sd {r['boot_sd_all']:.4f}), s = {fa['scale_Hz']:.0f} Hz, ell = {s['ell_um']:.2f} um, "
              f"c_L = {r['speeds']['cL']:.0f} mm/s, c_T = {r['speeds']['cT']:.0f} mm/s, nu = {r['speeds']['nu']:.3f}")
    config.ensure_dirs(config.DERIVED)
    json.dump(out, open(path, "w"), indent=1)
    return out


# ---- computing new segments (not needed to reproduce the paper) -------------------------------
def _theta6(P, a):
    d = np.linalg.norm(P[:, None] - P[None], axis=-1)
    i, j = np.nonzero(np.triu((d > 0) & (d < 1.3 * a), 1))
    v = P[j] - P[i]
    return np.angle(np.exp(6j * np.arctan2(v[:, 1], v[:, 0])).sum()) / 6


def compute_segment(R, ts):
    """Co-rotating velocity-flux spectra of one Welch segment of raft R starting at frame ts."""
    a = R.a_closest_px
    kv = np.concatenate([np.stack([KA * np.cos(m * np.pi / 6), KA * np.sin(m * np.pi / 6)], 1)
                         for m in range(12)]) / a
    khat = kv / np.linalg.norm(kv, axis=1, keepdims=True)
    kperp = np.stack([-khat[:, 1], khat[:, 0]], 1)
    JL = np.zeros((len(kv), SEG), complex); JT = np.zeros_like(JL)
    NL = np.zeros_like(JL); NT = np.zeros_like(JL)
    rng = np.random.default_rng(int(ts))
    th_prev, th_unw, nmatch = None, 0.0, []
    A = R.frame(ts)
    for j in range(SEG):
        B = R.frame(ts + j + 1)
        dd = np.linalg.norm(A[:, None] - B[None], axis=-1)
        ia, ib = dd.argmin(1), dd.argmin(0)
        ok = (ib[ia] == np.arange(len(A))) & (dd[np.arange(len(A)), ia] < 0.3 * a)
        r = 0.5 * (A[ok] + B[ia[ok]]); v = (B[ia[ok]] - A[ok]) / config.DT
        nmatch.append(ok.mean())
        th = _theta6(r, a)
        th_unw = th if th_prev is None else th_unw + ((th - th_prev + np.pi / 6) % (np.pi / 3) - np.pi / 6)
        th_prev = th
        rc = r - r.mean(0); vc = v - v.mean(0)
        om = np.sum(rc[:, 0] * vc[:, 1] - rc[:, 1] * vc[:, 0]) / np.sum(rc ** 2)
        vrel = vc - om * np.stack([-rc[:, 1], rc[:, 0]], 1)
        c, s = np.cos(-th_unw), np.sin(-th_unw)
        Rm = np.array([[c, -s], [s, c]])
        rp, vp = rc @ Rm.T, vrel @ Rm.T
        ph = np.exp(-1j * (kv @ rp.T))
        Jx, Jy = ph @ vp[:, 0], ph @ vp[:, 1]
        JL[:, j] = khat[:, 0] * Jx + khat[:, 1] * Jy
        JT[:, j] = kperp[:, 0] * Jx + kperp[:, 1] * Jy
        phn = ph[:, rng.permutation(len(rp))]
        Jx, Jy = phn @ vp[:, 0], phn @ vp[:, 1]
        NL[:, j] = khat[:, 0] * Jx + khat[:, 1] * Jy
        NT[:, j] = kperp[:, 0] * Jx + kperp[:, 1] * Jy
        A = B
    win = np.hanning(SEG)
    scale = config.DT / (win ** 2).sum()
    out = {}
    for nm, J in (("L", JL), ("T", JT), ("Ln", NL), ("Tn", NT)):
        F = np.fft.fft((J - J.mean(1, keepdims=True)) * win, axis=1)
        P = (np.abs(F) ** 2 * scale).reshape(12, len(KA), SEG)
        Pf = 0.5 * (P[:, :, 1:SEG // 2] + P[:, :, -1:-SEG // 2:-1])
        out[nm] = np.stack([Pf[0::2].mean(0), Pf[1::2].mean(0)]).astype(np.float32)
    return out, float(np.mean(nmatch)), float(np.degrees(th_unw))


def compute_segments(tag, nseg=6):
    """Add missing segments for raft `tag` to the cache (changes every average that uses it)."""
    R = rafts.Raft(tag)
    d = cache_dir(tag)
    os.makedirs(d, exist_ok=True)
    n = min(nseg, R.T // SEG - 1)
    for ts in np.linspace(0, R.T - SEG - 2, n).astype(int):
        fo = os.path.join(d, f"seg_{int(ts):06d}.npz")
        if os.path.exists(fo):
            continue
        S, m, rot = compute_segment(R, ts)
        np.savez_compressed(fo, SL=S["L"], ST=S["T"], SLn=S["Ln"], STn=S["Tn"], ka=KA,
                            a_px=R.a_closest_px, ts=ts, match=m, rot_deg=rot)
        print(f"  {tag}: segment at frame {ts}: matched {m:.3f}, lattice rotated {rot:+.0f} deg")
