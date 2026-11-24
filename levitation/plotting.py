"""Saving panels and drawing things several figures share."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402

matplotlib.rcParams["pdf.fonttype"] = 42      # editable text in Illustrator


def panel_path(figure, panel, ext="pdf"):
    d = os.path.join(config.FIG, figure)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{panel}.{ext}")


def save_panel(fig, figure, panel, **kw):
    """Write one published panel to OUT/figures/<figure>/<panel>.pdf and close the figure."""
    p = panel_path(figure, panel)
    fig.savefig(p, bbox_inches=kw.pop("bbox_inches", "tight"), **kw)
    plt.close(fig)
    print(f"wrote {p}")
    return p


def derived_path(name):
    os.makedirs(config.DERIVED, exist_ok=True)
    return os.path.join(config.DERIVED, name)


def save_derived(figure, arrays):
    """Merge `arrays` into OUT/derived/<figure>.npz (so running a few panels keeps the others)."""
    import numpy as np
    path = derived_path(f"{figure}.npz")
    old = dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else {}
    old.update(arrays)
    np.savez(path, **old)
    return path


def loess(x, y, xgrid=None, frac=0.4, robust_iters=2, n_grid=200):
    """Robust locally weighted linear regression (degree 1), for smooth trends through clouds.

    At each grid point the fit uses the nearest `frac` of the points with tricube weights, and
    `robust_iters` bisquare passes damp outliers.  Returns (xgrid, trend, local weighted sd of
    the residuals).
    """
    import numpy as np
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    order = np.argsort(x)
    x, y = x[order], y[order]
    n = len(x)
    if xgrid is None:
        xgrid = np.linspace(x.min(), x.max(), n_grid)
    if n < 3:
        yy = np.full_like(xgrid, np.nan)
        return xgrid, yy, yy
    k = int(np.clip(np.ceil(frac * n), 3, n))
    rw = np.ones(n)
    yhat = np.full_like(xgrid, np.nan)
    ystd = np.full_like(xgrid, np.nan)
    for _ in range(robust_iters + 1):
        for i, x0 in enumerate(xgrid):
            d = np.abs(x - x0)
            h = np.partition(d, k - 1)[k - 1]
            if h <= 0:
                pos = d[d > 0]
                h = pos.min() if pos.size else 1.0
            w = (1.0 - np.clip(d / h, 0.0, 1.0) ** 3) ** 3 * rw
            sw = w.sum()
            if sw <= 0:
                continue
            xw = x - x0
            Swx, Swxx = (w * xw).sum(), (w * xw * xw).sum()
            Swy, Swxy = (w * y).sum(), (w * xw * y).sum()
            denom = sw * Swxx - Swx * Swx
            if abs(denom) < 1e-12 * (sw * Swxx + 1e-30):
                b0, b1 = Swy / sw, 0.0
            else:
                b0 = (Swy * Swxx - Swx * Swxy) / denom
                b1 = (sw * Swxy - Swx * Swy) / denom
            yhat[i] = b0
            ystd[i] = np.sqrt(max((w * (y - (b0 + b1 * xw)) ** 2).sum() / sw, 0.0))
        if robust_iters:
            r = y - np.interp(x, xgrid, yhat)
            u = np.clip(r / (6.0 * (np.median(np.abs(r)) + 1e-30)), -1.0, 1.0)
            rw = (1.0 - u ** 2) ** 2
    return xgrid, yhat, ystd
