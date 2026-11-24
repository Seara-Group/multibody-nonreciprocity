"""si_shape_space.py -- SI figure "Net force in shape space" (fig:propulsion).

Net force on an isolated trimer, Eq. (trimer_netforce) of the Supplement, over the
complex shape coordinate Q, for k_a = -alpha_3 = 1 and every other coefficient zero.

Ported unchanged from uchicago/levitation/notebooks/2026-08-17_minimalModel.ipynb
(cells 0, 1, 3).  The notebook's output
figures/2026-08-19_shapeSpaceForce_lambdaA-1.0.pdf is byte-identical to the
paper's suppFigures/shapeSpaceForce_alpha3_-1.0.pdf.  Here lam plays the role of
alpha_3 (lam = -1).

    python si_shape_space.py            # writes OUT/figures/si_shape_space/shape_space.pdf
"""
import os
import sys

import numpy as np
import cmasher as cmr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from levitation.plotting import plt, save_panel  # noqa: E402

W = np.exp(2j * np.pi / 3)
A = 1.0


def tri(s12, s23, s31):
    p1 = np.array([0., 0.])
    p2 = np.array([s12, 0.])
    p3x = (s12**2 + s31**2 - s23**2) / (2 * s12)
    p3y = np.sqrt(max(s31**2 - p3x**2, 1e-18))
    p3 = [p3x, p3y]
    p = np.array([p1, p2, p3])
    return p - p.mean(0)


def deltas(P):
    d = lambda i, j: np.sum((P[i] - P[j])**2) / A**2 - 1
    return np.array([d(1, 2), d(2, 0), d(0, 1)])


cx = lambda v: v[0] + 1j * v[1]


def Ftot(P, lam=1.0):
    D = deltas(P)
    d = lambda i, j: cx(P[j] - P[i])
    return lam * ((D[1] - D[0]) * d(0, 1) + (D[2] - D[1]) * d(1, 2) + (D[0] - D[2]) * d(2, 0))


def shape_from_Q(Q):
    D = np.array([2 / 3 * np.real(np.conj(W)**k * Q) for k in range(3)])
    s = np.sqrt(1 + D)                       # s[a] = side opposite particle a
    return tri(s[2], s[0], s[1]), D          # tri(s12,s23,s31)


def netF(Q, lam=0.0):
    P, D = shape_from_Q(Q)
    return np.nan if P is None else Ftot(P, lam=lam)


def main(lam=-1.0):
    qrr, qii = np.mgrid[-1:1:101j, -1:1:101j]
    QQ = qrr + 1j * qii
    FF = np.vectorize(lambda Q: netF(Q, lam=lam))(QQ)
    FF_unit = FF / (np.abs(FF) + 1e-10)

    fig, ax = plt.subplots(figsize=(3, 3), dpi=144)
    pm = ax.pcolormesh(QQ.real, QQ.imag, np.abs(FF), cmap=cmr.arctic, vmin=0, vmax=3,
                       rasterized=True)
    cax = ax.inset_axes([1.05, 0.0, 0.05, 1])
    cbar = fig.colorbar(pm, cax=cax, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_ylabel(r"$\vert \mathbf{F}_\mathrm{net} \vert$", rotation=-90)

    for q in QQ[::10][:, ::10].ravel():
        P, D = shape_from_Q(q)
        xy = 0.1 * P + np.array([q.real, q.imag])[None, :]
        ax.add_artist(plt.Polygon(xy, edgecolor="k", facecolor="w"))

    ax.quiver(QQ.real[::10][:, ::10], QQ.imag[::10][:, ::10],
              FF_unit.real[::10][:, ::10], FF_unit.imag[::10][:, ::10],
              np.abs(FF[::10][:, ::10]), cmap="Reds", clim=(0, 3))
    ax.set(xlabel=r"$Re(Q)$", ylabel=r"$Im(Q)$", xticks=[-1, 0, 1], yticks=[-1, 0, 1], aspect=1)

    save_panel(fig, "si_shape_space", "shape_space")


if __name__ == "__main__":
    main()
