# Code for "Solids activated by multi-body nonreciprocity"

Wu, Seara, Mao, Vitelli, Jaeger. [arXiv:2611.30058](https://arxiv.org/abs/2611.30058)

This repository holds the code behind every data figure and table in the paper: one script per
figure in `figures/`, built on a small package of shared routines in `levitation/`.

## Setup

```bash
conda env create -f environment.yml && conda activate levitation-paper   # or any env with these packages
export LEVITATION_DATA=/path/to/data                   # input data (see "Data")
export LEVITATION_OUT=/path/to/output                  # optional; default output/ in this folder
export LEVITATION_ENGINEERED_RUNS=/path/to/runs        # Fig 6 only: the runs of simulations/engineered_dynamics
```

Inputs are only read. Every panel goes to `$LEVITATION_OUT/figures/<figure>/<panel>.pdf`, one PDF
per published panel; the arrays a panel was drawn from, and the slow intermediate results, go to
`$LEVITATION_OUT/derived/`. The published figures were assembled from these panels in Illustrator,
which also added the panel letters; panels marked *drawn* have no code.

## Figures and tables

| Paper | Panels | Command |
|---|---|---|
| Fig 1 | a drawn, b camera frame; c–e | `python figures/fig1.py` |
| Fig 2 | a–g | `python figures/fig2.py` |
| Fig 3 | a drawn; b, c | `python figures/fig3.py` |
| Fig 4 | a drawn; b, c | `python figures/fig4.py` |
| Fig 5 | a–f | `python figures/fig5.py` |
| Fig 6 | a–f | `python figures/fig6.py` |
| SI, net force in shape space | | `python figures/si_shape_space.py` |
| SI, raft spectra (A, B, D) | | `python figures/si_raft_spectra.py` |
| SI, spacing grows to the edge | a–c | `python figures/si_rdf.py` |
| SI, non-dispersive lines | a–f | `python figures/si_stripes.py` |
| SI table, dispersion fits | | `python figures/tables.py dispersion` |
| SI table, $\Vert D^- \Vert / \Vert D^+ \Vert$ vs number of modes | | `python figures/tables.py inference` |
| SI, engineered activity under pre-stress | | `simulations/engineered_dynamics/scripts/plot_fig_prestress_existing.py` (not yet refactored) |
| SI, force summation rule | | drawn |

Most figure scripts take `--panel` to draw a subset (e.g. `python figures/fig2.py --panel e`).
Each prints the numbers the text quotes from that figure. `figures/<name>.py` says in its
docstring what every panel shows and how it is computed.

Fig 6's raft simulations themselves run on a cluster: see `simulations/engineered_dynamics/README.md`.

## Shared code (`levitation/`)

| Module | What it does |
|---|---|
| `config` | data and output folders, frame rate (3000 fps), the five rafts |
| `io`, `rafts` | trajectories of raft A (and its cache); rafts B–E from the tracker files |
| `lattice` | Delaunay neighbours, 5/7 defects, closed nearest-neighbour shells, $\psi_6$, lattice spacing $\ell$ |
| `modes` | rigid-motion removal, PCA, winding numbers, cycling matrix, phase-randomised surrogates |
| `spectra`, `dispersion` | velocity-flux spectra, ridge extraction, spring-lattice dispersion fits |
| `minimal_model` | the three-body force law: finite rafts, dynamical matrices, lattice dispersion |
| `trimers` | COMSOL three-particle data and the minimal-model coefficients measured from them |
| `inference` | dynamical matrix inferred from trajectories, $\Vert D^- \Vert / \Vert D^+ \Vert$, surrogate noise floor |
| `comsol` | COMSOL lattice Jacobians and the COMSOL dynamical matrix at the experimental configuration |
| `plotting` | panel saving, LOESS trends |

## Conventions

- **Frame rate** 3000 fps for every experimental time and frequency axis.
- **Lattice spacing**, as in the published figures. The raft-average spacing $\ell$ is the median
  Delaunay bond length ($54.01~\mu\mathrm{m}$ for raft A, from its mean positions); it is the length unit of
  Fig 5, the SI stripes and the dispersion table's $\ell$ column and sound speeds. Two places use other
  spacings: Fig 4b's wavevector axis uses the first peak of $g(r)$ ($52.48~\mu\mathrm{m}$), and the dispersion
  fits and SI raft spectra use each raft's median closest-neighbour distance $a$.
- **Minimal-model coefficients** are a first-order expansion about $54.01~\mu\mathrm{m}$, the median
  nearest-neighbour bond of raft A, measured from the COMSOL 950-configuration phase scan
  (`levitation/trimers.py`; `figures/fig3.py` prints them with uncertainties).
- **Defects** are Delaunay coordination numbers, counted only where a particle's nearest-neighbour
  shell closes (Delaunay triangles with sides $< 1.45\,\ell$), so the free edge is not marked.

## Data

The data are on the Materials Data Facility: [doi:10.18126/z5pn-n620](https://doi.org/10.18126/z5pn-n620).
Unzip `data.zip` and `engineered_dynamics.zip`, then set

```bash
export LEVITATION_DATA=/path/to/data
export LEVITATION_ENGINEERED_RUNS=/path/to/engineered_dynamics/runs
```

Each folder has a `MANIFEST.txt` with every file's size and SHA-256 checksum.

Read from `$LEVITATION_DATA`:

- `expt_trajs/raft_<X>_<diameter>_<drive>.pkl`: the particle tracker's output for rafts A–E, one pickled
  pandas DataFrame per experiment (columns `x`, `y`, `radius` in pixels, `frame`, `label`)
- `_traj_cache_all.npz`: raft A's trajectories in a fast-loading form (rebuilt from the raw file if absent)
- `_raft_spectra_cache/`: the Welch segments the dispersion fits use (fixed on 2026-10-05; the fits depend on exactly these)
- `phonon_data.csv`, `force_jacobian/`: experimental sound speeds and COMSOL lattice Jacobians (Fig 4c)
- `comsol_dynamical_matrix.mat`, `comsol_exp_permutation.npz`: COMSOL matrix at the experimental configuration (SI stripes)
- `phase_scan_result2.mat`, `41um_particle/three_body_data_41um.pkl`: COMSOL three-particle simulations (Fig 3, coefficients)

Fig 6 reads the simulation runs in `$LEVITATION_ENGINEERED_RUNS` (`<run>/<run>.npz` and `<run>_params.yaml`).

The SI pre-stress figure also reads `engineered_dynamics/summary.csv`
(`plot_fig_prestress_existing.py --summary`). `tools/prepare_data_deposit.py` is the script that
assembled the deposit from these files.

## Citation

If you use this code, please cite

> Wu, Seara, Mao, Vitelli, Jaeger, "Solids activated by multi-body nonreciprocity",
> [arXiv:2611.30058](https://arxiv.org/abs/2611.30058) (2026).

## License

MIT; see `LICENSE`.
