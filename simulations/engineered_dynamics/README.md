# Engineered dynamics (Fig 6)

Forward simulations of the engineered-activity rafts (dislocation rotor and
swimmer) with the analytic triangle minimal model, to answer three questions:

1. **Real motion, not extrapolation.** Every frame is integrated. The rotor runs
   for a full turn and the swimmer travels 10 lattice spacings. The old
   `steady_movie` rigid extrapolation is not used anywhere.
2. **Overdamped vs underdamped.** Langevin dynamics with mass, scanned over
   damping ratio ζ = 0.3 … 0.003, plus the overdamped reference.
3. **Prestress like the experiment.** Two versions:
   - a **lab-fixed harmonic well**, which grades the compression (highest at
     the centre);
   - a **uniform pressure on the raft's own edge** that moves with it, with the
     same mean compression and no gradient.

Self-contained: the force law is written out in `scripts/eadyn.py`. The package
needs only `jax`, `numpy`, `scipy`, `pyyaml` and `matplotlib`, and imports
nothing from the rest of this repository.

## How to run (cluster)

```bash
cd simulations/engineered_dynamics   # set YOUR_PARTITION / YOUR_ACCOUNT in jobs/*.sh first; mkdir logs
sbatch --array=5 --time=00:20:00 --export=ALL,SMOKE=1 jobs/submit_grid.sh   # smoke test
sbatch --array=0-159%40 jobs/submit_grid.sh                                 # all 160 rows
# resubmit the same line until every row is done (runs checkpoint and resume)
sbatch jobs/submit_post.sh            # stability (if missing) + figures + movies
```

- **Already prepared and shipped, no need to rerun:**
  - `data/init/`: prestressed starting states and `calibration.json` (from `prep.py`);
  - `data/stability.json` (from `stability.py`);
  - `configs/grid.csv` (from `make_grid.py`).
- **One run by hand:** `python scripts/run.py --row N`, or give explicit flags
  (`--help`).
- **Cost:** `configs/grid.csv` has `est_hours` per row, measured at about
  300 µs/step on 2 CPU cores for the 1759-particle rotor. The expensive rows
  are the overdamped rotor full turns, 6–10 h each. Everything else takes
  minutes to about 1 h.
- **GPU:** drop `JAX_PLATFORMS=cpu` in the job script and use a CUDA jax build.
  Expect roughly 10× faster per step (I have not measured this).

## What a run writes

Each run goes to `data/runs/<name>/`:

- `<name>.npz`:
  - `trajectory_R`: about 1500 integrated frames;
  - `times`;
  - `log_*`: observables every chunk of steps — centroid, Kabsch angle θ,
    net force, torque, internal kinetic energy, bond Lindemann ratio, ψ6,
    and `flips` (the number of inverted triangles).
- `<name>_params.yaml`:
  - the coefficients actually used;
  - the static prediction (`Omega_pred`, `v_pred`);
  - the rates measured from the trajectory, fitted over its last 60%;
  - `synthetic_rigid_extrapolation: false`.

Afterwards:

- `scripts/analyze.py` writes `figs/summary.csv` and the Q1–Q3 figures;
- `scripts/render.py --all` makes one mp4 per run, with 5- and 7-fold sites
  recomputed every frame.

## Model and units

- **Units:** lattice units, rest spacing a = 1, bond stiffness k_a = 1.
- **Force law:** every coefficient is zero except k_a and the nonreciprocal
  α1, the same choice as the September runs.
- **Overdamped:** γ = 1, so time is in units of γ/k_a.
- **Underdamped:** m = 1 and γ = 2ζ, where ζ = γ/(2√(m k_a)) is the damping
  ratio of one bond.
- **Comparing damping:** steady rates scale as 1/γ under both dynamics, so
  `analyze.py` compares Ω·γ and v·γ.
- **Starting configurations:** the September α1 = 0 burn-in endpoints (`md_R`),
  copied into `data/init_raw/`. The defect geometry is therefore identical to
  the published-findings runs.
- **Topology** is frozen unless `--retri T` is given. In that case the network
  is retriangulated every T time units, so defects can move.

## Validation done before shipping

- **Step-by-step reproduction of the September numpy run** (rotor, α1/k_a = 0.1):
  max position difference 9.5e-7 after 10,000 steps. That equals the float32
  rounding of the stored frames.
- **Step size:** the overdamped stability limit is dt < 2γ/λ_max = 0.164.
  Runs use 0.08, against the old 0.037. The steady Ω is identical at both
  (−1.1511e−6, equal to the September value). Underdamped runs use
  0.2 × 2/ω_max ≈ 0.115.
- **Nulls:** at α1 = 0, every design × prestress has |net force| and |torque|
  below 1e−14. The edge-pressure force alone is force- and torque-free to 1e−15,
  even on a randomly deformed raft. A defect-free raft in the well has
  τ ≈ 1e−14.

## Prestress calibration — read this before quoting Q3

- **Raft A's measured gradient cannot be reached.** Raft A shows an outer/inner
  local spacing of 57.4/52.1 = **1.102** (`2026-11-05_radial_prestress_test.md`).
  The linear minimal model cannot hold that.
- **What fails:** a slowly ramped harmonic well folds the bond network once the
  ratio passes about **1.06–1.07**, at a centre strain of about −13%. Triangles
  invert and particles pass through each other, which hard-core particles cannot
  do. Applied suddenly instead of ramped, it folds even earlier.
- **What is used instead:** `prep.py` therefore targets a ratio of **1.05**,
  about half of raft A's gradient. That gives a centre strain of −9.6%, an edge
  strain of −5%, and a mean compression of 7.5%.
  - The edge-pressure version matches the mean compression with a flat profile:
    P = 0.27 k_a, ratio 1.000.
  - `--target` changes the target ratio.
  - Every run aborts with `status: folded` if any triangle inverts.
- **Likely missing physics:** in the real raft the bond stiffness itself rises
  about 2.4× toward the centre (same note). The linear model keeps k_a uniform.
  So raft A's gradient probably needs bonds that stiffen under compression, or a
  hard core, and that is a change to the force law.

## Results already in hand (static or linear, not yet the dynamics)

**Linear stability** (`stability.py`, `data/stability.json`):

- The critical damping ratio below which some mode grows is
  ζ_c ≈ |Im λ| / (2√(Re λ)).
- It scales linearly with α1. The values below are rotor / swimmer.

  | α1 / k_a | 0.1 | 0.033 | 0.003 (physical) |
  |---|---|---|---|
  | ζ_c | 0.004 / 0.0025 | 0.0009 / 0.0004 | 0.0001 / 0.0000 |

- At the physical α1, ζ_c sits well below every damping estimate for raft A,
  which spans about 0.001–0.1. So inertia alone should not destabilize these
  rafts. Only the exaggerated α1 = 0.1 at ζ = 0.003 should heat up, and the grid
  includes exactly those rows.

**Static torque and force under prestress** (per unit α1, at the relaxed
prestressed state):

| design | none | trap (graded) | pressure (uniform) |
|---|---|---|---|
| rotor, τ | −4.38 | **+13.7** | −4.40 |
| swimmer, \|F\| | 4.98 | 4.25 | 4.55 |
| swimmer, τ | −0.89 | **−7.38** | −0.76 |

- **Rotor:** the graded compression reverses its static torque and makes it
  3× bigger. Uniform compression leaves it unchanged.
- **Swimmer:** in the well it also picks up 8× more torque.
- **Not a background source:** a defect-free raft in the same well has zero
  torque. So the effect comes from how the gradient reshapes the defect cores.
- **Still to confirm:** whether the reversal survives the dynamics is
  what rows 24–47 test.

## Preview: 28 rows already run in the sandbox (α1/k_a = 0.1)

These rows are in `figs/preview/`: summary CSV, figure and movies. They were
not shipped as data, so the cluster reruns them, cheaply.

| run | Ω·γ | \|v\|·γ | notes |
|---|---|---|---|
| rotor, none, ζ = 0.01 | −1.151e−6 | 2e−6 | full turn, integrated; equals the overdamped September value |
| rotor, none, ζ = 0.003 (< ζ_c) | −1.533e−6 | 2e−6 | heats up (Lindemann ratio 7× higher), turns 33% faster, drifts about 1 a |
| rotor, trap, ζ = 0.01 and 0.003 | **+4.10e−6** | 0 | **turns the opposite way**, 3.6× faster; the trap holds it in place |
| rotor, pressure, ζ = 0.003 (< ζ_c) | −1.206e−6 | 8e−6 | heats (Lindemann ratio 0.046, ψ6 drops from 0.990 to 0.973) and **drifts 12.6 a** |
| swimmer, none, overdamped → ζ = 0.003 | −4.89e−7 | 3.97e−4 | identical to four digits at every ζ |
| swimmer, pressure, all ζ | −5.0e−7 | 3.66e−4 | 8% slower than without prestress |
| swimmer, trap, all ζ | **−5.50e−6** | ≈0 | **held in place** (offset about 0.1 a) **and spins 11× faster** |

- **Above ζ_c, inertia changes nothing measurable.** Only the start-up time
  (m/γ) is new.
- **Below ζ_c, the linear prediction holds.** Unstable modes heat the raft, and
  the heated raft also translates. Even so, ψ6 stays above 0.97 within one turn,
  so it has not melted.
- **All α1 = 0 controls stay at zero:** |Ω|, |v| ≲ 1e−20.

## Damping-ratio range: why ζ = 0.003–0.3

The raft A spectra do not give one number:

- The broad phonon ridge gives Q ≈ 5–10. That is a lower bound on Q, because
  disorder broadening adds to the width.
- The narrow "stripe" modes give Q ≈ 20–90 in 1.4 s windows, and 38–550 in the
  September full-record analysis. Those modes may be driven rather than freely
  ringing.
- Stokes drag in air gives Q ≈ 5.

All of these mean underdamped. A single-mode ring-down on the raw trajectory is
the open item that would pin ζ down.

## Not done / open

- **Physical units.** Converting to seconds and µm/s needs k_a in physical
  units. Not done.
- **Size-dependent α1(s)** (G6 in `engineered_activity.py`) is not included:
  `alpha1_profile.npz` was never written. Constant α1 only.

## Add-on (2026-11-12): all-underdamped engineered-activity figure

Fifteen extra runs so every panel of the paper's engineered-activity figure
can come from underdamped dynamics (ζ = 0.01, no prestress):

- `configs/grid_fig.csv` (15 rows):
  - rows 0–10: tilt walls n = 1..4 at α1/k_a = −0.05, 0, +0.05;
  - rows 11–14: rotor at α1/k_a = −0.1, −0.066, −0.033, +0.066.
  - Reused from the main grid: rotor at 0.1, 0.033, 0 (grid.csv rows 16, 17, 19)
    and wall4 at 0 (row 91).
- `jobs/submit_fig.sh`: `sbatch --array=0-14 jobs/submit_fig.sh`. Each row takes minutes.
- `scripts/plot_fig_underdamped.py`: writes `figs/fig_engineered_activity_ud.{pdf,png}`
  (panels a–f) and `figs/panel_{a,d}_trails_ud.pdf` (the trail maps alone).
  Missing runs are skipped and listed.
- Time is in √(m/k_a); one bond oscillation is 2π of these.

### Supplementary: the same figure with uniform edge pressure

- `configs/grid_fig_pressure.csv`: the same 15 rows as `grid_fig.csv`, but with
  `prestress = pressure` (P ≈ 0.28 k_a on the raft's edge, 7.5% uniform compression).
- Reused from the main grid: rotor at 0, 0.033, 0.1 and wall4 at 0, all with pressure, ζ = 0.01.
- Submit: `sbatch --array=0-14 --export=ALL,GRID=grid_fig_pressure.csv jobs/submit_fig.sh`
- Plot: `scripts/plot_fig_prestress.py` writes `figs/fig_engineered_activity_prestress.{pdf,png}`:
  - a, b, d, e: the compressed rafts;
  - c, f: no prestress (filled) against pressure (open) on the same axes;
  - it also prints the pressure / no-prestress ratio of every rate, for the caption.
