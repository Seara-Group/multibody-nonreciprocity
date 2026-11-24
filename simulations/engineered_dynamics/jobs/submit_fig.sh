#!/bin/bash
#SBATCH --job-name=eadyn_fig
#SBATCH --output=logs/eadyn_fig_%a_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#
# Add-on for the all-underdamped engineered-activity figure (zeta = 0.01, no
# prestress).  One row of configs/grid_fig.csv per array task: 15 rows.
#
#   sbatch --array=0-14 jobs/submit_fig.sh
#
# Supplementary version with uniform edge pressure (configs/grid_fig_pressure.csv):
#
#   sbatch --array=0-14 --export=ALL,GRID=grid_fig_pressure.csv jobs/submit_fig.sh
#
# then  python plot_fig_prestress.py
#
# Rows 0-10: tilt walls n = 1..4 at alpha1/k_a = -0.05, 0, +0.05 (panels a-c).
# Rows 11-14: rotor at alpha1/k_a = -0.1, -0.066, -0.033, +0.066 (panels d-f).
# The rest of the figure reuses runs already in data/runs (grid.csv rows 16,
# 17, 19 for the rotor at 0.1, 0.033, 0; row 91 for wall4 at 0).
# Every row should take minutes (longest: rotor -0.033, ~0.3 h on 2 cores).
# Runs checkpoint and resume exactly like submit_grid.sh.
# Then:  cd scripts && python plot_fig_underdamped.py

# Submit from the package folder (paper/simulations/engineered_dynamics), or set ROOT.
ROOT=${ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
ROW=${SLURM_ARRAY_TASK_ID}
GRID=${GRID:-grid_fig.csv}
WALL=$(( 2*3600 - 600 ))

export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

eval "$(conda shell.bash hook)"
conda activate jax   # any environment with jax, numpy, scipy, pyyaml, matplotlib
cd $ROOT/scripts

echo "${GRID} row ${ROW} on $(hostname)  $(date)"
python run.py --grid $ROOT/configs/${GRID} --row ${ROW} --walltime ${WALL}
echo "exit $?  $(date)"
