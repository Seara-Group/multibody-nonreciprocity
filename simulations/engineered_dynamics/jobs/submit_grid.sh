#!/bin/bash
#SBATCH --job-name=eadyn
#SBATCH --output=logs/eadyn_%a_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#
# One row of configs/grid.csv per array task.  Runs CHECKPOINT and exit cleanly
# 20 min before the time limit; resubmitting the same array resumes them, and
# finished rows exit immediately.  So just resubmit until analyze.py shows every
# row done:
#
#   sbatch --array=0-159%40 jobs/submit_grid.sh          # everything
#   sbatch --array=0-3 jobs/submit_grid.sh               # rotor/none/overdamped (Q1 headline)
#   sbatch --array=5 --time=00:20:00 --export=ALL,SMOKE=1 jobs/submit_grid.sh
#
# Row numbers and per-row cost estimates are in configs/grid.csv (est_hours is
# for 2 CPU cores; check it against the first finished row's wall_seconds).
# GPU: drop JAX_PLATFORMS below, request a GPU, and use a CUDA jax build --
# expect roughly 10x per step.

# Submit from the package folder (paper/simulations/engineered_dynamics), or set ROOT.
ROOT=${ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
ROW=${SLURM_ARRAY_TASK_ID}
WALL=$(( 12*3600 - 1200 ))

export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

eval "$(conda shell.bash hook)"
conda activate jax   # any environment with jax, numpy, scipy, pyyaml, matplotlib
cd $ROOT/scripts

EXTRA=()
if [[ -n "${SMOKE:-}" ]]; then
    EXTRA=(--smoke)
fi
echo "row ${ROW} on $(hostname)  $(date)"
python run.py --row ${ROW} --walltime ${WALL} "${EXTRA[@]}"
echo "exit $?  $(date)"
