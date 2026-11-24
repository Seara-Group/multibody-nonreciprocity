#!/bin/bash
#SBATCH --job-name=eadyn_post
#SBATCH --output=logs/eadyn_post_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
#SBATCH --partition=YOUR_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#
# Linear stability (zeta_c), and the summary figures.
#   sbatch jobs/submit_post.sh
# stability.py needs no runs and can go first; analyze and the plotting scripts pick up whatever
# has finished so far, so this can be rerun at any time.
# Submit from the package folder (paper/simulations/engineered_dynamics), or set ROOT.
ROOT=${ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
export JAX_PLATFORMS=cpu
eval "$(conda shell.bash hook)"
conda activate jax   # any environment with jax, numpy, scipy, pyyaml, matplotlib
cd $ROOT/scripts
[[ -f ../data/stability.json ]] || python stability.py
python analyze.py
python plot_fig_underdamped.py
python plot_fig_prestress_existing.py
