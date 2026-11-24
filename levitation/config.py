"""Paths, units and constants shared by every figure.

Folders (environment variables):

    LEVITATION_DATA  experiment and COMSOL data (read-only inputs, plus the slow caches).
    LEVITATION_OUT   results.  Panels go to OUT/figures/<figure>/<panel>.pdf; arrays that one
                     step hands to another go to OUT/derived/.  Default: output/.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(_HERE, ".."))           # .../multibody_interactions/paper
REPO = os.path.abspath(os.path.join(PAPER, ".."))

DATA = os.path.expanduser(os.environ.get("LEVITATION_DATA", os.path.join(PAPER, "data")))
OUT = os.path.expanduser(os.environ.get("LEVITATION_OUT", os.path.join(PAPER, "output")))
FIG = os.path.join(OUT, "figures")
DERIVED = os.path.join(OUT, "derived")

# ---- units -----------------------------------------------------------------
# Camera frame rate used for every experimental time and frequency axis.  All published
# figures use 3000 fps (the trajectory cache also stores dt = 1/3000 s).
FRAME_RATE = 3000.0          # frames per second
DT = 1.0 / FRAME_RATE        # s

# Raft A: particle diameter, used to convert the tracker's pixels to microns through the
# tracked radius (um per px = DIAMETER_UM / (2 <radius_px>)).
DIAMETER_UM = 41.1

# The experiments: tag -> (trajectory file in DATA/expt_trajs, particle diameter um, label).
# Each file is the particle tracker's output for one experiment: a pickled pandas DataFrame with
# columns x, y (px), radius (px), frame, label.
RAFTS = {
    "A": ("raft_A_41.1um_34.7kHz.pkl", 41.1, "41.1 µm, 34.7 kHz"),
    "B": ("raft_B_41.1um_64.5kHz.pkl", 41.11, "41.1 µm, 64.5 kHz"),
    "C": ("raft_C_59.7um_34.7kHz.pkl", 59.7, "59.7 µm, 34.7 kHz"),
    "D": ("raft_D_59.7um_64.4kHz.pkl", 59.7, "59.7 µm, 64.4 kHz"),
    "E": ("raft_E_32um_34.8kHz.pkl", 32.0, "32 µm, 34.8 kHz"),
}
# The same files under the names the tracker gave them (still accepted).
TRACKER_NAMES = {
    "A": "27060_crystal_1_phonon.txt",
    "B": "2025123001 64.54kHz 9Vpp raft 41.11um.txt",
    "C": "27060_60um_348khz.txt",
    "D": "2025123002 64.41kHz 10Vpp raft 59.70um.txt",
    "E": "27060_32um_348khz.txt",
}


def raw_trajectory(tag):
    """Path of a raft's trajectory file: DATA/expt_trajs (or DATA), deposited or tracker name."""
    for name in (RAFTS[tag][0], TRACKER_NAMES[tag]):
        for d in (os.path.join(DATA, "expt_trajs"), DATA):
            if os.path.exists(os.path.join(d, name)):
                return os.path.join(d, name)
    raise FileNotFoundError(f"raft {tag}: neither {RAFTS[tag][0]} nor {TRACKER_NAMES[tag]} in {DATA}/expt_trajs")


# Fig 6: the engineered-dynamics runs (simulations/engineered_dynamics writes them to
# <its folder>/data/runs on the cluster).  Folder of <name>/<name>.npz + <name>_params.yaml.
ENGINEERED_RUNS = os.path.expanduser(os.environ.get("LEVITATION_ENGINEERED_RUNS",
                                                    os.path.join(DATA, "engineered_dynamics", "runs")))


def ensure_dirs(*dirs):
    for d in dirs or (FIG, DERIVED):
        os.makedirs(d, exist_ok=True)
    return dirs[0] if len(dirs) == 1 else None
