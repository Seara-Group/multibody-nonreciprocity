"""Collect exactly the files the code reads into two folders for the data deposit.

    python tools/prepare_data_deposit.py --data <data folder> --runs <engineered-dynamics runs folder> \
        --summary <engineered-dynamics figs/summary.csv> --out <empty folder> [--zip]

Writes
    <out>/data/                  the folder LEVITATION_DATA should point to
    <out>/engineered_dynamics/   runs/ (LEVITATION_ENGINEERED_RUNS) and summary.csv
each with a MANIFEST.txt (path, size in bytes, SHA-256).  The five raft trajectory files are
renamed from the tracker's names to config.RAFTS (raft_<X>_<diameter>_<drive>.pkl); the code
accepts either name.  Files are copied, never moved; the sources are only read.  With --zip,
also writes <out>/data.zip and <out>/engineered_dynamics.zip.
"""
import argparse
import hashlib
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from levitation import config  # noqa: E402

# ---- what the code reads ---------------------------------------------------------------------
SMALL = ["_traj_cache_all.npz",                                   # raft A, fast-loading form
         "phase_scan_result2.mat",                                # Fig 3c, coefficients
         os.path.join("41um_particle", "three_body_data_41um.pkl"),   # Fig 3b
         "comsol_dynamical_matrix.mat", "comsol_exp_permutation.npz",  # SI stripes d-f
         "phonon_data.csv"]                                       # Fig 4c
JACOBIAN_RUNS = ["41.1um_34700hz", "41.1um_64500hz", "59.7um_64500hz"]   # Fig 4c
SPECTRA_RAFTS = "ABCDE"                                           # _raft_spectra_cache/<X>/seg_*.npz

FIG6 = ([f"wall{n}_none_z0.01_a{a:g}" for n in (1, 2, 3, 4) for a in (-0.05, 0, 0.05)]
        + [f"rotor_none_z0.01_a{a:g}" for a in (-0.1, -0.066, -0.033, 0, 0.033, 0.066, 0.1)])
PRESTRESS = [f"{d}_{p}_z0.01_a{a:g}" for d in ("wall4", "rotor") for p in ("none", "pressure")
             for a in (0, 0.003, 0.033, 0.1)]


def data_files(src):
    """[(source path, path inside data/)]."""
    out = []
    for tag in config.RAFTS:
        out.append((config.raw_trajectory(tag), os.path.join("expt_trajs", config.RAFTS[tag][0])))
    for f in SMALL:
        out.append((os.path.join(src, f), f))
    for run in JACOBIAN_RUNS:
        for pre in ("force_jacobian", "force_jacobian_095_", "force_jacobian_105_"):
            f = os.path.join("force_jacobian", f"{pre}{run}_.mat")
            out.append((os.path.join(src, f), f))
    for tag in SPECTRA_RAFTS:
        d = os.path.join("_raft_spectra_cache", tag)
        for f in sorted(os.listdir(os.path.join(src, d))):
            if f.startswith("seg_") and f.endswith(".npz"):
                out.append((os.path.join(src, d, f), os.path.join(d, f)))
    return out


def run_files(runs, summary):
    out = []
    for nm in sorted(set(FIG6) | set(PRESTRESS)):
        for f in (f"{nm}.npz", f"{nm}_params.yaml"):
            out.append((os.path.join(runs, nm, f), os.path.join("runs", nm, f)))
    if summary:
        out.append((summary, "summary.csv"))
    return out


def sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(block)
            if not b:
                return h.hexdigest()
            h.update(b)


def collect(pairs, dest):
    missing = [s for s, _ in pairs if not os.path.exists(s)]
    if missing:
        sys.exit("missing source files:\n  " + "\n  ".join(missing))
    lines = []
    for i, (s, rel) in enumerate(pairs, 1):
        t = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(t), exist_ok=True)
        if not (os.path.exists(t) and os.path.getsize(t) == os.path.getsize(s)):
            shutil.copy2(s, t)
        lines.append(f"{rel}\t{os.path.getsize(t)}\t{sha256(t)}")
        print(f"  [{i}/{len(pairs)}] {rel}", flush=True)
    with open(os.path.join(dest, "MANIFEST.txt"), "w") as fh:
        fh.write("# path\tbytes\tsha256\n" + "\n".join(lines) + "\n")
    total = sum(os.path.getsize(os.path.join(dest, r)) for _, r in pairs)
    print(f"{dest}: {len(pairs)} files, {total / 1e9:.2f} GB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=config.DATA, help="the folder LEVITATION_DATA points to")
    ap.add_argument("--runs", default=config.ENGINEERED_RUNS, help="the engineered-dynamics runs folder")
    ap.add_argument("--summary", default=None, help="engineered-dynamics figs/summary.csv (SI pre-stress d, h)")
    ap.add_argument("--out", required=True, help="an empty folder with ~10 GB free")
    ap.add_argument("--zip", action="store_true", help="also write data.zip and engineered_dynamics.zip")
    a = ap.parse_args()
    config.DATA = os.path.expanduser(a.data)          # raw_trajectory looks here
    out = os.path.expanduser(a.out)
    print("== data/")
    collect(data_files(config.DATA), os.path.join(out, "data"))
    print("== engineered_dynamics/")
    collect(run_files(os.path.expanduser(a.runs), a.summary and os.path.expanduser(a.summary)),
            os.path.join(out, "engineered_dynamics"))
    if a.zip:
        for name in ("data", "engineered_dynamics"):
            print(f"zipping {name} ...", flush=True)
            shutil.make_archive(os.path.join(out, name), "zip", out, name)
    print("done")


if __name__ == "__main__":
    main()
