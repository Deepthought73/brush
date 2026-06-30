#!/usr/bin/env python3
"""Grid-search sweep over single-view-experiment parameters.

Edit the CONFIG section below: PARAM_GRID is the cartesian grid to sweep, and
every combination is run once. Each run's training CSV is saved to OUT_DIR (so
you can plot the interesting ones afterwards for comparison) together with a
manifest.csv mapping every CSV to the params that produced it.

PLY splats are always discarded. SSIM maps are not stored (the binary only
writes them when passed --save-ssim, which this sweep does not).

Usage:
    python grid_sweep.py
"""
import csv
import itertools
import os
import shutil
import subprocess
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BINARY = os.path.join(REPO_ROOT, "target", "release", "single-view-experiment")

# ============================== CONFIG =======================================
OUT_DIR = "/tmp/grid_sweep"

# Flags passed to every run. Use the binary's long flags (see --help).
BASE_ARGS = [
    "--total-train-iters", "20",
    "--depth-loss-weight", "0.1",
    "--scale-ratio-penalty", "0.1",
    "--lr-mean-end", "2e-5",
    "--refine-knn-scales"
]

# Parameters to grid-search. Keys are CLI flags, values are the lists to sweep.
# The full grid is the cartesian product of all value lists. Presence flags
# (e.g. --refine-knn-scales, which take no value) are swept with bool values:
# True passes the flag, False omits it.
PARAM_GRID = {
    "--init-mode": ["depth"],
    "--stride": [5, 10],
    "--initial-cov-scale": [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7],
    "--initial-opacity": [0.9],
    "--refine-samples": [1, 2, 3, 4, 5],
    "--refine-every": [50, 100, 200, 500, 1000],
}
# =============================================================================


def build_binary():
    subprocess.run(
        ["cargo", "build", "--release", "-p", "single-view-experiment"],
        cwd=REPO_ROOT,
        check=True,
    )


def sanitize(value):
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value).replace("/", "_").replace(".", "p").replace("-", "m")


def label_for(combo):
    return "_".join(f"{flag.lstrip('-')}-{sanitize(val)}" for flag, val in combo.items())


def run_combo(run_id, combo):
    label = label_for(combo)
    csv_path = os.path.join(OUT_DIR, f"run_{run_id:03d}_{label}.csv")

    cmd = [
        BINARY,
        *BASE_ARGS,
        "--out-csv-path", csv_path,
    ]
    for flag, val in combo.items():
        if isinstance(val, bool):
            if val:
                cmd.append(flag)
        else:
            cmd += [flag, str(val)]

    print(f"[{run_id:03d}] {label}", flush=True)
    returncode = subprocess.run(cmd, cwd=REPO_ROOT).returncode

    return csv_path, returncode


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    build_binary()

    flags = list(PARAM_GRID)
    combos = [dict(zip(flags, values)) for values in itertools.product(*PARAM_GRID.values())]
    print(f"Running {len(combos)} parameter combinations.")

    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "csv_file", "returncode", *flags])
        for run_id, combo in enumerate(combos):
            csv_path, returncode = run_combo(run_id, combo)
            if returncode != 0:
                print(f"  run {run_id:03d} FAILED (returncode={returncode})")
            writer.writerow(
                [run_id, os.path.basename(csv_path), returncode, *[combo[k] for k in flags]]
            )
            f.flush()

    print(f"Wrote {len(combos)} CSVs and {manifest_path}")


if __name__ == "__main__":
    main()
