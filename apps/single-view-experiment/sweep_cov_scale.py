#!/usr/bin/env python3
"""Sweep --initial-cov-scale for the single-view-experiment binary and plot the
first-step PSNR/SSIM against it.

Runs the executable n times with --initial-cov-scale ranging from 1 to 5 in
equal steps (the other args are fixed), reads the first data row of each run's
CSV, and plots PSNR and SSIM vs initial_cov_scale.

Usage:
    python sweep_cov_scale.py [--runs N] [--min 1.0] [--max 5.0]
"""
import argparse
import csv
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np

# Repo root is two levels up from this script (apps/single-view-experiment/).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Fixed args for every run (initial-cov-scale is swept, so it's added per run).
BASE_ARGS = [
    "--total-train-iters", "2",
    "--initial-opacity", "0.5",
    "--stride", "5",
    "--lr-mean-end", "2e-5"
]


def first_row(csv_path):
    """Return the first data row of the CSV as a dict."""
    with open(csv_path, newline="") as f:
        return next(csv.DictReader(f))


def run_once(cov_scale, out_csv):
    """Run the binary for one cov_scale and return (psnr, ssim) at the first step."""
    cmd = [
        "cargo", "run", "--quiet", "-p", "single-view-experiment", "--",
        *BASE_ARGS,
        "--initial-cov-scale", repr(float(cov_scale)),
        "--out-csv-path", out_csv,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    row = first_row(out_csv)
    return float(row["psnr"]), float(row["ssim"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5, help="number of cov-scale samples")
    ap.add_argument("--min", type=float, default=1.0, help="min initial-cov-scale")
    ap.add_argument("--max", type=float, default=5.0, help="max initial-cov-scale")
    ap.add_argument("--out-dir", default="/tmp/cov_sweep", help="where to write per-run CSVs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cov_scales = np.linspace(args.min, args.max, args.runs)

    psnrs, ssims = [], []
    for i, cov in enumerate(cov_scales):
        out_csv = os.path.join(args.out_dir, f"run_{i}_cov_{cov:.3f}.csv")
        print(f"[{i + 1}/{args.runs}] initial-cov-scale={cov:.3f} ...", flush=True)
        psnr, ssim = run_once(cov, out_csv)
        print(f"    PSNR={psnr:.4f} SSIM={ssim:.4f}")
        psnrs.append(psnr)
        ssims.append(ssim)

    fig, (ax_psnr, ax_ssim) = plt.subplots(1, 2, figsize=(12, 5))
    ax_psnr.plot(cov_scales, psnrs, marker="o")
    ax_psnr.set(title="First-step PSNR", xlabel="initial_cov_scale", ylabel="PSNR")
    ax_ssim.plot(cov_scales, ssims, marker="o", color="tab:orange")
    ax_ssim.set(title="First-step SSIM", xlabel="initial_cov_scale", ylabel="SSIM")
    for ax in (ax_psnr, ax_ssim):
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_png = os.path.join(args.out_dir, "cov_sweep.png")
    fig.savefig(out_png, dpi=120)
    print(f"Wrote {out_png}")
    plt.show()


if __name__ == "__main__":
    main()
