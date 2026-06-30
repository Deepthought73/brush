#!/usr/bin/env python3
"""Plot the per-step SSIM maps (.npy) written by the single-view-experiment
binary into its --ssim-folder (default /tmp/ssim).

With no extra args it opens an interactive viewer with a slider to scrub
through iterations. Pass a specific .npy file to show just that one, or
--all to tile every map in the folder.

Usage:
    python plot_ssim.py [/tmp/ssim | path/to/ssim_42.npy] [--all]
"""
import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


def step_of(path):
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def show_one(path):
    data = np.load(path)
    plt.figure()
    im = plt.imshow(data, cmap="viridis")
    plt.colorbar(im, label="SSIM")
    plt.title(f"{os.path.basename(path)}  (mean={data.mean():.4f})")
    plt.axis("off")


def show_slider(files):
    """Interactive viewer: a slider scrubs through iterations, redrawing the map."""
    files = sorted(files, key=step_of)
    steps = [step_of(f) for f in files]
    maps = [np.load(f) for f in files]
    # Shared color scale so brightness is comparable while scrubbing.
    vmin = min(m.min() for m in maps)
    vmax = max(m.max() for m in maps)

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    im = ax.imshow(maps[-1], cmap="viridis", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label="SSIM")
    ax.axis("off")

    def render(i):
        ax.set_title(f"step {steps[i]}  (mean={maps[i].mean():.4f})")
        im.set_data(maps[i])
        fig.canvas.draw_idle()

    slider_ax = fig.add_axes([0.15, 0.07, 0.7, 0.04])
    slider = Slider(
        slider_ax, "iter", 0, len(files) - 1,
        valinit=len(files) - 1, valstep=1,
    )
    # Label the slider by the actual step number rather than the list index.
    slider.valtext.set_text(str(steps[-1]))
    slider.on_changed(lambda v: (render(int(v)), slider.valtext.set_text(str(steps[int(v)]))))
    render(len(files) - 1)
    # Keep a reference so the slider isn't garbage-collected.
    fig._ssim_slider = slider


def show_all(files):
    files = sorted(files, key=step_of)
    n = len(files)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    # Shared color scale across all maps for fair comparison.
    vmin = min(np.load(f).min() for f in files)
    vmax = max(np.load(f).max() for f in files)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()
    im = None
    for ax, f in zip(axes, files):
        data = np.load(f)
        im = ax.imshow(data, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"step {step_of(f)}  (mean={data.mean():.4f})")
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes.tolist(), label="SSIM", shrink=0.8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="/tmp/ssim",
                    help="SSIM folder or a single .npy file")
    ap.add_argument("--all", action="store_true",
                    help="tile every map in the folder instead of just the latest")
    args = ap.parse_args()

    if os.path.isfile(args.path):
        show_one(args.path)
    else:
        files = glob.glob(os.path.join(args.path, "*.npy"))
        if not files:
            print(f"No .npy SSIM maps found in {args.path}")
            return
        if args.all:
            show_all(files)
        else:
            show_slider(files)

    plt.show()


if __name__ == "__main__":
    main()
