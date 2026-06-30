#!/usr/bin/env python3
"""Plot every per-step metric (duration, lr_mean, loss, psnr, ssim, ...) over
iterations from the CSV written by the single-view-experiment binary
(default /tmp/train.csv).

Every column besides `step` is plotted in its own subplot, so this adapts
automatically if the binary's CSV columns change.

Usage: python plot_train.py [path/to/train.csv]
"""
import csv
import sys

import matplotlib.pyplot as plt


def load(path):
    """Return (steps, {column: [values]}) for every numeric column besides `step`."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        columns = [c for c in reader.fieldnames if c != "step"]
        steps = []
        data = {c: [] for c in columns}
        for row in reader:
            steps.append(int(row["step"]))
            for c in columns:
                data[c].append(float(row[c]))
    return steps, data


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/train.csv"
    steps, data = load(path)

    columns = list(data)
    ncols = 3
    nrows = (len(columns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.flatten()

    # Markers only help when there are few points; over many iterations they
    # just smear the line, so use a plain line past a small threshold.
    marker = "." if len(steps) <= 50 else None
    for ax, col in zip(axes, columns):
        ax.plot(steps, data[col], marker=marker)
        ax.set(title=col, xlabel="iteration", ylabel=col)
        ax.grid(True, alpha=0.3)

    # Hide any unused subplots.
    for ax in axes[len(columns):]:
        ax.axis("off")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
