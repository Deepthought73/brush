#!/usr/bin/env python3
"""Explore a grid_sweep.py run: overlay PSNR / SSIM / num_splats curves for all
runs instead of plotting them one by one.

The idea that makes 160 runs tractable: draw every run as a thin line in just
three axes (PSNR, SSIM, num_splats vs step), colour the lines by ONE parameter
you want to study, and draw a bold mean curve per colour-group. Then filter the
other parameters interactively to drill into slices.

Interactive (default):
    python plot_sweep.py [OUT_DIR]
        - radio buttons on the left pick which parameter colours the curves
        - check buttons filter which value of each parameter is shown

Static (for sharing / scripting):
    python plot_sweep.py [OUT_DIR] --static --color-by init-mode [--out fig.png]

OUT_DIR defaults to /tmp/grid_sweep and must contain manifest.csv plus the
per-run CSVs it references.
"""
import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons, RadioButtons

METRICS = ["psnr", "ssim", "num_splats"]
MANIFEST_META = {"run_id", "csv_file", "returncode"}


def value_sort_key(v):
    try:
        return (0, float(v), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(v))


def load_run_csv(path):
    cols = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        names = reader.fieldnames or []
        for n in names:
            cols[n] = []
        for row in reader:
            for n in names:
                cols[n].append(float(row[n]))
    return {n: np.asarray(v) for n, v in cols.items()}


def load_runs(out_dir):
    manifest = os.path.join(out_dir, "manifest.csv")
    with open(manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No runs in {manifest}")

    param_flags = [c for c in rows[0] if c not in MANIFEST_META]
    runs = []
    for r in rows:
        if r.get("returncode", "0") not in ("0", ""):
            continue
        path = os.path.join(out_dir, r["csv_file"])
        if not os.path.exists(path):
            continue
        data = load_run_csv(path)
        if "step" not in data or not all(m in data for m in METRICS):
            continue
        runs.append({"params": {f: r[f] for f in param_flags}, "data": data})

    if not runs:
        raise SystemExit(f"No usable run CSVs found under {out_dir}")
    return param_flags, runs


def param_values(runs, flag):
    return sorted({run["params"][flag] for run in runs}, key=value_sort_key)


def draw(axes, runs, color_by, active, all_color_values):
    cmap = plt.get_cmap("tab10" if len(all_color_values) <= 10 else "tab20")
    color_of = {v: cmap(i % cmap.N) for i, v in enumerate(all_color_values)}

    selected = [
        run for run in runs
        if all(run["params"][f] in vals for f, vals in active.items())
    ]

    groups = {}
    for run in selected:
        groups.setdefault(run["params"][color_by], []).append(run)

    for ax, metric in zip(axes, METRICS):
        ax.clear()
        for value in all_color_values:
            members = groups.get(value, [])
            if not members:
                continue
            color = color_of[value]
            for run in members:
                ax.plot(run["data"]["step"], run["data"][metric],
                        color=color, alpha=0.18, linewidth=0.8)
            min_len = min(len(run["data"][metric]) for run in members)
            stacked = np.stack([run["data"][metric][:min_len] for run in members])
            steps = members[0]["data"]["step"][:min_len]
            ax.plot(steps, stacked.mean(axis=0), color=color, linewidth=2.4)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("iteration")

    handles = [
        plt.Line2D([], [], color=color_of[v], linewidth=2.4,
                   label=f"{color_by.lstrip('-')}={v}")
        for v in all_color_values if v in groups
    ]
    axes[0].legend(handles=handles, fontsize=8, loc="lower right", ncol=2)
    axes[0].set_title(
        f"{len(selected)} runs  |  colour = {color_by.lstrip('-')}  "
        f"(thin = per run, bold = group mean)"
    )


def interactive(param_flags, runs):
    fig = plt.figure(figsize=(14, 9))
    axes = [
        fig.add_axes([0.30, 0.69, 0.67, 0.26]),
        fig.add_axes([0.30, 0.38, 0.67, 0.26]),
        fig.add_axes([0.30, 0.07, 0.67, 0.26]),
    ]

    state = {"color_by": param_flags[0]}
    active = {f: set(param_values(runs, f)) for f in param_flags}
    color_values = {f: param_values(runs, f) for f in param_flags}

    def redraw():
        draw(axes, runs, state["color_by"], active, color_values[state["color_by"]])
        fig.canvas.draw_idle()

    radio_ax = fig.add_axes([0.02, 0.80, 0.22, 0.16])
    radio_ax.set_title("colour by", fontsize=9)
    radio = RadioButtons(radio_ax, param_flags)

    def on_color_by(label):
        state["color_by"] = label
        redraw()

    radio.on_clicked(on_color_by)

    checks = []
    top = 0.74
    for flag in param_flags:
        labels = color_values[flag]
        height = max(0.04, 0.035 * len(labels))
        cax = fig.add_axes([0.02, top - height, 0.22, height])
        cax.set_title(flag.lstrip("-"), fontsize=8, loc="left")
        cb = CheckButtons(cax, labels, [True] * len(labels))

        def make_cb(f):
            def on_check(label):
                if label in active[f]:
                    active[f].discard(label)
                else:
                    active[f].add(label)
                redraw()
            return on_check

        cb.on_clicked(make_cb(flag))
        checks.append(cb)
        top -= height + 0.02

    fig._sweep_widgets = (radio, checks)
    redraw()
    return fig


def resolve_flag(name, param_flags):
    name = name.lstrip("-")
    for f in param_flags:
        if f.lstrip("-") == name:
            return f
    raise SystemExit(f"--color-by {name} not in params {[f.lstrip('-') for f in param_flags]}")


def static(param_flags, runs, color_by, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    active = {f: set(param_values(runs, f)) for f in param_flags}
    draw(list(axes), runs, color_by, active, param_values(runs, color_by))
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Wrote {out_path}")


def print_best(runs, param_flags, top=5):
    ranked = sorted(runs, key=lambda r: r["data"]["psnr"][-1], reverse=True)
    print(f"Top {top} runs by final PSNR:")
    for run in ranked[:top]:
        d = run["data"]
        params = " ".join(f"{f.lstrip('-')}={run['params'][f]}" for f in param_flags)
        print(f"  PSNR={d['psnr'][-1]:.3f} SSIM={d['ssim'][-1]:.4f} "
              f"num_splats={int(d['num_splats'][-1])}  |  {params}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", default="/tmp/grid_sweep")
    ap.add_argument("--static", action="store_true")
    ap.add_argument("--color-by", default=None,
                    help="parameter to colour by, without dashes (e.g. init-mode)")
    ap.add_argument("--out", default="sweep.png", help="output PNG for --static")
    args = ap.parse_args()

    param_flags, runs = load_runs(args.out_dir)
    print(f"Loaded {len(runs)} runs with params {param_flags}")
    print_best(runs, param_flags)

    if args.static:
        color_by = resolve_flag(args.color_by, param_flags) if args.color_by else param_flags[0]
        static(param_flags, runs, color_by, args.out)
    else:
        interactive(param_flags, runs)
        plt.show()


if __name__ == "__main__":
    main()
