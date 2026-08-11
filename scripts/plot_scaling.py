import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

## dataviz reference palette, slots 1 and 2. Validated as a pair in both modes:
## worst CVD dE 24.7 light / 26.8 dark, normal-vision 33.6 / 31.8, all >= 3:1 on
## surface. Both series are also direct-labelled, so identity is never colour
## alone.
THEME = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "vi": "#2a78d6",
        "agg": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "vi": "#3987e5",
        "agg": "#d95926",
    },
}

FONTS = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
ARMS = (("vi", "value iteration"), ("adaptive", "aggregation"))

LABELS = {
    "configs/maze_200.json": r"$\gamma=0.95$, $200^2$",
    "configs/maze_500.json": r"$\gamma=0.95$, $500^2$",
    "configs/maze_1000.json": r"$\gamma=0.95$, $1000^2$",
    "configs/maze_200_sensitivity.json": r"$\gamma=0.999$, $200^2$",
    "configs/maze_500_sensitivity.json": r"$\gamma=0.999$, $500^2$",
}
ORDER = list(LABELS)


def collect(paths: list[Path]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for path in paths:
        for row in json.loads(path.read_text())["rows"]:
            rows.setdefault((row["config"], row["arm"]), []).append(row)
    return rows


## Whatever thread counts the run actually measured, not an assumed ladder. A
## reduced run is still a valid run and must still plot.
def thread_counts(rows: list[dict[str, Any]]) -> list[int]:
    return sorted(int(p) for p in rows[0]["timings"])


def series(
    rows: list[dict[str, Any]], key: str, threads: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ## Spread across seeds, not just the mean: the intervals are narrow here and
    ## a plot that showed only a line would be hiding that, not conveying it.
    per_thread = [[r["timings"][str(p)][key] for r in rows] for p in threads]
    values = np.array([np.mean(v) for v in per_thread])
    lo = np.array([np.percentile(v, 10) for v in per_thread])
    hi = np.array([np.percentile(v, 90) for v in per_thread])
    return values, lo, hi


def style(ax, c, xlabel, ylabel, title, threads):
    ax.set_xscale("log", base=2)
    ax.set_xticks(threads)
    ax.set_xticklabels([str(p) for p in threads])
    ax.minorticks_off()
    ax.set_xlabel(xlabel, fontsize=9, color=c["ink2"])
    ax.set_ylabel(ylabel, fontsize=9, color=c["ink2"])
    ax.set_title(title, fontsize=10, color=c["ink"], pad=8, loc="left")
    ax.grid(True, which="major", color=c["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["axis"])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=c["muted"], labelsize=8)


def figure(rows, mode):
    c = THEME[mode]
    configs = [k for k in ORDER if (k, "vi") in rows]

    fig, axes = plt.subplots(2, len(configs), figsize=(3.5 * len(configs), 7.2), dpi=200)
    fig.patch.set_facecolor(c["surface"])

    for col, config in enumerate(configs):
        top, bottom = axes[0][col], axes[1][col]
        for panel in (top, bottom):
            panel.set_facecolor(c["surface"])

        threads = thread_counts(rows[(config, "vi")])

        for arm, label in ARMS:
            data = rows[(config, arm)]
            colour = c["vi"] if arm == "vi" else c["agg"]

            times, t_lo, t_hi = series(data, "median_s", threads)
            top.fill_between(threads, t_lo, t_hi, color=colour, alpha=0.18, linewidth=0)
            top.plot(threads, times, marker="o", markersize=8, linewidth=2, color=colour,
                     markeredgecolor=c["surface"], markeredgewidth=2, label=label, zorder=3)

            base = times[0]
            speedup = base / times
            bottom.plot(threads, speedup, marker="o", markersize=8, linewidth=2,
                        color=colour, markeredgecolor=c["surface"], markeredgewidth=2,
                        label=label, zorder=3)

            ## Direct label at the right edge, so the two arms are never told
            ## apart by colour alone. VI sits below aggregation everywhere, so
            ## the offsets never collide.
            top.annotate(label, (threads[-1], times[-1]), textcoords="offset points",
                         xytext=(-4, 9 if arm == "adaptive" else -17), ha="right",
                         fontsize=8, color=c["ink2"])
            bottom.annotate(f"{speedup[-1]:.2f}x", (threads[-1], speedup[-1]),
                            textcoords="offset points",
                            xytext=(-4, 9 if arm == "vi" else -16), ha="right",
                            fontsize=8, color=c["ink2"])

        ## Log-log, so perfect scaling is a straight diagonal. On a log x-axis a
        ## linear y-axis would bend y = x into what looks like exponential
        ## growth and squash the measured curves against the floor.
        bottom.plot(threads, threads, linestyle=(0, (5, 3)), linewidth=1.6,
                    color=c["muted"], zorder=1)
        bottom.annotate("perfect scaling", (threads[-1], threads[-1]),
                        textcoords="offset points", xytext=(-2, -13), ha="right",
                        fontsize=8, color=c["muted"])

        ## Absolute time on a linear axis from zero: within a panel the range is
        ## under 2x, where a log axis produces almost no labelled ticks and the
        ## reader cannot recover a value.
        style(top, c, "", "time to target (s)" if col == 0 else "", LABELS[config],
              threads)
        top.set_ylim(0, None)
        style(bottom, c, "threads", "speedup vs 1 thread" if col == 0 else "", "",
              threads)
        bottom.set_yscale("log", base=2)
        bottom.set_ylim(0.9, threads[-1] * 1.25)
        ticks = [t for t in (1, 2, 4, 8, 16) if t <= threads[-1]]
        bottom.set_yticks(ticks)
        bottom.set_yticklabels([str(t) for t in ticks])

    fig.suptitle(
        "Value iteration is faster in absolute time and gains more from threads — "
        "and neither solver comes close to linear",
        fontsize=12.5, color=c["ink"], x=0.005, ha="left", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/plot_scaling.py")
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--outdir", type=Path, default=Path("results/figures"))
    parser.add_argument("--name", default="scaling")
    args = parser.parse_args(argv)

    plt.rcParams["font.family"] = FONTS
    rows = collect(args.results)
    args.outdir.mkdir(parents=True, exist_ok=True)

    for mode in ("light", "dark"):
        fig = figure(rows, mode)
        out = args.outdir / f"{args.name}_{mode}.png"
        fig.savefig(out, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"wrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
