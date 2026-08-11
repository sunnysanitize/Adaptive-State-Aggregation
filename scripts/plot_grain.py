import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

## Same slots as plot_scaling.py. Repeated rather than imported because scripts/
## is not a package, and each plotting script here stands alone.
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

## Where the two measured regimes sit on this axis. Both are below the point
## where threading the aggregate sweep starts to pay, which is the whole reason
## the end-to-end runs call the serial kernel.
REGIMES = ((65, r"$\gamma=0.95$" "\n" r"$K \approx 65$"),
           (4156, r"$\gamma=0.999$" "\n" r"$K = 4156$"))


def style(ax, c, xlabel, ylabel, title, ks):
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 64, 1024, 8192, 65536])
    ax.set_xticklabels(["4", "64", "1k", "8k", "64k"])
    ax.minorticks_off()
    ax.set_xlim(min(ks) * 0.7, max(ks) * 1.4)
    ax.set_xlabel(xlabel, fontsize=9, color=c["ink2"])
    ax.set_ylabel(ylabel, fontsize=9, color=c["ink2"])
    ax.set_title(title, fontsize=10.5, color=c["ink"], pad=8, loc="left")
    ax.grid(True, which="major", color=c["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["axis"])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=c["muted"], labelsize=8)


def mark_regimes(ax, c, ks):
    for k, label in REGIMES:
        ax.axvline(k, color=c["muted"], linewidth=1.0, linestyle=(0, (2, 3)), zorder=1)
        ax.annotate(label, (k, ax.get_ylim()[1]), textcoords="offset points",
                    xytext=(4, -22), ha="left", fontsize=7.5, color=c["muted"])


def figure(doc, mode):
    c = THEME[mode]
    rows = doc["rows"]
    ks = np.array([r["k"] for r in rows])
    serial = np.array([r["serial_ns"] for r in rows]) / 1e3
    threaded = np.array([min(r["threaded_ns"].values()) for r in rows]) / 1e3
    speedup = serial / threaded
    crossover = doc["crossover_k"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=200)
    fig.patch.set_facecolor(c["surface"])
    for ax in (left, right):
        ax.set_facecolor(c["surface"])

    left.plot(ks, serial, marker="o", markersize=7, linewidth=2, color=c["vi"],
              markeredgecolor=c["surface"], markeredgewidth=1.6, zorder=3)
    left.plot(ks, threaded, marker="o", markersize=7, linewidth=2, color=c["agg"],
              markeredgecolor=c["surface"], markeredgewidth=1.6, zorder=3)
    left.set_yscale("log")
    style(left, c, "groups $K$", "time per sweep (µs)",
          "One aggregate sweep: serial against best threaded", ks)
    ## Serial runs above threaded at the right edge, so its label goes above and
    ## threaded's below; the reverse puts both on top of a line.
    left.annotate("serial", (ks[-1], serial[-1]), textcoords="offset points",
                  xytext=(-6, 9), ha="right", fontsize=8.5, color=c["ink2"])
    left.annotate("threaded", (ks[-1], threaded[-1]), textcoords="offset points",
                  xytext=(-6, -17), ha="right", fontsize=8.5, color=c["ink2"])

    right.axhline(1.0, color=c["muted"], linewidth=1.6, linestyle=(0, (5, 3)), zorder=1)
    right.plot(ks, speedup, marker="o", markersize=7, linewidth=2, color=c["vi"],
               markeredgecolor=c["surface"], markeredgewidth=1.6, zorder=3)
    right.set_yscale("log", base=2)
    right.set_ylim(0.008, 6)
    right.set_yticks([0.01, 0.1, 1, 4])
    right.set_yticklabels(["0.01x", "0.1x", "1x", "4x"])
    style(right, c, "groups $K$", "threaded ÷ serial", "Threading pays only above "
          f"$K \\approx {crossover}$", ks)
    right.annotate("break-even", (ks[0], 1.0), textcoords="offset points",
                   xytext=(2, 6), ha="left", fontsize=8, color=c["muted"])

    for ax in (left, right):
        mark_regimes(ax, c, ks)

    fig.suptitle(
        "Both measured regimes have too few groups for threading to pay",
        fontsize=12.5, color=c["ink"], x=0.005, ha="left", y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/plot_grain.py")
    parser.add_argument("grain", type=Path)
    parser.add_argument("--outdir", type=Path, default=Path("results/figures"))
    args = parser.parse_args(argv)

    plt.rcParams["font.family"] = FONTS
    doc = json.loads(args.grain.read_text())
    args.outdir.mkdir(parents=True, exist_ok=True)

    for mode in ("light", "dark"):
        fig = figure(doc, mode)
        out = args.outdir / f"aggregate_grain_{mode}.png"
        fig.savefig(out, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"wrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
