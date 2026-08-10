import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

## dataviz reference palette. Slot 1 blue carries the measured series; the bound
## is chrome, not a series -- muted ink, dashed, so identity is never colour-alone.
THEME = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": "#2a78d6",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": "#3987e5",
    },
}

FONTS = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def figure(summary: list[dict[str, Any]], rows: list[dict[str, Any]], mode: str) -> Any:
    c = THEME[mode]
    eps = [s["eps"] for s in summary]
    err = [s["err_mean"] for s in summary]
    bound = [s["bound"] for s in summary]

    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=200)
    fig.patch.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])

    ax.plot(
        eps,
        bound,
        linestyle=(0, (5, 3)),
        linewidth=2,
        color=c["muted"],
        label=r"Theoretical bound  $2\varepsilon/(1-\gamma)$",
        zorder=2,
    )

    ## Every seed is drawn, not just the mean -- on this instance they coincide
    ## exactly, and a plot that hid that would hide the finding.
    ax.scatter(
        [r["eps"] for r in rows],
        [r["err_inf"] for r in rows],
        s=30,
        color=c["series"],
        alpha=0.35,
        linewidths=0,
        zorder=3,
    )
    ax.plot(
        eps,
        err,
        marker="o",
        markersize=9,
        linewidth=2,
        color=c["series"],
        markeredgecolor=c["surface"],
        markeredgewidth=2,
        label="Measured final $\\ell_\\infty$ error (20 seeds)",
        zorder=4,
    )

    for x, y in zip(eps, err, strict=True):
        ax.annotate(
            f"{y:.3g}",
            (x, y),
            textcoords="offset points",
            xytext=(0, -18),
            ha="center",
            fontsize=9,
            color=c["ink2"],
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(eps)
    ax.set_xticklabels([f"{e:g}" for e in eps])
    ## Log minor labels (6x10^-2, 2x10^-1, ...) collide with the eps ticks.
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(min(err) / 2.4, max(bound) * 1.8)
    ax.set_xlabel(r"aggregation width  $\varepsilon$", fontsize=10, color=c["ink2"])
    ax.set_ylabel(r"final $\ell_\infty$ error", fontsize=10, color=c["ink2"])
    ax.set_title(
        "Error scales with $\\varepsilon$ — standard maze $200^2$, $\\gamma=0.95$",
        fontsize=12,
        color=c["ink"],
        pad=14,
        loc="left",
    )

    ax.grid(True, which="both", color=c["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["axis"])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=c["muted"], labelsize=9, which="both")

    legend = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for text in legend.get_texts():
        text.set_color(c["ink2"])

    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/plot_sweep.py")
    parser.add_argument("sweep", type=Path)
    parser.add_argument("--outdir", type=Path, default=Path("results/figures"))
    args = parser.parse_args(argv)

    plt.rcParams["font.family"] = FONTS
    data = json.loads(args.sweep.read_text())
    args.outdir.mkdir(parents=True, exist_ok=True)

    for mode in ("light", "dark"):
        fig = figure(data["summary"], data["rows"], mode)
        out = args.outdir / f"{args.sweep.stem}_{mode}.png"
        fig.savefig(out, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"wrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
