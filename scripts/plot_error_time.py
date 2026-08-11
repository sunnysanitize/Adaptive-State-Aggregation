import argparse
import json
import sys
from pathlib import Path

import matplotlib

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

LABELS = {
    "configs/maze_200.json": r"$\gamma=0.95$, $200^2$",
    "configs/maze_500.json": r"$\gamma=0.95$, $500^2$",
    "configs/maze_1000.json": r"$\gamma=0.95$, $1000^2$",
    "configs/maze_200_sensitivity.json": r"$\gamma=0.999$, $200^2$",
    "configs/maze_500_sensitivity.json": r"$\gamma=0.999$, $500^2$",
}


def figure(doc, mode):
    c = THEME[mode]
    rows = doc["rows"]
    configs = list(dict.fromkeys(r["config"] for r in rows))

    fig, axes = plt.subplots(1, len(configs), figsize=(6.4 * len(configs), 5.0), dpi=200)
    if len(configs) == 1:
        axes = [axes]
    fig.patch.set_facecolor(c["surface"])

    for ax, config in zip(axes, configs, strict=True):
        ax.set_facecolor(c["surface"])
        gamma = next(r["gamma"] for r in rows if r["config"] == config)
        budgets = doc["budgets"][str(gamma)]

        for b in budgets:
            ax.axvline(b, color=c["grid"], linewidth=1.0, zorder=0)

        for row in (r for r in rows if r["config"] == config):
            ## Colour carries the solver, dash carries the thread count -- so
            ## the two encodings never have to be told apart by hue alone.
            colour = c["vi"] if row["arm"] == "vi" else c["agg"]
            solid = row["threads"] > 1
            ax.plot(
                row["wall_s"], row["err_inf"],
                linewidth=2 if solid else 1.4,
                linestyle="-" if solid else (0, (4, 2)),
                color=colour, zorder=3,
                label=f"{'value iteration' if row['arm'] == 'vi' else 'aggregation'}"
                      f", {row['threads']} thread{'s' if row['threads'] > 1 else ''}",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("solver wall-clock (s)", fontsize=9, color=c["ink2"])
        ax.set_ylabel(r"$\ell_\infty$ error", fontsize=9, color=c["ink2"])
        ax.set_title(LABELS.get(config, config), fontsize=10.5, color=c["ink"],
                     pad=8, loc="left")
        ax.grid(True, which="major", color=c["grid"], linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(c["axis"])
            ax.spines[side].set_linewidth(1.0)
        ax.tick_params(colors=c["muted"], labelsize=8, which="both")

        legend = ax.legend(frameon=False, fontsize=8, loc="lower left")
        for text in legend.get_texts():
            text.set_color(c["ink2"])

    fig.suptitle(
        "At equal wall-clock, aggregation flattens at its approximation floor "
        "while value iteration keeps descending",
        fontsize=12.5, color=c["ink"], x=0.005, ha="left", y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/plot_error_time.py")
    parser.add_argument("curves", type=Path)
    parser.add_argument("--outdir", type=Path, default=Path("results/figures"))
    args = parser.parse_args(argv)

    plt.rcParams["font.family"] = FONTS
    doc = json.loads(args.curves.read_text())
    args.outdir.mkdir(parents=True, exist_ok=True)

    for mode in ("light", "dark"):
        fig = figure(doc, mode)
        out = args.outdir / f"error_vs_time_{mode}.png"
        fig.savefig(out, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"wrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
