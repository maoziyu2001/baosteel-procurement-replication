"""Create a publication-style Gap bar chart for supplier-count sensitivity.

The script reads existing sensitivity-analysis CSV files only. It plots the
Gap to lower bound for the algorithm and manual baseline with hatch fills, so
the figure remains distinguishable in grayscale printing.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "outputs" / "supplier_count_sensitivity"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "figures_paper"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def configure_chinese_font() -> None:
    preferred_fonts = [
        "PingFang SC",
        "Songti SC",
        "Heiti SC",
        "STHeiti",
        "Arial Unicode MS",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in preferred_fonts:
        if font in available:
            plt.rcParams["font.sans-serif"] = [font]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "axes.linewidth": 0.9,
            "hatch.linewidth": 0.8,
        }
    )


def load_gap_data(input_dir: Path) -> pd.DataFrame:
    path = input_dir / "comparison_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find comparison summary: {path}")
    df = pd.read_csv(path)
    required = {"supplier_count", "method", "gap_to_lower_bound_pct_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df[df["method"].isin(["ga_greedy_only", "manual"])].copy()


def plot_gap_bar(data: pd.DataFrame, output_dir: Path) -> None:
    counts = sorted(data["supplier_count"].dropna().astype(int).unique())
    methods = ["ga_greedy_only", "manual"]
    labels = {"ga_greedy_only": "GA-hybrid", "manual": "人工经验算法"}
    hatches = {"ga_greedy_only": "///", "manual": "..."}
    facecolors = {"ga_greedy_only": "#f2f2f2", "manual": "#d9d9d9"}
    edgecolor = "#222222"

    x = np.arange(len(counts))
    width = 0.32
    offsets = {"ga_greedy_only": -width / 2, "manual": width / 2}

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for method in methods:
        df = data[data["method"] == method].set_index("supplier_count")
        y = np.array(
            [
                df.loc[c, "gap_to_lower_bound_pct_mean"] if c in df.index else np.nan
                for c in counts
            ],
            dtype=float,
        )
        ax.bar(
            x + offsets[method],
            y,
            width=width,
            color=facecolors[method],
            edgecolor=edgecolor,
            linewidth=0.9,
            hatch=hatches[method],
            zorder=3,
        )

    legend_handles = [
        Patch(
            facecolor=facecolors[method],
            edgecolor=edgecolor,
            hatch=hatches[method],
            label=labels[method],
            linewidth=0.9,
        )
        for method in methods
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        frameon=False,
        handlelength=2.2,
        columnspacing=2.2,
        borderaxespad=0.0,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in counts])
    ax.set_xlabel("候选供应商数量")
    ax.set_ylabel("Gap to LB（%）")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_max = float(np.nanmax(data["gap_to_lower_bound_pct_mean"].to_numpy(dtype=float)))
    ax.set_ylim(0, y_max * 1.22)
    fig.subplots_adjust(top=0.84, left=0.105, right=0.985, bottom=0.15)

    output_dir.mkdir(parents=True, exist_ok=True)
    for stem in ["supplier_count_gap_bar_academic", "supplier_count_gap_runtime_dual_axis"]:
        fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
        fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot academic Gap bar chart from sensitivity results.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_chinese_font()
    data = load_gap_data(args.input_dir)
    plot_gap_bar(data, args.output_dir)
    print(f"Saved academic Gap figure to {args.output_dir}")


if __name__ == "__main__":
    main()
