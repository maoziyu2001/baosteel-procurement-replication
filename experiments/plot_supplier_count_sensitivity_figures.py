"""Create publication-style figures from supplier-count sensitivity results.

The script only reads existing CSV outputs and does not rerun any algorithm.
It focuses on academic-friendly styling: hatch fills, grayscale-compatible
patterns, unobtrusive legends, and high-resolution PNG/PDF exports.
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
    """Use an installed Chinese font so labels render correctly."""
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
            "axes.titlesize": 13,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "axes.linewidth": 0.9,
            "hatch.linewidth": 0.8,
        }
    )


def load_comparison(input_dir: Path) -> pd.DataFrame:
    comparison_path = input_dir / "comparison_summary.csv"
    if comparison_path.exists():
        df = pd.read_csv(comparison_path)
    else:
        raw = pd.read_csv(input_dir / "raw_results.csv")
        rows = []
        for supplier_count, group in raw.groupby("supplier_count"):
            lower_bound = group["lower_bound"].dropna()
            lb_value = float(lower_bound.iloc[0]) if not lower_bound.empty else np.nan
            for algorithm_name, alg_group in group.groupby("algorithm_name"):
                rows.append(
                    {
                        "supplier_count": supplier_count,
                        "method": algorithm_name,
                        "cost_mean": alg_group["best_cost"].mean(),
                        "cost_std": alg_group["best_cost"].std(),
                    }
                )
            rows.append(
                {
                    "supplier_count": supplier_count,
                    "method": "lower_bound",
                    "cost_mean": lb_value,
                    "cost_std": np.nan,
                }
            )
        df = pd.DataFrame(rows)

    required = {"supplier_count", "method", "cost_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in comparison data: {sorted(missing)}")
    return df


def load_convergence(input_dir: Path) -> pd.DataFrame:
    convergence_path = input_dir / "convergence_mean.csv"
    if not convergence_path.exists():
        raise FileNotFoundError(f"Cannot find convergence file: {convergence_path}")
    df = pd.read_csv(convergence_path)
    required = {"supplier_count", "generation", "best_cost_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in convergence data: {sorted(missing)}")
    if "algorithm_name" in df.columns:
        df = df[df["algorithm_name"] == "ga_greedy_only"].copy()
    return df


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_figure_aliases(fig: plt.Figure, output_dir: Path, stems: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
        fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_total_cost_bar(comparison: pd.DataFrame, output_dir: Path) -> None:
    """Grouped bar chart with hatch fills for paper use."""
    methods = ["ga_greedy_only", "manual", "lower_bound"]
    method_labels = {
        "ga_greedy_only": "GA-hybrid",
        "manual": "人工经验算法",
        "lower_bound": "理论下界",
    }
    hatches = {
        "ga_greedy_only": "///",
        "manual": "...",
        "lower_bound": "|||",
    }
    facecolors = {
        "ga_greedy_only": "#f2f2f2",
        "manual": "#d9d9d9",
        "lower_bound": "#ffffff",
    }
    edgecolors = {
        "ga_greedy_only": "#222222",
        "manual": "#222222",
        "lower_bound": "#222222",
    }

    counts = sorted(comparison["supplier_count"].dropna().astype(int).unique())
    x = np.arange(len(counts))
    width = 0.24
    offsets = {
        "ga_greedy_only": -width,
        "manual": 0.0,
        "lower_bound": width,
    }

    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    for method in methods:
        df = comparison[comparison["method"] == method].set_index("supplier_count")
        y = np.array([df.loc[c, "cost_mean"] / 1e9 if c in df.index else np.nan for c in counts])
        yerr = np.array(
            [
                df.loc[c, "cost_std"] / 1e9
                if c in df.index and "cost_std" in df.columns and pd.notna(df.loc[c, "cost_std"])
                else 0.0
                for c in counts
            ]
        )
        ax.bar(
            x + offsets[method],
            y,
            width=width,
            yerr=yerr,
            capsize=2.5,
            color=facecolors[method],
            edgecolor=edgecolors[method],
            linewidth=0.85,
            hatch=hatches[method],
            error_kw={"elinewidth": 0.85, "capthick": 0.85},
            zorder=3,
        )

    legend_handles = [
        Patch(
            facecolor=facecolors[method],
            edgecolor=edgecolors[method],
            hatch=hatches[method],
            label=method_labels[method],
            linewidth=0.85,
        )
        for method in methods
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.8,
        borderaxespad=0.0,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in counts])
    ax.set_xlabel("候选供应商数量")
    ax.set_ylabel("成本（十亿元）")
    # ax.set_title("不同供应商数量下的成本对比", pad=28)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_max = np.nanmax(comparison["cost_mean"].to_numpy(dtype=float) / 1e9)
    ax.set_ylim(0, y_max * 1.18)
    fig.subplots_adjust(top=0.82, left=0.10, right=0.985, bottom=0.14)
    save_figure(fig, output_dir, "supplier_count_total_cost_bar_academic")


def plot_convergence_curve(convergence: pd.DataFrame, output_dir: Path) -> None:
    """Convergence curve with line styles and markers for grayscale printing."""
    line_styles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    gray_levels = ["#111111", "#2b2b2b", "#454545", "#5f5f5f", "#777777", "#8c8c8c", "#a0a0a0"]

    counts = sorted(convergence["supplier_count"].dropna().astype(int).unique())
    fig, ax = plt.subplots(figsize=(8.4, 5.0))

    for idx, supplier_count in enumerate(counts):
        group = convergence[convergence["supplier_count"] == supplier_count].sort_values("generation")
        marker_every = max(1, len(group) // 12)
        ax.plot(
            group["generation"],
            group["best_cost_mean"] / 1e9,
            linestyle=line_styles[idx % len(line_styles)],
            marker=markers[idx % len(markers)],
            markevery=marker_every,
            markersize=4.0,
            linewidth=1.45,
            color=gray_levels[idx % len(gray_levels)],
            label=f"{supplier_count}家供应商",
            zorder=3,
        )

    ax.set_xlabel("迭代次数")
    ax.set_ylabel("平均最优总成本/亿元")
    # ax.set_title("不同供应商数量下的收敛曲线", pad=34)
    ax.grid(axis="both", linestyle="--", linewidth=0.55, alpha=0.42, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.4,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(top=0.78, left=0.11, right=0.985, bottom=0.14)
    save_figure_aliases(
        fig,
        output_dir,
        ["supplier_count_convergence_curve_academic", "supplier_count_convergence_curve"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot academic supplier-count sensitivity figures from existing CSV files."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_chinese_font()
    comparison = load_comparison(args.input_dir)
    convergence = load_convergence(args.input_dir)
    plot_total_cost_bar(comparison, args.output_dir)
    plot_convergence_curve(convergence, args.output_dir)
    print(f"Saved academic figures to {args.output_dir}")


if __name__ == "__main__":
    main()
