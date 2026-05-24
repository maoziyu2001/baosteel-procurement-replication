"""Build paper tables and hatch-style figures for supplier-count sensitivity.

This script reads existing results in outputs/supplier_count_sensitivity/.
It does not rerun any algorithm. Outputs include:
- supplier_count_sensitivity_paper_table.csv
- supplier_count_sensitivity_paper_table_zh.csv
- supplier_count_sensitivity_detailed_comparison.csv
- supplier_count_selected_supplier_bar_academic.(png|pdf)
- supplier_count_discount_amount_bar_academic.(png|pdf)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "outputs" / "supplier_count_sensitivity"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "paper_outputs"

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


def load_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = input_dir / "summary.csv"
    comparison_path = input_dir / "comparison_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Cannot find {summary_path}")
    if not comparison_path.exists():
        raise FileNotFoundError(f"Cannot find {comparison_path}")
    summary = pd.read_csv(summary_path)
    comparison = pd.read_csv(comparison_path)
    return summary, comparison


def build_tables(summary: pd.DataFrame, comparison: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ga = summary[summary["algorithm_name"] == "ga_greedy_only"].set_index("supplier_count")
    manual = summary[summary["algorithm_name"] == "manual"].set_index("supplier_count")
    lb = comparison[comparison["method"] == "lower_bound"].set_index("supplier_count")

    rows = []
    detailed_rows = []
    for supplier_count in sorted(ga.index.astype(int).unique()):
        ga_row = ga.loc[supplier_count]
        manual_row = manual.loc[supplier_count] if supplier_count in manual.index else pd.Series(dtype=float)
        lb_row = lb.loc[supplier_count] if supplier_count in lb.index else pd.Series(dtype=float)

        rows.append(
            {
                "supplier_count": supplier_count,
                "ga_greedy_mean_cost": ga_row["best_cost_mean"],
                "manual_cost": manual_row.get("best_cost_mean", np.nan),
                "lower_bound": lb_row.get("cost_mean", np.nan),
                "gap_pct": ga_row["gap_mean"],
                "runtime_sec": ga_row["runtime_mean"],
                "iter_to_final": ga_row["iter_to_final_mean"],
                "feasible_rate": ga_row["feasible_rate"],
                "selected_supplier_count": ga_row["selected_supplier_count_mean"],
                "discount_amount": ga_row["total_discount_amount_mean"],
            }
        )

        for algorithm_name, row in [("GA-hybrid", ga_row), ("人工经验算法", manual_row)]:
            detailed_rows.append(
                {
                    "supplier_count": supplier_count,
                    "algorithm_name": algorithm_name,
                    "cost": row.get("best_cost_mean", np.nan),
                    "gap_pct": row.get("gap_mean", np.nan),
                    "runtime_sec": row.get("runtime_mean", np.nan),
                    "feasible_rate": row.get("feasible_rate", np.nan),
                    "selected_supplier_count": row.get("selected_supplier_count_mean", np.nan),
                    "discount_amount": row.get("total_discount_amount_mean", np.nan),
                    "discount_saving_rate": row.get("discount_saving_rate_mean", np.nan),
                }
            )

    paper = pd.DataFrame(rows)
    paper_zh = paper.rename(
        columns={
            "supplier_count": "供应商数量",
            "ga_greedy_mean_cost": "GA-hybrid平均成本",
            "manual_cost": "人工算法成本",
            "lower_bound": "理论下界",
            "gap_pct": "Gap(%)",
            "runtime_sec": "运行时间(s)",
            "iter_to_final": "达到最终解迭代数",
            "feasible_rate": "可行率",
            "selected_supplier_count": "启用供应商数量",
            "discount_amount": "折扣金额",
        }
    )
    detailed = pd.DataFrame(detailed_rows)
    return paper, paper_zh, detailed


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.6f")


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_pair_bar(
    detailed: pd.DataFrame,
    value_col: str,
    ylabel: str,
    stem: str,
    output_dir: Path,
    scale: float = 1.0,
) -> None:
    counts = sorted(detailed["supplier_count"].dropna().astype(int).unique())
    labels = {"GA-hybrid": "GA-hybrid", "人工经验算法": "人工经验算法"}
    hatches = {"GA-hybrid": "///", "人工经验算法": "..."}
    facecolors = {"GA-hybrid": "#f2f2f2", "人工经验算法": "#d9d9d9"}
    edgecolor = "#222222"

    x = np.arange(len(counts))
    width = 0.32
    offsets = {"GA-hybrid": -width / 2, "人工经验算法": width / 2}

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for algorithm_name in ["GA-hybrid", "人工经验算法"]:
        df = detailed[detailed["algorithm_name"] == algorithm_name].set_index("supplier_count")
        y = np.array([df.loc[c, value_col] / scale if c in df.index else np.nan for c in counts], dtype=float)
        ax.bar(
            x + offsets[algorithm_name],
            y,
            width=width,
            color=facecolors[algorithm_name],
            edgecolor=edgecolor,
            linewidth=0.9,
            hatch=hatches[algorithm_name],
            zorder=3,
        )

    handles = [
        Patch(
            facecolor=facecolors[name],
            edgecolor=edgecolor,
            hatch=hatches[name],
            label=labels[name],
            linewidth=0.9,
        )
        for name in ["GA-hybrid", "人工经验算法"]
    ]
    ax.legend(
        handles=handles,
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
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_max = float(np.nanmax(detailed[value_col].to_numpy(dtype=float) / scale))
    ax.set_ylim(0, y_max * 1.22 if y_max > 0 else 1)
    fig.subplots_adjust(top=0.84, left=0.105, right=0.985, bottom=0.15)
    save_figure(fig, output_dir, stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build supplier-count sensitivity paper tables and figures.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_chinese_font()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary, comparison = load_inputs(args.input_dir)
    paper, paper_zh, detailed = build_tables(summary, comparison)
    save_csv(paper, args.output_dir / "supplier_count_sensitivity_paper_table.csv")
    save_csv(paper_zh, args.output_dir / "supplier_count_sensitivity_paper_table_zh.csv")
    save_csv(detailed, args.output_dir / "supplier_count_sensitivity_detailed_comparison.csv")

    plot_pair_bar(
        detailed,
        value_col="selected_supplier_count",
        ylabel="启用供应商数量",
        stem="supplier_count_selected_supplier_bar_academic",
        output_dir=args.output_dir,
    )
    plot_pair_bar(
        detailed,
        value_col="discount_amount",
        ylabel="折扣金额（亿元）",
        stem="supplier_count_discount_amount_bar_academic",
        output_dir=args.output_dir,
        scale=1e8,
    )
    print(f"Saved paper tables and figures to {args.output_dir}")


if __name__ == "__main__":
    main()
