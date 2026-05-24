#!/usr/bin/env python3
"""Generate figures and narrative analysis from procurement output CSV files."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = OUTPUT_DIR / "procurement_scheme_analysis.md"
ALGORITHMS = ["manual", "ga_greedy_only"]
ALG_LABELS = {"manual": "人工经验算法", "ga_greedy_only": "GA-贪心算法"}


def configure_chinese_font() -> None:
    """Configure matplotlib so Chinese labels render correctly."""
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
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in installed:
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    else:
        print("Warning: no preferred Chinese font found; Chinese text may not render correctly.")
    plt.rcParams["axes.unicode_minus"] = False


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def discover_cases(output_dir: Path) -> list[str]:
    cases = []
    for path in output_dir.glob("*_cost_summary.csv"):
        case = path.name.replace("_cost_summary.csv", "")
        df = read_csv(path)
        if set(ALGORITHMS).issubset(set(df["algorithm_name"])):
            cases.append(case)
    return sorted(cases)


def safe_div(num: float, den: float) -> float:
    return num / den if abs(den) > 1e-12 else np.nan


def pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value * 100:.2f}%"


def money(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value / 1e8:.2f}亿元"


def num(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.4g}"


def load_case_tables(case: str) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {
        "cost": read_csv(OUTPUT_DIR / f"{case}_cost_summary.csv"),
        "cost_bar": read_csv(OUTPUT_DIR / f"{case}_cost_component_bar_data.csv"),
    }
    for alg in ALGORITHMS:
        prefix = OUTPUT_DIR / f"{case}_{alg}"
        tables[f"{alg}_supplier_structure"] = read_csv(Path(f"{prefix}_supplier_structure_metrics.csv"))
        tables[f"{alg}_cross_base"] = read_csv(Path(f"{prefix}_cross_base_metrics.csv"))
        tables[f"{alg}_inventory"] = read_csv(Path(f"{prefix}_inventory_strategy_metrics.csv"))
        tables[f"{alg}_base"] = read_csv(Path(f"{prefix}_base_summary.csv"))
        tables[f"{alg}_supplier"] = read_csv(Path(f"{prefix}_supplier_summary.csv"))
        tables[f"{alg}_pareto"] = read_csv(Path(f"{prefix}_supplier_pareto_data.csv"))
        tables[f"{alg}_heatmap"] = read_csv(Path(f"{prefix}_heatmap_base_supplier.csv"))
        tables[f"{alg}_inventory_line"] = read_csv(Path(f"{prefix}_inventory_line_data.csv"))
        tables[f"{alg}_grade"] = read_csv(Path(f"{prefix}_grade_cost_summary.csv"))
    return tables


def row(df: pd.DataFrame, alg: str | None = None) -> pd.Series:
    if alg is not None and "algorithm_name" in df.columns:
        df = df[df["algorithm_name"] == alg]
    return df.iloc[0]


def save_fig(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return str(path.relative_to(OUTPUT_DIR))


def plot_cost_components(case: str, tables: dict[str, pd.DataFrame]) -> str:
    df = tables["cost_bar"].copy()
    df = df[df["algorithm_name"].isin(ALGORITHMS)]
    components = ["purchase_cost", "transport_cost", "inventory_cost", "fixed_supplier_cost"]
    x = np.arange(len(components))
    width = 0.36
    plt.figure(figsize=(8, 4.5))
    for idx, alg in enumerate(ALGORITHMS):
        vals = []
        for comp in components:
            sub = df[(df["algorithm_name"] == alg) & (df["cost_component"] == comp)]
            vals.append(sub["cost_value"].iloc[0] / 1e8 if not sub.empty else 0)
        plt.bar(x + (idx - 0.5) * width, vals, width=width, label=ALG_LABELS[alg])
    plt.xticks(x, ["采购成本", "运输成本", "库存成本", "合作成本"], rotation=0)
    plt.ylabel("成本（亿元）")
    plt.title(f"{case}：成本分项对比")
    plt.legend()
    return save_fig(FIG_DIR / f"{case}_cost_components.png")


def plot_supplier_pareto(case: str, tables: dict[str, pd.DataFrame]) -> str:
    plt.figure(figsize=(8, 4.8))
    for alg in ALGORITHMS:
        df = tables[f"{alg}_pareto"].sort_values("rank")
        plt.plot(df["rank"], df["cumulative_purchase_share"], marker="o", linewidth=1.8, label=ALG_LABELS[alg])
    plt.ylim(0, 1.05)
    plt.xlabel("按采购量排序的供应商位次")
    plt.ylabel("累计采购量占比")
    plt.title(f"{case}：供应商采购量 Pareto 曲线")
    plt.grid(alpha=0.25)
    plt.legend()
    return save_fig(FIG_DIR / f"{case}_supplier_pareto.png")


def plot_structure_metrics(case: str, tables: dict[str, pd.DataFrame]) -> str:
    metrics = [
        ("selected_supplier_count", "启用供应商数"),
        ("top5_supplier_share", "Top-5采购占比"),
        ("herfindahl_index", "HHI集中度"),
        ("long_contract_supplier_purchase_share", "长协采购占比"),
    ]
    x = np.arange(len(metrics))
    width = 0.36
    plt.figure(figsize=(9, 4.8))
    for idx, alg in enumerate(ALGORITHMS):
        vals = [row(tables[f"{alg}_supplier_structure"])[m] for m, _ in metrics]
        plt.bar(x + (idx - 0.5) * width, vals, width=width, label=ALG_LABELS[alg])
    plt.xticks(x, [label for _, label in metrics], rotation=15, ha="right")
    plt.title(f"{case}：供应商结构指标")
    plt.legend()
    return save_fig(FIG_DIR / f"{case}_supplier_structure.png")


def plot_cross_base_metrics(case: str, tables: dict[str, pd.DataFrame]) -> str:
    metrics = [
        ("multi_base_supplier_ratio", "多基地供应商比例"),
        ("multi_base_purchase_share", "多基地采购占比"),
        ("avg_supplier_count_per_base", "基地平均供应商数"),
        ("avg_base_count_per_selected_supplier", "供应商平均服务基地数"),
    ]
    x = np.arange(len(metrics))
    width = 0.36
    plt.figure(figsize=(9, 4.8))
    for idx, alg in enumerate(ALGORITHMS):
        vals = [row(tables[f"{alg}_cross_base"])[m] for m, _ in metrics]
        plt.bar(x + (idx - 0.5) * width, vals, width=width, label=ALG_LABELS[alg])
    plt.xticks(x, [label for _, label in metrics], rotation=15, ha="right")
    plt.title(f"{case}：跨基地协同指标")
    plt.legend()
    return save_fig(FIG_DIR / f"{case}_cross_base_metrics.png")


def plot_inventory_metrics(case: str, tables: dict[str, pd.DataFrame]) -> str:
    metrics = [
        ("avg_ending_inventory_iron", "平均期末铁库存"),
        ("avg_safety_stock_redundancy", "安全库存冗余"),
        ("avg_capacity_utilization", "仓储能力利用率"),
        ("inventory_cost_share", "库存成本占比"),
    ]
    x = np.arange(len(metrics))
    width = 0.36
    plt.figure(figsize=(9, 4.8))
    for idx, alg in enumerate(ALGORITHMS):
        vals = [row(tables[f"{alg}_inventory"])[m] for m, _ in metrics]
        # Keep the inventory quantity readable in 10k tons equivalent scale.
        vals[0] = vals[0] / 10000
        plt.bar(x + (idx - 0.5) * width, vals, width=width, label=ALG_LABELS[alg])
    plt.xticks(x, ["平均期末铁库存/万", "安全库存冗余", "仓储能力利用率", "库存成本占比"], rotation=15, ha="right")
    plt.title(f"{case}：库存策略指标")
    plt.legend()
    return save_fig(FIG_DIR / f"{case}_inventory_metrics.png")


def plot_heatmaps(case: str, tables: dict[str, pd.DataFrame]) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    vmax = 0.0
    matrices = []
    for alg in ALGORITHMS:
        df = tables[f"{alg}_heatmap"]
        mat = df.pivot(index="base_id", columns="supplier_id", values="total_purchase_qty").fillna(0.0)
        matrices.append((alg, mat))
        vmax = max(vmax, mat.to_numpy().max())
    for ax, (alg, mat) in zip(axes, matrices):
        im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
        ax.set_title(ALG_LABELS[alg])
        ax.set_xlabel("供应商")
        ax.set_ylabel("基地")
        ax.set_yticks(np.arange(len(mat.index)))
        ax.set_yticklabels(mat.index)
        ax.set_xticks(np.arange(len(mat.columns)))
        ax.set_xticklabels(mat.columns, rotation=90, fontsize=7)
    fig.suptitle(f"{case}：基地-供应商采购热力图")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="采购量")
    path = FIG_DIR / f"{case}_base_supplier_heatmap.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return str(path.relative_to(OUTPUT_DIR))


def plot_inventory_lines(case: str, tables: dict[str, pd.DataFrame]) -> str:
    fig, axes = plt.subplots(len(ALGORITHMS), 1, figsize=(9, 6.2), sharex=True)
    if len(ALGORITHMS) == 1:
        axes = [axes]
    for ax, alg in zip(axes, ALGORITHMS):
        df = tables[f"{alg}_inventory_line"]
        for base_id, group in df.groupby("base_id"):
            group = group.sort_values("period")
            ax.plot(group["period"], group["ending_inventory_iron"], marker="o", label=f"基地 {base_id}")
        safety = df.groupby("period")["safety_stock"].mean().reset_index()
        ax.plot(safety["period"], safety["safety_stock"], linestyle="--", color="black", label="平均安全库存")
        ax.set_title(ALG_LABELS[alg])
        ax.set_ylabel("铁元素库存")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=3)
    axes[-1].set_xlabel("周期")
    fig.suptitle(f"{case}：库存水平折线图")
    path = FIG_DIR / f"{case}_inventory_lines.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return str(path.relative_to(OUTPUT_DIR))


def summarize_case(case: str, tables: dict[str, pd.DataFrame]) -> dict[str, float]:
    cost = tables["cost"]
    manual_cost = row(cost, "manual")
    ga_cost = row(cost, "ga_greedy_only")
    manual_struct = row(tables["manual_supplier_structure"])
    ga_struct = row(tables["ga_greedy_only_supplier_structure"])
    manual_cross = row(tables["manual_cross_base"])
    ga_cross = row(tables["ga_greedy_only_cross_base"])
    manual_inv = row(tables["manual_inventory"])
    ga_inv = row(tables["ga_greedy_only_inventory"])
    manual_grade = row(tables["manual_grade"][tables["manual_grade"]["base_id"].astype(str) == "ALL"])
    ga_grade = row(tables["ga_greedy_only_grade"][tables["ga_greedy_only_grade"]["base_id"].astype(str) == "ALL"])
    return {
        "manual_total_cost": manual_cost["total_cost"],
        "ga_total_cost": ga_cost["total_cost"],
        "cost_rel": safe_div(ga_cost["total_cost"] - manual_cost["total_cost"], manual_cost["total_cost"]),
        "manual_selected": manual_struct["selected_supplier_count"],
        "ga_selected": ga_struct["selected_supplier_count"],
        "manual_top5": manual_struct["top5_supplier_share"],
        "ga_top5": ga_struct["top5_supplier_share"],
        "manual_hhi": manual_struct["herfindahl_index"],
        "ga_hhi": ga_struct["herfindahl_index"],
        "manual_multi_share": manual_cross["multi_base_purchase_share"],
        "ga_multi_share": ga_cross["multi_base_purchase_share"],
        "manual_avg_suppliers_per_base": manual_cross["avg_supplier_count_per_base"],
        "ga_avg_suppliers_per_base": ga_cross["avg_supplier_count_per_base"],
        "manual_inventory_redundancy": manual_inv["avg_safety_stock_redundancy"],
        "ga_inventory_redundancy": ga_inv["avg_safety_stock_redundancy"],
        "manual_inventory_cost_share": manual_inv["inventory_cost_share"],
        "ga_inventory_cost_share": ga_inv["inventory_cost_share"],
        "manual_grade": manual_grade["weighted_average_grade"],
        "ga_grade": ga_grade["weighted_average_grade"],
        "manual_unit_iron_cost": manual_cost["average_unit_iron_cost"],
        "ga_unit_iron_cost": ga_cost["average_unit_iron_cost"],
    }


def make_case_figures(case: str, tables: dict[str, pd.DataFrame]) -> list[str]:
    return [
        plot_cost_components(case, tables),
        plot_supplier_pareto(case, tables),
        plot_structure_metrics(case, tables),
        plot_cross_base_metrics(case, tables),
        plot_inventory_metrics(case, tables),
        plot_heatmaps(case, tables),
        plot_inventory_lines(case, tables),
    ]


def build_markdown(cases: list[str], summaries: dict[str, dict[str, float]], figures: dict[str, list[str]]) -> str:
    lines: list[str] = []
    lines.append("# 采购方案结构性分析报告\n")
    lines.append("本报告基于 `outputs/` 目录中 GA-greedy-only（优化后）与 manual（优化前）结果数据自动生成，重点回应审稿人关于供应商数量、跨基地协同、库存策略和优化前后协同效果展示的意见。\n")

    lines.append("## 一、详细分析版本\n")
    for case in cases:
        s = summaries[case]
        lines.append(f"### {case}\n")
        lines.append(
            f"- 成本表现：manual 总成本为 {money(s['manual_total_cost'])}，GA-greedy-only 总成本为 {money(s['ga_total_cost'])}，"
            f"相对变化为 {pct(s['cost_rel'])}。当前数据中，GA-greedy-only 并未在该算例上获得成本优势，说明纯贪心解码 GA 的搜索结果仍有改进空间。"
        )
        lines.append(
            f"- 供应商结构：manual 启用 {s['manual_selected']:.0f} 家供应商，GA-greedy-only 启用 {s['ga_selected']:.0f} 家供应商；"
            f"Top-5 采购份额分别为 {pct(s['manual_top5'])} 和 {pct(s['ga_top5'])}，HHI 分别为 {s['manual_hhi']:.4f} 和 {s['ga_hhi']:.4f}。"
        )
        lines.append(
            f"- 跨基地协同：manual 的多基地共享供应商采购占比为 {pct(s['manual_multi_share'])}，"
            f"GA-greedy-only 为 {pct(s['ga_multi_share'])}；平均每个基地使用供应商数分别为 "
            f"{s['manual_avg_suppliers_per_base']:.2f} 和 {s['ga_avg_suppliers_per_base']:.2f}。"
        )
        lines.append(
            f"- 库存策略：manual 平均安全库存冗余为 {s['manual_inventory_redundancy']:.4f}，"
            f"GA-greedy-only 为 {s['ga_inventory_redundancy']:.4f}；库存成本占比分别为 "
            f"{pct(s['manual_inventory_cost_share'])} 和 {pct(s['ga_inventory_cost_share'])}。"
        )
        lines.append(
            f"- 配矿与单位铁成本：manual 加权平均品位为 {s['manual_grade']:.4f}，GA-greedy-only 为 {s['ga_grade']:.4f}；"
            f"单位铁元素综合成本分别为 {s['manual_unit_iron_cost']:.2f} 和 {s['ga_unit_iron_cost']:.2f}。"
        )
        lines.append("\n图形输出：")
        for fig in figures[case]:
            lines.append(f"- `{fig}`")
        lines.append("")

    lines.append("## 二、用于论文对应章节的内容版本\n")
    lines.append(
        "为进一步揭示采购优化方案的结构性特征，本文从成本构成、供应商集中度、跨基地共享供应商、库存安全冗余和配矿质量等维度，对优化前后的采购方案进行了对比分析。"
        "结果表明，不同算法生成的采购方案不仅在总成本上存在差异，而且在供应商组合和基地间协同方式上呈现出明显不同的结构特征。"
    )
    for case in cases:
        s = summaries[case]
        lines.append(
            f"以 {case} 为例，优化前 manual 方案启用 {s['manual_selected']:.0f} 家供应商，"
            f"优化后 GA-greedy-only 方案启用 {s['ga_selected']:.0f} 家供应商；"
            f"其 Top-5 供应商采购份额由 {pct(s['manual_top5'])} 变为 {pct(s['ga_top5'])}，"
            f"采购集中度 HHI 由 {s['manual_hhi']:.4f} 变为 {s['ga_hhi']:.4f}。"
            f"跨基地协同方面，多基地共享供应商采购占比由 {pct(s['manual_multi_share'])} 变为 {pct(s['ga_multi_share'])}。"
            f"库存策略方面，平均安全库存冗余由 {s['manual_inventory_redundancy']:.4f} 变为 {s['ga_inventory_redundancy']:.4f}。"
        )
    lines.append(
        "上述结果说明，采购方案优化会改变供应商选择、基地间供应商共享关系以及库存配置方式。"
        "本文据此增加供应商 Pareto 曲线、基地-供应商采购热力图、库存水平折线图和成本分项柱状图，以直观展示优化前后采购协同模式的差异。"
    )
    lines.append("")

    lines.append("## 三、审稿人回复版本\n")
    lines.append("感谢审稿人的建议。根据该意见，本文补充了采购方案结构性分析，从供应商数量、供应商集中度、跨基地共享供应商、库存安全冗余和成本分项等角度对优化前后的采购方案进行比较。")
    lines.append("具体而言，新增了供应商采购量 Pareto 图、基地-供应商采购热力图、库存水平折线图和成本分项柱状图，用于直观展示优化方案与人工经验方案在协同采购结构上的差异。")
    lines.append("同时，本文在结果分析中报告了启用供应商数量、Top-5 供应商采购份额、Herfindahl 指数、多基地共享供应商采购占比、平均安全库存冗余和单位铁元素成本等指标，从而更清晰地说明采购优化方案在供应商组合、跨基地协同和库存策略方面的具体特征。")
    lines.append("")

    lines.append("## 四、图表清单\n")
    for case in cases:
        lines.append(f"### {case}")
        for fig in figures[case]:
            lines.append(f"![{fig}]({fig})")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    configure_chinese_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cases = discover_cases(OUTPUT_DIR)
    if not cases:
        raise RuntimeError("No cases with both manual and ga_greedy_only results were found.")
    summaries = {}
    figures = {}
    for case in cases:
        tables = load_case_tables(case)
        summaries[case] = summarize_case(case, tables)
        figures[case] = make_case_figures(case, tables)
    REPORT_PATH.write_text(build_markdown(cases, summaries, figures), encoding="utf-8")
    print(f"Generated {sum(len(v) for v in figures.values())} figures in {FIG_DIR}")
    print(f"Generated report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
