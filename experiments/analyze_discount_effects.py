#!/usr/bin/env python3
"""Analyze whether procurement plans aggregate demand to obtain discounts."""

from __future__ import annotations

import os
import sys
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
DISCOUNT_DIR = OUTPUT_DIR / "discount_analysis"
FIG_DIR = DISCOUNT_DIR / "figures"
REPORT_PATH = DISCOUNT_DIR / "discount_effect_analysis.md"
PROCUREMENT_DIR = ROOT / "src" / "ga_hybrid"
sys.path.insert(0, str(PROCUREMENT_DIR))

from read_excel_file import read_xlsx_file_supplier  # noqa: E402


ALGORITHMS = ["manual", "ga_greedy_only"]
ALG_LABELS = {"manual": "人工经验算法", "ga_greedy_only": "GA-贪心算法"}


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


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.6f")


def discover_cases() -> list[str]:
    cases = []
    for path in OUTPUT_DIR.glob("*_cost_summary.csv"):
        case = path.name.replace("_cost_summary.csv", "")
        if all((OUTPUT_DIR / f"{case}_{alg}_purchase_detail.csv").exists() for alg in ALGORITHMS):
            cases.append(case)
    return sorted(cases)


def load_supplier_discount_data() -> dict[str, np.ndarray]:
    _m, J, _R, _J1, _J2, _Jsp, _z, _l, _u, r, Q, P = read_xlsx_file_supplier(ROOT / "data" / "raw_excel_optional" / "supplier_data.xlsx")
    return {"J": J, "r": r, "Q": Q, "P": P}


def build_discount_table(case: str, supplier_data: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    r = supplier_data["r"]
    Q = supplier_data["Q"]
    P = supplier_data["P"]
    for alg in ALGORITHMS:
        detail = read_csv(OUTPUT_DIR / f"{case}_{alg}_purchase_detail.csv")
        grouped = detail.groupby(["period", "supplier_id"], as_index=False).agg(
            qjt=("purchase_qty", "sum"),
            actual_purchase_cost=("purchase_cost", "sum"),
            active_base_count=("is_selected", "sum"),
            is_long_contract_supplier=("is_long_contract_supplier", "max"),
        )
        for _, item in grouped.iterrows():
            period = int(item["period"])
            supplier_id = int(item["supplier_id"])
            qjt = float(item["qjt"])
            threshold = float(Q[supplier_id])
            discount_rate = float(r[supplier_id])
            base_price = float(P[supplier_id][period])
            discount_per_ton = discount_rate * min(qjt, threshold)
            discounted_price = base_price - discount_per_ton
            undiscounted_cost = base_price * qjt
            discount_amount = discount_per_ton * qjt
            rows.append({
                "instance_name": case,
                "algorithm_name": alg,
                "algorithm_label": ALG_LABELS[alg],
                "period": period,
                "supplier_id": supplier_id,
                "qjt": qjt,
                "discount_threshold_Qj": threshold,
                "discount_rate_rj": discount_rate,
                "base_price_Pjt": base_price,
                "discounted_price_pjt": discounted_price,
                "discount_per_ton": discount_per_ton,
                "undiscounted_purchase_cost": undiscounted_cost,
                "discount_amount": discount_amount,
                "actual_purchase_cost": float(item["actual_purchase_cost"]),
                "threshold_ratio": qjt / threshold if abs(threshold) > 1e-12 else np.nan,
                "reached_discount_threshold": int(qjt >= threshold if threshold > 0 else False),
                "active_base_count": int(item["active_base_count"]),
                "is_long_contract_supplier": int(item["is_long_contract_supplier"]),
            })
    return pd.DataFrame(rows)


def summarize_discount(table: pd.DataFrame) -> pd.DataFrame:
    summary = table.groupby(["instance_name", "algorithm_name", "algorithm_label"], as_index=False).agg(
        total_purchase_qty=("qjt", "sum"),
        total_undiscounted_purchase_cost=("undiscounted_purchase_cost", "sum"),
        total_discount_amount=("discount_amount", "sum"),
        supplier_period_count=("qjt", lambda s: int((s > 1e-9).sum())),
        reached_threshold_count=("reached_discount_threshold", "sum"),
        avg_threshold_ratio_active=("threshold_ratio", lambda s: s[table.loc[s.index, "qjt"] > 1e-9].mean()),
    )
    summary["discount_saving_rate"] = summary["total_discount_amount"] / summary["total_undiscounted_purchase_cost"].replace(0, np.nan)
    return summary


def key_supplier_table(table: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    active = table[table["qjt"] > 1e-9].copy()
    active = active.sort_values(["instance_name", "algorithm_name", "discount_amount"], ascending=[True, True, False])
    return active.groupby(["instance_name", "algorithm_name"], group_keys=False).head(top_n).reset_index(drop=True)


def save_fig(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return str(path.relative_to(DISCOUNT_DIR))


def plot_total_discount(case: str, summary: pd.DataFrame) -> str:
    df = summary[summary["instance_name"] == case]
    x = np.arange(len(ALGORITHMS))
    vals = [df[df["algorithm_name"] == alg]["total_discount_amount"].iloc[0] / 1e8 for alg in ALGORITHMS]
    plt.figure(figsize=(6.8, 4.4))
    plt.bar(x, vals, color=["#7f8c8d", "#2f80ed"])
    plt.xticks(x, [ALG_LABELS[alg] for alg in ALGORITHMS])
    plt.ylabel("折扣金额（亿元）")
    plt.title(f"{case}：采购折扣总额对比")
    for i, v in enumerate(vals):
        plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    return save_fig(FIG_DIR / f"{case}_discount_total_comparison.png")


def plot_threshold_counts(case: str, summary: pd.DataFrame) -> str:
    df = summary[summary["instance_name"] == case]
    x = np.arange(len(ALGORITHMS))
    reached = [df[df["algorithm_name"] == alg]["reached_threshold_count"].iloc[0] for alg in ALGORITHMS]
    active = [df[df["algorithm_name"] == alg]["supplier_period_count"].iloc[0] for alg in ALGORITHMS]
    width = 0.36
    plt.figure(figsize=(7.2, 4.5))
    plt.bar(x - width / 2, active, width=width, label="有采购的供应商-周期数", color="#95a5a6")
    plt.bar(x + width / 2, reached, width=width, label="达到折扣门槛数量", color="#27ae60")
    plt.xticks(x, [ALG_LABELS[alg] for alg in ALGORITHMS])
    plt.ylabel("数量")
    plt.title(f"{case}：达到折扣门槛的供应商-周期数量")
    plt.legend()
    return save_fig(FIG_DIR / f"{case}_discount_threshold_counts.png")


def plot_key_supplier_discount(case: str, key_table: pd.DataFrame, top_n: int = 8) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, alg in zip(axes, ALGORITHMS):
        df = key_table[(key_table["instance_name"] == case) & (key_table["algorithm_name"] == alg)].copy()
        df = df.sort_values("discount_amount", ascending=True).tail(top_n)
        labels = [f"供应商{int(r.supplier_id)}-期{int(r.period)}" for r in df.itertuples()]
        ax.barh(labels, df["discount_amount"] / 1e8, color="#2f80ed" if alg == "ga_greedy_only" else "#7f8c8d")
        ax.set_title(ALG_LABELS[alg])
        ax.set_xlabel("折扣金额（亿元）")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle(f"{case}：关键供应商折扣贡献")
    path = FIG_DIR / f"{case}_key_supplier_discount.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return str(path.relative_to(DISCOUNT_DIR))


def plot_q_vs_threshold(case: str, key_table: pd.DataFrame, top_n: int = 8) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, alg in zip(axes, ALGORITHMS):
        df = key_table[(key_table["instance_name"] == case) & (key_table["algorithm_name"] == alg)].copy()
        df = df.sort_values("discount_amount", ascending=False).head(top_n)
        x = np.arange(len(df))
        width = 0.36
        labels = [f"{int(r.supplier_id)}-期{int(r.period)}" for r in df.itertuples()]
        ax.bar(x - width / 2, df["qjt"] / 10000, width=width, label="采购总量 qjt")
        ax.bar(x + width / 2, df["discount_threshold_Qj"] / 10000, width=width, label="折扣门槛 Qj")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(ALG_LABELS[alg])
        ax.set_ylabel("数量（万吨）")
        ax.legend(fontsize=8)
    fig.suptitle(f"{case}：关键供应商采购总量与折扣门槛")
    path = FIG_DIR / f"{case}_qjt_vs_discount_threshold.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return str(path.relative_to(DISCOUNT_DIR))


def make_figures(cases: list[str], summary: pd.DataFrame, key_table: pd.DataFrame) -> dict[str, list[str]]:
    figures = {}
    for case in cases:
        figures[case] = [
            plot_total_discount(case, summary),
            plot_threshold_counts(case, summary),
            plot_key_supplier_discount(case, key_table),
            plot_q_vs_threshold(case, key_table),
        ]
    return figures


def money(value: float) -> str:
    return f"{value / 1e8:.2f}亿元"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_report(cases: list[str], summary: pd.DataFrame, key_table: pd.DataFrame, figures: dict[str, list[str]]) -> str:
    lines = ["# 采购折扣效应分析报告\n"]
    lines.append("本报告基于 `outputs/` 中的采购明细结果，按供应商-周期汇总采购总量 `qjt`，并结合原始供应商数据中的折扣门槛 `Qj` 与折扣系数 `rj`，计算由需求整合带来的价格折扣金额。折扣金额按现有模型价格规则计算：`折扣金额 = rj * min(qjt, Qj) * qjt`。\n")

    lines.append("## 一、详细分析版本\n")
    for case in cases:
        lines.append(f"### {case}\n")
        case_summary = summary[summary["instance_name"] == case]
        manual = case_summary[case_summary["algorithm_name"] == "manual"].iloc[0]
        ga = case_summary[case_summary["algorithm_name"] == "ga_greedy_only"].iloc[0]
        lines.append(
            f"- 折扣总额：人工经验算法实现折扣 {money(manual['total_discount_amount'])}，"
            f"GA-贪心算法实现折扣 {money(ga['total_discount_amount'])}。"
        )
        lines.append(
            f"- 达到门槛情况：人工经验算法有 {manual['supplier_period_count']:.0f} 个供应商-周期发生采购，其中 "
            f"{manual['reached_threshold_count']:.0f} 个达到折扣门槛；GA-贪心算法有 "
            f"{ga['supplier_period_count']:.0f} 个供应商-周期发生采购，其中 {ga['reached_threshold_count']:.0f} 个达到折扣门槛。"
        )
        lines.append(
            f"- 折扣节约率：人工经验算法折扣金额占未折扣采购成本的 {pct(manual['discount_saving_rate'])}，"
            f"GA-贪心算法为 {pct(ga['discount_saving_rate'])}。"
        )
        lines.append("- 关键供应商折扣贡献如下：")
        kt = key_table[key_table["instance_name"] == case].sort_values("discount_amount", ascending=False).head(12)
        lines.append("| 算法 | 周期 | 供应商 | qjt | Qj | 是否达门槛 | 折扣金额 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for item in kt.itertuples():
            lines.append(
                f"| {ALG_LABELS[item.algorithm_name]} | {int(item.period)} | {int(item.supplier_id)} | "
                f"{item.qjt:.2f} | {item.discount_threshold_Qj:.2f} | {int(item.reached_discount_threshold)} | "
                f"{item.discount_amount:.2f} |"
            )
        lines.append("\n图形输出：")
        for fig in figures[case]:
            lines.append(f"- `{fig}`")
        lines.append("")

    lines.append("## 二、用于论文对应章节的内容版本\n")
    lines.append("为验证协同采购是否通过整合多基地需求获得供应商价格折扣，本文进一步从供应商-周期层面统计各方案的采购总量 `qjt`、折扣门槛 `Qj` 以及由折扣函数产生的节约金额。")
    for case in cases:
        case_summary = summary[summary["instance_name"] == case]
        manual = case_summary[case_summary["algorithm_name"] == "manual"].iloc[0]
        ga = case_summary[case_summary["algorithm_name"] == "ga_greedy_only"].iloc[0]
        lines.append(
            f"在 {case} 算例中，人工经验算法和 GA-贪心算法分别实现折扣金额 {money(manual['total_discount_amount'])} "
            f"和 {money(ga['total_discount_amount'])}；达到折扣门槛的供应商-周期数量分别为 "
            f"{manual['reached_threshold_count']:.0f} 和 {ga['reached_threshold_count']:.0f}。"
            f"这说明采购方案确实存在通过集中采购量跨越折扣门槛、降低实际采购单价的机制。"
        )
    lines.append("本文据此补充关键供应商折扣表和采购量-折扣门槛对比图，以更直观地展示协同采购中的需求整合和折扣获取过程。")
    lines.append("")

    lines.append("## 三、审稿人回复版本\n")
    lines.append("感谢审稿人的建议。根据该意见，本文新增了供应商折扣效应分析。具体而言，本文按供应商-周期统计采购总量 `qjt`，并与各供应商折扣门槛 `Qj` 进行比较，同时计算由价格折扣函数产生的折扣金额。")
    lines.append("新增结果包括关键供应商折扣贡献表、采购总量与折扣门槛对比图、总折扣金额对比图以及达到折扣门槛的供应商-周期数量对比图。这些补充结果能够直接说明采购方案通过整合多基地需求形成较大的供应商采购量，从而触发或接近价格折扣门槛，并获得相应采购成本节约。")
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
    DISCOUNT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cases = discover_cases()
    if not cases:
        raise RuntimeError("No purchase detail files were found for both manual and ga_greedy_only.")
    supplier_data = load_supplier_discount_data()
    all_tables = []
    for case in cases:
        table = build_discount_table(case, supplier_data)
        write_csv(table, DISCOUNT_DIR / f"{case}_supplier_period_discount_detail.csv")
        all_tables.append(table)
    detail = pd.concat(all_tables, ignore_index=True)
    summary = summarize_discount(detail)
    key_table = key_supplier_table(detail, top_n=10)
    write_csv(detail, DISCOUNT_DIR / "supplier_period_discount_detail_all.csv")
    write_csv(summary, DISCOUNT_DIR / "discount_summary_by_case_algorithm.csv")
    write_csv(key_table, DISCOUNT_DIR / "key_supplier_discount_table.csv")
    figures = make_figures(cases, summary, key_table)
    REPORT_PATH.write_text(build_report(cases, summary, key_table, figures), encoding="utf-8")
    print(f"Generated discount tables in {DISCOUNT_DIR}")
    print(f"Generated {sum(len(v) for v in figures.values())} figures in {FIG_DIR}")
    print(f"Generated report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
