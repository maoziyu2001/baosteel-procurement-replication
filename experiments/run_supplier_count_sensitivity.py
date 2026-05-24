"""Run supplier-count sensitivity experiments based on GA-greedy-only.py.

This script does not modify the original algorithm file. It loads the GA
implementation dynamically, fixes bases/demand/safety-stock/algorithm
parameters, and changes only the candidate supplier set size.

Outputs are written to outputs/supplier_count_sensitivity/ by default:
- raw_results.csv: seed-level results.
- summary.csv: supplier-count-level summary statistics.
- convergence_history.csv: filled best-cost trajectory for each run.
- convergence_mean.csv: mean best cost per iteration and supplier count.
- supplier_discount_summary.csv: final supplier structure and discount metrics.
- figures/*.png: paper-ready plots with Chinese labels.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCUREMENT_DIR = ROOT / "src" / "ga_hybrid"
DATA_DIR = ROOT / "data" / "raw_excel_optional"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "supplier_count_sensitivity"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

sys.path.insert(0, str(PROCUREMENT_DIR))

from read_excel_file import (  # noqa: E402
    read_xlsx_file_base,
    read_xlsx_file_supplier,
    read_xlsx_file_transport,
)


def configure_chinese_font() -> None:
    """Configure matplotlib to render Chinese labels when a local font exists."""
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


def load_ga_module() -> Any:
    """Load GA-greedy-only.py, whose filename cannot be imported normally."""
    path = PROCUREMENT_DIR / "GA-greedy-only.py"
    spec = importlib.util.spec_from_file_location("ga_greedy_only_sensitivity", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_manual_module() -> Any:
    """Load manual.py for the enterprise heuristic baseline."""
    path = PROCUREMENT_DIR / "manual.py"
    spec = importlib.util.spec_from_file_location("manual_sensitivity", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_instance_data() -> dict[str, Any]:
    base_file = DATA_DIR / "base_data.xlsx"
    supplier_file = DATA_DIR / "supplier_data.xlsx"
    transport_file = DATA_DIR / "transport_data.xlsx"

    n, I, beta, W, w0, e0, H, f, D, S, alpha, L = read_xlsx_file_base(base_file)
    m, J, R, J1, J2, Jsp, z, l, u, r, Q, P = read_xlsx_file_supplier(supplier_file)
    G, cjg, cgi, cij = read_xlsx_file_transport(transport_file)
    return {
        "n": n,
        "I_all": list(I),
        "beta": beta,
        "W": W,
        "w0": w0,
        "e0": e0,
        "H": H,
        "f": f,
        "D": D,
        "S": S,
        "alpha": alpha,
        "L": L,
        "m": m,
        "J_all": list(J),
        "R": R,
        "J1": J1,
        "J2": J2,
        "Jsp": Jsp,
        "z": z,
        "l": l,
        "u": u,
        "r": r,
        "Q": Q,
        "P": P,
        "G": G,
        "cjg": cjg,
        "cgi": cgi,
        "cij": cij,
    }


def configure_algorithm_globals(module: Any, data: dict[str, Any], bases: list[int], suppliers: list[int]) -> None:
    """Set module-level variables expected by the legacy algorithm files."""
    module.n = data["n"]
    module.I = bases
    module.beta = data["beta"]
    module.W = data["W"]
    module.w0 = data["w0"]
    module.e0 = data["e0"]
    module.H = data["H"]
    module.f = data["f"]
    module.D = data["D"]
    module.S = data["S"]
    module.alpha = data["alpha"]
    module.L = data["L"]
    module.m = data["m"]
    module.J = suppliers
    module.R = data["R"]
    module.J1 = data["J1"]
    module.J2 = data["J2"]
    module.Jsp = data["Jsp"]
    module.z = data["z"]
    module.l = data["l"]
    module.u = data["u"]
    module.r = data["r"]
    module.Q = data["Q"]
    module.P = data["P"]
    module.G = data["G"]
    module.cjg = data["cjg"]
    module.cgi = data["cgi"]
    module.cij = data["cij"]
    module.t = 0
    module.K = 1
    module.T = [0]
    module.tau_list = [0, 1]


def compute_lower_bound(
    data: dict[str, Any],
    bases: list[int],
    suppliers: list[int],
    time_limit: float,
) -> float | None:
    """Compute a lower bound with lowerbound_gurobi.py; fail soft."""
    try:
        from lowerbound_gurobi import solve_procument_plan

        result = solve_procument_plan(
            data["n"],
            bases,
            data["beta"],
            data["W"],
            data["w0"],
            data["e0"],
            data["H"],
            data["f"],
            data["D"],
            data["S"],
            data["alpha"],
            data["L"],
            data["m"],
            suppliers,
            data["R"],
            data["J1"],
            data["J2"],
            data["Jsp"],
            data["z"],
            data["l"],
            data["u"],
            data["r"],
            data["Q"],
            data["P"],
            data["G"],
            data["cjg"],
            data["cgi"],
            data["cij"],
            0,
            1,
            [0, 1],
            [0],
            time_limit=time_limit,
            outputflag=0,
            return_results=True,
        )
        lower_bound = result.get("lower_bound")
        return float(lower_bound) if lower_bound is not None else None
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] lower bound failed for J={len(suppliers)}: {exc}")
        return None


def parse_history_from_stdout(stdout: str, max_gen: int, returned_history: list[tuple]) -> pd.DataFrame:
    """Build a per-generation best-cost trajectory from GA output and history.

    GA-greedy-only.py records history only when the population is updated.
    For convergence plots, generations with no update should carry forward the
    previous best cost; this function reconstructs that stepwise trajectory.
    """
    records: list[dict[str, float]] = []
    match0 = re.search(r"Gen\s+0:\s+Best Cost=([0-9.]+)", stdout)
    if match0:
        records.append({"generation": 0, "best_cost": float(match0.group(1))})
    for item in returned_history:
        if len(item) >= 2:
            records.append({"generation": int(item[0]), "best_cost": float(item[1])})

    if not records:
        return pd.DataFrame(
            {"generation": list(range(max_gen)), "best_cost": [np.nan] * max_gen}
        )

    sparse = pd.DataFrame(records).drop_duplicates("generation", keep="last")
    dense = pd.DataFrame({"generation": list(range(max_gen))})
    dense = dense.merge(sparse, on="generation", how="left")
    dense["best_cost"] = dense["best_cost"].ffill()
    dense["best_cost"] = dense["best_cost"].bfill()
    return dense


def analyze_final_solution(best_solution: Any, data: dict[str, Any], suppliers: list[int]) -> dict[str, float]:
    """Compute final supplier structure and price-discount metrics."""
    decode = np.asarray(best_solution.decode)
    if decode.ndim != 4:
        return {}

    eps = 1e-6
    selected_supplier_count = 0
    total_purchase_qty = 0.0
    total_discount_amount = 0.0
    total_undiscounted_purchase_cost = 0.0
    reached_threshold_count = 0
    supplier_period_active_count = 0
    top_shares: list[float] = []

    supplier_totals = []
    for j in suppliers:
        q_total_j = 0.0
        for tau in [0, 1]:
            qjt = float(sum(decode[i, j, 0, tau] for i in range(decode.shape[0])))
            q_total_j += qjt
            if qjt > eps:
                supplier_period_active_count += 1
                if qjt >= float(data["Q"][j]) - eps:
                    reached_threshold_count += 1
                discount_per_ton = float(data["r"][j]) * min(qjt, float(data["Q"][j]))
                total_discount_amount += discount_per_ton * qjt
                total_undiscounted_purchase_cost += float(data["P"][j][tau]) * qjt
        if q_total_j > eps:
            selected_supplier_count += 1
        supplier_totals.append(q_total_j)
        total_purchase_qty += q_total_j

    if total_purchase_qty > eps:
        sorted_totals = sorted(supplier_totals, reverse=True)
        top_shares = [
            sum(sorted_totals[:k]) / total_purchase_qty for k in (1, 3, 5)
        ]
    else:
        top_shares = [math.nan, math.nan, math.nan]

    return {
        "selected_supplier_count": selected_supplier_count,
        "total_purchase_qty": total_purchase_qty,
        "supplier_period_active_count": supplier_period_active_count,
        "reached_discount_threshold_count": reached_threshold_count,
        "total_discount_amount": total_discount_amount,
        "discount_saving_rate": total_discount_amount / total_undiscounted_purchase_cost
        if total_undiscounted_purchase_cost > eps
        else math.nan,
        "top1_supplier_share": top_shares[0],
        "top3_supplier_share": top_shares[1],
        "top5_supplier_share": top_shares[2],
    }


def build_discount_detail(
    best_solution: Any,
    data: dict[str, Any],
    supplier_count: int,
    seed: int,
    args: argparse.Namespace,
    algorithm_name: str,
) -> list[dict[str, Any]]:
    """Build supplier-period discount details for the final solution."""
    decode = np.asarray(best_solution.decode)
    if decode.ndim != 4:
        return []

    rows: list[dict[str, Any]] = []
    eps = 1e-6
    for j in range(supplier_count):
        for tau in [0, 1]:
            qjt = float(sum(decode[i, j, 0, tau] for i in range(decode.shape[0])))
            threshold = float(data["Q"][j])
            discount_rate = float(data["r"][j])
            base_price = float(data["P"][j][tau])
            discount_per_ton = discount_rate * min(qjt, threshold)
            rows.append(
                {
                    "supplier_count": supplier_count,
                    "case_id": f"I{args.base_count}_J{supplier_count}",
                    "algorithm_name": algorithm_name,
                    "random_seed": seed,
                    "period": tau,
                    "supplier_id": j,
                    "qjt": qjt,
                    "discount_threshold_Qj": threshold,
                    "discount_rate_rj": discount_rate,
                    "base_price_Pjt": base_price,
                    "discounted_price_pjt": base_price - discount_per_ton,
                    "discount_per_ton": discount_per_ton,
                    "discount_amount": discount_per_ton * qjt,
                    "threshold_ratio": qjt / threshold if threshold > eps else math.nan,
                    "reached_discount_threshold": int(qjt >= threshold - eps and qjt > eps),
                    "is_active": int(qjt > eps),
                }
            )
    return rows


def run_one(
    module: Any,
    data: dict[str, Any],
    supplier_count: int,
    seed: int,
    args: argparse.Namespace,
    lower_bound: float | None,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    bases = list(range(args.base_count))
    suppliers = list(range(supplier_count))
    configure_algorithm_globals(module, data, bases, suppliers)
    random.seed(seed)
    np.random.seed(seed)

    ga = module.GA(
        pop_size=args.population_size,
        max_gen=args.max_iter,
        mut_prob=args.mutation_prob,
        max_time=args.time_limit,
        RA_max=args.ra_max,
        RA_min=args.ra_min,
    )

    start = time.time()
    stdout_buffer = io.StringIO()
    status = "success"
    error_message = ""
    best_solution = None
    history: list[tuple] = []
    try:
        with contextlib.redirect_stdout(stdout_buffer):
            best_solution, history = ga.run()
        runtime = time.time() - start
    except Exception as exc:  # noqa: BLE001
        runtime = time.time() - start
        status = "error"
        error_message = str(exc)[:300]

    stdout = stdout_buffer.getvalue()
    history_df = parse_history_from_stdout(stdout, args.max_iter, history)
    history_df["supplier_count"] = supplier_count
    history_df["algorithm_name"] = "ga_greedy_only"
    history_df["random_seed"] = seed

    if best_solution is not None:
        final_best_cost = float(best_solution.cost)
        feasible = bool(best_solution.feasible)
        final_metrics = analyze_final_solution(best_solution, data, suppliers)
        discount_detail = build_discount_detail(
            best_solution, data, supplier_count, seed, args, "ga_greedy_only"
        )
    else:
        final_best_cost = math.nan
        feasible = False
        final_metrics = {}
        discount_detail = []

    if pd.notna(final_best_cost):
        final_cost_rows = history_df.loc[np.isclose(history_df["best_cost"], final_best_cost)]
        iter_to_final = int(final_cost_rows["generation"].iloc[0]) if not final_cost_rows.empty else math.nan
    else:
        iter_to_final = math.nan

    gap = (
        (final_best_cost - lower_bound) / lower_bound * 100.0
        if lower_bound is not None and lower_bound > 0 and pd.notna(final_best_cost)
        else math.nan
    )

    raw_row = {
        "supplier_count": supplier_count,
        "case_id": f"I{args.base_count}_J{supplier_count}",
        "algorithm_name": "ga_greedy_only",
        "random_seed": seed,
        "status": status,
        "error_message": error_message,
        "best_cost": final_best_cost,
        "lower_bound": lower_bound if lower_bound is not None else math.nan,
        "gap_to_lower_bound_pct": gap,
        "runtime_sec": runtime,
        "iter_to_final_solution": iter_to_final,
        "max_iter": args.max_iter,
        "population_size": args.population_size,
        "final_feasible": feasible,
    }
    raw_row.update(final_metrics)

    supplier_row = {
        "supplier_count": supplier_count,
        "case_id": f"I{args.base_count}_J{supplier_count}",
        "algorithm_name": "ga_greedy_only",
        "random_seed": seed,
        **final_metrics,
    }
    return raw_row, history_df, supplier_row, discount_detail


def manual_procurement_to_solution(procurement: np.ndarray, cost: float, feasible: bool, runtime: float) -> Any:
    """Adapt manual.py's 3-D procurement array to GA-style 4-D decode output."""
    n_bases, n_suppliers, n_periods = procurement.shape
    decode = np.zeros((n_bases, n_suppliers, 1, n_periods), dtype=float)
    for tau in range(n_periods):
        decode[:, :, 0, tau] = procurement[:, :, tau]
    return SimpleNamespace(cost=cost, feasible=feasible, decode=decode, runtime_seconds=runtime)


def run_manual_one(
    module: Any,
    data: dict[str, Any],
    supplier_count: int,
    args: argparse.Namespace,
    lower_bound: float | None,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    """Run the manual heuristic once for a supplier-count instance."""
    bases = list(range(args.base_count))
    suppliers = list(range(supplier_count))
    configure_algorithm_globals(module, data, bases, suppliers)

    status = "success"
    error_message = ""
    best_solution = None
    start = time.time()
    try:
        algorithm = module.HeuristicProcurementAlgorithm(
            bases,
            suppliers,
            [0, 1],
            data["D"],
            data["z"],
            data["H"],
            data["l"],
            data["u"],
            data["P"],
            data["r"],
            data["R"],
            data["Q"],
        )
        with contextlib.redirect_stdout(io.StringIO()):
            procurement, feasible, total_cost = algorithm.execute()
        runtime = time.time() - start
        best_solution = manual_procurement_to_solution(procurement, float(total_cost), bool(feasible), runtime)
    except Exception as exc:  # noqa: BLE001
        runtime = time.time() - start
        status = "error"
        error_message = str(exc)[:300]

    if best_solution is not None:
        final_best_cost = float(best_solution.cost)
        feasible = bool(best_solution.feasible)
        final_metrics = analyze_final_solution(best_solution, data, suppliers)
        discount_detail = build_discount_detail(
            best_solution, data, supplier_count, -1, args, "manual"
        )
    else:
        final_best_cost = math.nan
        feasible = False
        final_metrics = {}
        discount_detail = []

    gap = (
        (final_best_cost - lower_bound) / lower_bound * 100.0
        if lower_bound is not None and lower_bound > 0 and pd.notna(final_best_cost)
        else math.nan
    )

    history_df = pd.DataFrame(
        {
            "generation": list(range(args.max_iter)),
            "best_cost": [final_best_cost] * args.max_iter,
            "supplier_count": supplier_count,
            "algorithm_name": "manual",
            "random_seed": -1,
        }
    )

    raw_row = {
        "supplier_count": supplier_count,
        "case_id": f"I{args.base_count}_J{supplier_count}",
        "algorithm_name": "manual",
        "random_seed": -1,
        "status": status,
        "error_message": error_message,
        "best_cost": final_best_cost,
        "lower_bound": lower_bound if lower_bound is not None else math.nan,
        "gap_to_lower_bound_pct": gap,
        "runtime_sec": runtime,
        "iter_to_final_solution": 0,
        "max_iter": 0,
        "population_size": math.nan,
        "final_feasible": feasible,
    }
    raw_row.update(final_metrics)

    supplier_row = {
        "supplier_count": supplier_count,
        "case_id": f"I{args.base_count}_J{supplier_count}",
        "algorithm_name": "manual",
        "random_seed": -1,
        **final_metrics,
    }
    return raw_row, history_df, supplier_row, discount_detail


def summarize_results(raw: pd.DataFrame) -> pd.DataFrame:
    grouped = raw.groupby(["supplier_count", "algorithm_name"], as_index=False).agg(
        best_cost_mean=("best_cost", "mean"),
        best_cost_std=("best_cost", "std"),
        best_cost_min=("best_cost", "min"),
        best_cost_max=("best_cost", "max"),
        gap_mean=("gap_to_lower_bound_pct", "mean"),
        gap_std=("gap_to_lower_bound_pct", "std"),
        runtime_mean=("runtime_sec", "mean"),
        runtime_std=("runtime_sec", "std"),
        iter_to_final_mean=("iter_to_final_solution", "mean"),
        feasible_rate=("final_feasible", "mean"),
        selected_supplier_count_mean=("selected_supplier_count", "mean"),
        reached_discount_threshold_count_mean=("reached_discount_threshold_count", "mean"),
        total_discount_amount_mean=("total_discount_amount", "mean"),
        discount_saving_rate_mean=("discount_saving_rate", "mean"),
    )
    return grouped


def build_convergence_mean(history: pd.DataFrame) -> pd.DataFrame:
    return (
        history.groupby(["supplier_count", "algorithm_name", "generation"], as_index=False)
        .agg(best_cost_mean=("best_cost", "mean"), best_cost_std=("best_cost", "std"))
        .sort_values(["supplier_count", "algorithm_name", "generation"])
    )


def build_comparison_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Create a method-comparison table with GA, manual, and lower bound."""
    rows: list[dict[str, Any]] = []
    for supplier_count, group in raw.groupby("supplier_count"):
        lower_bound = group["lower_bound"].dropna()
        lb_value = float(lower_bound.iloc[0]) if not lower_bound.empty else math.nan
        for algorithm_name, alg_group in group.groupby("algorithm_name"):
            rows.append(
                {
                    "supplier_count": supplier_count,
                    "method": algorithm_name,
                    "cost_mean": alg_group["best_cost"].mean(),
                    "cost_std": alg_group["best_cost"].std(),
                    "gap_to_lower_bound_pct_mean": alg_group["gap_to_lower_bound_pct"].mean(),
                    "runtime_sec_mean": alg_group["runtime_sec"].mean(),
                    "feasible_rate": alg_group["final_feasible"].mean(),
                    "run_count": len(alg_group),
                }
            )
        rows.append(
            {
                "supplier_count": supplier_count,
                "method": "lower_bound",
                "cost_mean": lb_value,
                "cost_std": math.nan,
                "gap_to_lower_bound_pct_mean": 0.0 if pd.notna(lb_value) else math.nan,
                "runtime_sec_mean": math.nan,
                "feasible_rate": math.nan,
                "run_count": 1 if pd.notna(lb_value) else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(["supplier_count", "method"])


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.6f")


def plot_total_cost(comparison: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    counts = sorted(comparison["supplier_count"].unique())
    x = np.arange(len(counts))
    width = 0.26
    styles = {
        "ga_greedy_only": ("GA-greedy", "#2f80ed", -width),
        "manual": ("manual", "#7f8c8d", 0.0),
        "lower_bound": ("下界", "#27ae60", width),
    }
    for method, (label, color, offset) in styles.items():
        df = comparison[comparison["method"] == method].set_index("supplier_count")
        y = [df.loc[c, "cost_mean"] / 1e9 if c in df.index else math.nan for c in counts]
        yerr = [
            df.loc[c, "cost_std"] / 1e9 if c in df.index and pd.notna(df.loc[c, "cost_std"]) else 0
            for c in counts
        ]
        ax.bar(x + offset, y, width=width, yerr=yerr, capsize=3, color=color, alpha=0.86, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in counts])
    ax.set_xlabel("候选供应商数量")
    ax.set_ylabel("成本（十亿元）")
    ax.set_title("不同供应商数量下 GA-greedy、manual 与下界对比")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "supplier_count_total_cost_bar.png", dpi=300)
    plt.close(fig)


def plot_gap_runtime(summary: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8.8, 5.2))
    ga_summary = summary[summary["algorithm_name"] == "ga_greedy_only"].copy()
    manual_summary = summary[summary["algorithm_name"] == "manual"].copy()
    x = ga_summary["supplier_count"]
    ax1.set_xlabel("候选供应商数量")
    if ga_summary["gap_mean"].notna().any():
        ax1.plot(x, ga_summary["gap_mean"], marker="o", color="#2f80ed", label="GA-greedy Gap")
        if not manual_summary.empty:
            ax1.plot(
                manual_summary["supplier_count"],
                manual_summary["gap_mean"],
                marker="^",
                color="#7f8c8d",
                label="manual Gap",
            )
        ax1.set_ylabel("Gap to LB（%）", color="#2f80ed")
        ax1.tick_params(axis="y", labelcolor="#2f80ed")
    else:
        ax1.text(
            0.02,
            0.92,
            "未计算下界，Gap 留空",
            transform=ax1.transAxes,
            color="#2f80ed",
        )
        ax1.set_ylabel("Gap to LB（%）", color="#2f80ed")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, ga_summary["runtime_mean"], marker="s", color="#d35400", label="GA-greedy运行时间")
    if not manual_summary.empty:
        ax2.plot(
            manual_summary["supplier_count"],
            manual_summary["runtime_mean"],
            marker="x",
            color="#8e44ad",
            label="manual运行时间",
        )
    ax2.set_ylabel("平均运行时间（秒）", color="#d35400")
    ax2.tick_params(axis="y", labelcolor="#d35400")
    ax1.set_title("供应商数量对求解质量与运行时间的影响")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / "supplier_count_gap_runtime_dual_axis.png", dpi=300)
    plt.close(fig)


def plot_convergence(convergence_mean: pd.DataFrame, figures_dir: Path) -> None:
    line_styles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    gray_levels = ["#111111", "#2b2b2b", "#454545", "#5f5f5f", "#777777", "#8c8c8c", "#a0a0a0"]

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ga_only = convergence_mean[convergence_mean["algorithm_name"] == "ga_greedy_only"]
    for idx, (supplier_count, group) in enumerate(ga_only.groupby("supplier_count")):
        group = group.sort_values("generation")
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
    ax.set_ylabel("平均最优总成本（十亿元）")
    ax.set_title("不同供应商数量下的收敛曲线", pad=34)
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
    fig.savefig(figures_dir / "supplier_count_convergence_curve.png", dpi=300)
    fig.savefig(figures_dir / "supplier_count_convergence_curve.pdf", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supplier-count sensitivity analysis based on GA-greedy-only.py."
    )
    parser.add_argument("--supplier-counts", nargs="+", type=int, default=[20, 25, 30, 35, 40, 45, 50])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--base-count", type=int, default=4)
    parser.add_argument("--population-size", type=int, default=200)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--mutation-prob", type=float, default=0.2)
    parser.add_argument("--time-limit", type=float, default=3600)
    parser.add_argument("--ra-max", type=float, default=0.8)
    parser.add_argument("--ra-min", type=float, default=0.5)
    parser.add_argument("--lower-bound-time-limit", type=float, default=300)
    parser.add_argument(
        "--skip-lower-bound",
        action="store_true",
        help="Skip Gurobi lower-bound computation. Gap columns will be NaN.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_chinese_font()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = load_instance_data()
    ga_module = load_ga_module()
    manual_module = load_manual_module()

    raw_rows: list[dict[str, Any]] = []
    supplier_rows: list[dict[str, Any]] = []
    discount_detail_rows: list[dict[str, Any]] = []
    histories: list[pd.DataFrame] = []

    lower_bounds: dict[int, float | None] = {}
    for supplier_count in args.supplier_counts:
        suppliers = list(range(supplier_count))
        bases = list(range(args.base_count))
        if args.skip_lower_bound:
            lower_bounds[supplier_count] = None
        else:
            lower_bounds[supplier_count] = compute_lower_bound(
                data, bases, suppliers, args.lower_bound_time_limit
            )

        print(f"Running manual supplier_count={supplier_count}")
        manual_row, manual_history, manual_supplier_row, manual_discount_detail = run_manual_one(
            manual_module, data, supplier_count, args, lower_bounds[supplier_count]
        )
        raw_rows.append(manual_row)
        histories.append(manual_history)
        supplier_rows.append(manual_supplier_row)
        discount_detail_rows.extend(manual_discount_detail)

        for seed in args.seeds:
            print(f"Running supplier_count={supplier_count}, seed={seed}")
            row, history, supplier_row, discount_detail = run_one(
                ga_module, data, supplier_count, seed, args, lower_bounds[supplier_count]
            )
            raw_rows.append(row)
            histories.append(history)
            supplier_rows.append(supplier_row)
            discount_detail_rows.extend(discount_detail)

    raw = pd.DataFrame(raw_rows)
    history = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    supplier_summary = pd.DataFrame(supplier_rows)
    discount_detail_df = pd.DataFrame(discount_detail_rows)
    summary = summarize_results(raw)
    comparison = build_comparison_summary(raw)
    convergence_mean = build_convergence_mean(history) if not history.empty else pd.DataFrame()

    write_csv(raw, args.output_dir / "raw_results.csv")
    write_csv(summary, args.output_dir / "summary.csv")
    write_csv(comparison, args.output_dir / "comparison_summary.csv")
    write_csv(history, args.output_dir / "convergence_history.csv")
    write_csv(convergence_mean, args.output_dir / "convergence_mean.csv")
    write_csv(supplier_summary, args.output_dir / "supplier_discount_summary.csv")
    write_csv(discount_detail_df, args.output_dir / "supplier_period_discount_detail.csv")

    plot_total_cost(comparison, figures_dir)
    plot_gap_runtime(summary, figures_dir)
    if not convergence_mean.empty:
        plot_convergence(convergence_mean, figures_dir)

    print(f"Supplier-count sensitivity analysis finished: {args.output_dir}")


if __name__ == "__main__":
    main()
