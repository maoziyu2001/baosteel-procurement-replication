#!/usr/bin/env python3
"""Generate Chinese runtime distribution bar charts for GA-hybrid runs.

This script instruments ``procurement-v3/GA-greedy-solver.py`` without editing
the original algorithm file. It runs two representative scales by default:

* 2 bases x 20 suppliers
* 4 bases x 40 suppliers

The measured runtime distribution covers the 100-generation main GA loop after
initial population construction. Initialization time is still exported to the
CSV for reproducibility, but it is not included in the plotted four categories.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCUREMENT_DIR = ROOT / "src" / "ga_hybrid"
DATA_DIR = ROOT / "data" / "raw_excel_optional"
OUTPUT_DIR = ROOT / "outputs" / "runtime_distribution"
GA_FILE = PROCUREMENT_DIR / "GA-greedy-solver.py"

if str(PROCUREMENT_DIR) not in sys.path:
    sys.path.insert(0, str(PROCUREMENT_DIR))


@dataclass
class RuntimeStats:
    greedy_decode_sec: float = 0.0
    solver_decode_sec: float = 0.0
    local_search_sec: float = 0.0
    loop_total_sec: float = 0.0
    initialization_sec: float = 0.0
    greedy_decode_calls: int = 0
    solver_decode_calls: int = 0
    local_search_calls: int = 0
    best_cost: float = np.nan
    feasible_count: int = 0


def configure_chinese_font() -> None:
    """Configure matplotlib for Chinese labels on macOS/Windows/Linux."""
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
    plt.rcParams["axes.unicode_minus"] = False


def load_ga_module() -> Any:
    spec = importlib.util.spec_from_file_location("ga_greedy_solver_runtime", GA_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load GA module from {GA_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def set_module_instance(module: Any, data: dict[str, Any], base_count: int, supplier_count: int) -> None:
    for key, value in data.items():
        setattr(module, key, value)
    module.I = list(range(base_count))
    module.J = list(range(supplier_count))
    module.t = 0
    module.K = 1
    module.T = [0]
    module.tau_list = [0, 1]


def load_data(module: Any) -> dict[str, Any]:
    base_file = DATA_DIR / "base_data.xlsx"
    supplier_file = DATA_DIR / "supplier_data.xlsx"
    transport_file = DATA_DIR / "transport_data.xlsx"
    n, I, beta, W, w0, e0, H, f, D, S, alpha, L = module.read_xlsx_file_base(base_file)
    m, J, R, J1, J2, Jsp, z, l, u, r, Q, P = module.read_xlsx_file_supplier(supplier_file)
    G, cjg, cgi, cij = module.read_xlsx_file_transport(transport_file)
    return {
        "n": n,
        "I": I,
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
        "J": J,
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


def make_timed_ga_class(module: Any, stats: RuntimeStats, solver_time_limit: float, max_retry_count: int) -> type:
    class TimedGA(module.GA):
        def evaluate(self, individual):  # noqa: ANN001 - original class is untyped
            individual.yijt = self.repair_long_contract_selection(individual.yijt)
            started = time.perf_counter()
            try:
                if self.gen < self.max_gen // 2:
                    with contextlib.redirect_stdout(io.StringIO()):
                        individual.feasible, individual.cost, individual.decode, individual.lowerbound = (
                            module.solve_procument_plan_greedy(
                                module.I,
                                module.beta,
                                module.W,
                                module.w0,
                                module.e0,
                                module.H,
                                module.f,
                                module.D,
                                module.S,
                                module.alpha,
                                module.L,
                                module.J,
                                module.R,
                                module.Jsp,
                                module.z,
                                module.l,
                                module.u,
                                module.r,
                                module.Q,
                                module.P,
                                module.cij,
                                module.t,
                                module.K,
                                module.tau_list,
                                module.T,
                                individual.yijt,
                            )
                        )
                    stats.greedy_decode_sec += time.perf_counter() - started
                    stats.greedy_decode_calls += 1
                else:
                    with contextlib.redirect_stdout(io.StringIO()):
                        individual.feasible, individual.cost, individual.decode, individual.lowerbound = (
                            module.solve_procument_plan(
                                module.n,
                                module.I,
                                module.beta,
                                module.W,
                                module.w0,
                                module.e0,
                                module.H,
                                module.f,
                                module.D,
                                module.S,
                                module.alpha,
                                module.L,
                                module.m,
                                module.J,
                                module.R,
                                module.J1,
                                module.J2,
                                module.Jsp,
                                module.z,
                                module.l,
                                module.u,
                                module.r,
                                module.Q,
                                module.P,
                                module.G,
                                module.cjg,
                                module.cgi,
                                module.cij,
                                module.t,
                                module.K,
                                module.tau_list,
                                module.T,
                                individual.yijt,
                                time_limit=solver_time_limit,
                                outputflag=0,
                            )
                        )
                    stats.solver_decode_sec += time.perf_counter() - started
                    stats.solver_decode_calls += 1
            except Exception:
                if self.gen < self.max_gen // 2:
                    stats.greedy_decode_sec += time.perf_counter() - started
                    stats.greedy_decode_calls += 1
                else:
                    stats.solver_decode_sec += time.perf_counter() - started
                    stats.solver_decode_calls += 1
                individual.feasible = False
                individual.cost = 1e12
                individual.lowerbound = getattr(individual, "lowerbound", 1e10)

        def run(self):
            init_started = time.perf_counter()
            population = self.initialize_population()
            stats.initialization_sec = time.perf_counter() - init_started

            # Plot only the 100-generation main loop; reset decoder counters
            # because initialization can dominate and obscure the dynamics.
            stats.greedy_decode_sec = 0.0
            stats.solver_decode_sec = 0.0
            stats.local_search_sec = 0.0
            stats.greedy_decode_calls = 0
            stats.solver_decode_calls = 0
            stats.local_search_calls = 0

            loop_started = time.perf_counter()
            best_individual = min(population, key=lambda x: x.cost)
            history = []
            for gen in range(1, self.max_gen + 1):
                self.gen = gen
                self.update_fitness(population)
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                pc = self.adapative_pc(population, parent1, parent2)
                child1, child2 = self.crossover(parent1, parent2, pc=pc)
                ra = self.RA_min + (self.RA_max - self.RA_min) * (1 - self.gen / self.max_gen)

                for child in (child1, child2):
                    self.adapative_pm(population, child)
                    if np.random.rand() < self.mut_prob:
                        self.mutate2(child) if np.random.rand() < ra else self.mutate1(child)
                    self.evaluate(child)

                retry_count = 0
                while not (child1.feasible and child2.feasible) and retry_count < max_retry_count:
                    retry_count += 1
                    child1, child2 = self.crossover(parent1, parent2, pc=pc)
                    for child in (child1, child2):
                        self.adapative_pm(population, child)
                        if np.random.rand() < self.mut_prob:
                            self.mutate2(child) if np.random.rand() < ra else self.mutate1(child)
                        self.evaluate(child)

                if child1.feasible and child2.feasible:
                    if (
                        not self.is_individual_in_population(child1, population)
                        and not self.is_individual_in_population(child2, population)
                    ):
                        population.sort(key=lambda x: x.cost)
                        if child1.cost < population[-2].cost and child2.cost < population[-2].cost:
                            population = population[:-2] + [child1, child2]

                if gen % 50 == 0:
                    population.sort(key=lambda x: x.cost)
                    elite_count = min(5, len(population))
                    for idx in range(elite_count):
                        local_started = time.perf_counter()
                        with contextlib.redirect_stdout(io.StringIO()):
                            population[idx] = module.LocalSearch(population[idx], max_iter=5).run()
                        stats.local_search_sec += time.perf_counter() - local_started
                        stats.local_search_calls += 1

                current_best = min(population, key=lambda x: x.cost)
                if current_best.cost < best_individual.cost:
                    best_individual = current_best
                avg_cost = sum(ind.cost for ind in population) / len(population)
                feasible_count = sum(1 for ind in population if ind.feasible)
                history.append((gen, best_individual.cost, avg_cost, feasible_count))

            stats.loop_total_sec = time.perf_counter() - loop_started
            stats.best_cost = best_individual.cost
            stats.feasible_count = sum(1 for ind in population if ind.feasible)
            return best_individual, history

    return TimedGA


def run_case(
    module: Any,
    data: dict[str, Any],
    base_count: int,
    supplier_count: int,
    pop_size: int,
    max_gen: int,
    solver_time_limit: float,
    max_retry_count: int,
    seed: int,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    set_module_instance(module, data, base_count, supplier_count)
    stats = RuntimeStats()
    timed_ga_class = make_timed_ga_class(module, stats, solver_time_limit, max_retry_count)
    ga = timed_ga_class(pop_size=pop_size, max_gen=max_gen, mut_prob=0.2, max_time=10**9, RA_max=0.8, RA_min=0.5)
    with contextlib.redirect_stdout(io.StringIO()):
        ga.run()

    other_sec = max(0.0, stats.loop_total_sec - stats.greedy_decode_sec - stats.solver_decode_sec - stats.local_search_sec)
    case_name = f"{base_count}基地{supplier_count}供应商"
    total = stats.loop_total_sec if stats.loop_total_sec > 0 else 1.0
    return {
        "case_name": case_name,
        "base_count": base_count,
        "supplier_count": supplier_count,
        "population_size": pop_size,
        "max_generation": max_gen,
        "seed": seed,
        "solver_time_limit_sec": solver_time_limit,
        "max_retry_count": max_retry_count,
        "initialization_sec": stats.initialization_sec,
        "loop_total_sec": stats.loop_total_sec,
        "greedy_decode_sec": stats.greedy_decode_sec,
        "solver_decode_sec": stats.solver_decode_sec,
        "local_search_sec": stats.local_search_sec,
        "other_ga_sec": other_sec,
        "greedy_decode_pct": stats.greedy_decode_sec / total,
        "solver_decode_pct": stats.solver_decode_sec / total,
        "local_search_pct": stats.local_search_sec / total,
        "other_ga_pct": other_sec / total,
        "greedy_decode_calls": stats.greedy_decode_calls,
        "solver_decode_calls": stats.solver_decode_calls,
        "local_search_calls": stats.local_search_calls,
        "best_cost": stats.best_cost,
        "feasible_count": stats.feasible_count,
    }


def plot_distribution(df: pd.DataFrame, output_path: Path) -> None:
    configure_chinese_font()
    labels = ["贪心解码", "数学规划解码", "局部搜索", "其他遗传操作"]
    pct_cols = ["greedy_decode_pct", "solver_decode_pct", "local_search_pct", "other_ga_pct"]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]

    x = np.arange(len(df))
    width = 0.18
    plt.figure(figsize=(9.5, 5.6))
    for idx, (label, col, color) in enumerate(zip(labels, pct_cols, colors)):
        vals = df[col].to_numpy() * 100
        xpos = x + (idx - 1.5) * width
        bars = plt.bar(xpos, vals, width=width, label=label, color=color)
        for bar, val in zip(bars, vals):
            if val >= 1:
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    plt.xticks(x, df["case_name"])
    plt.ylabel("耗时占比（%）")
    plt.xlabel("算例规模")
    plt.title("遗传算法计算时间分布")
    plt.ylim(0, max(100, float((df[pct_cols].max().max() * 100) + 8)))
    plt.grid(axis="y", alpha=0.25)
    plt.legend(ncol=2, frameon=False)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_split_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Plot total runtime share plus a zoomed non-solver share panel.

    Mathematical-programming decoding often occupies more than 99% of the
    runtime. A single grouped bar chart therefore hides the smaller components.
    This two-panel chart keeps the full distribution visible on the left and
    separately rescales the non-solver components on the right.
    """
    configure_chinese_font()
    total_labels = ["贪心解码", "数学规划解码", "局部搜索", "其他遗传操作"]
    total_cols = ["greedy_decode_pct", "solver_decode_pct", "local_search_pct", "other_ga_pct"]
    total_colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]

    zoom_labels = ["贪心解码", "局部搜索", "其他遗传操作"]
    zoom_cols = ["greedy_decode_pct", "local_search_pct", "other_ga_pct"]
    zoom_colors = ["#4C78A8", "#54A24B", "#B279A2"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [1.05, 1.15]})
    x = np.arange(len(df))

    bottom = np.zeros(len(df))
    for label, col, color in zip(total_labels, total_cols, total_colors):
        vals = df[col].to_numpy() * 100
        axes[0].bar(x, vals, bottom=bottom, label=label, color=color, width=0.55)
        bottom += vals
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["case_name"])
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("总耗时占比（%）")
    axes[0].set_title("总耗时分布")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=2)

    zoom_df = df[zoom_cols].copy()
    zoom_den = zoom_df.sum(axis=1).replace(0, np.nan)
    zoom_df = zoom_df.div(zoom_den, axis=0) * 100
    width = 0.22
    for idx, (label, col, color) in enumerate(zip(zoom_labels, zoom_cols, zoom_colors)):
        vals = zoom_df[col].to_numpy()
        xpos = x + (idx - 1) * width
        bars = axes[1].bar(xpos, vals, width=width, label=label, color=color)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.2,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["case_name"])
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("非数学规划解码耗时占比（%）")
    axes[1].set_title("剔除数学规划解码后的局部放大")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3)

    fig.suptitle("遗传算法计算时间分布", fontsize=15)
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 GA 计算时间分布中文条形图")
    parser.add_argument("--pop-size", type=int, default=30, help="种群规模，默认 30")
    parser.add_argument("--max-gen", type=int, default=100, help="迭代代数，默认 100")
    parser.add_argument("--solver-time-limit", type=float, default=3.0, help="单次数学规划解码时间上限，默认 3 秒")
    parser.add_argument("--max-retry-count", type=int, default=5, help="每代不可行子代的最大重试次数，默认 5")
    parser.add_argument("--seed", type=int, default=0, help="随机种子，默认 0")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module = load_ga_module()
    data = load_data(module)
    cases = [(2, 20), (4, 40)]
    records = [
        run_case(
            module=module,
            data=data,
            base_count=base_count,
            supplier_count=supplier_count,
            pop_size=args.pop_size,
            max_gen=args.max_gen,
            solver_time_limit=args.solver_time_limit,
            max_retry_count=args.max_retry_count,
            seed=args.seed,
        )
        for base_count, supplier_count in cases
    ]
    df = pd.DataFrame(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "runtime_distribution_summary.csv"
    fig_path = args.output_dir / "runtime_distribution_bar.png"
    split_fig_path = args.output_dir / "runtime_distribution_split_bar.png"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    plot_distribution(df, fig_path)
    plot_split_distribution(df, split_fig_path)
    print(f"统计结果已保存：{csv_path}")
    print(f"计算时间分布图已保存：{fig_path}")
    print(f"双视图时间分布图已保存：{split_fig_path}")


if __name__ == "__main__":
    main()
