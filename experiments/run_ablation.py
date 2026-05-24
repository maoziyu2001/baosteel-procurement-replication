#!/usr/bin/env python3
"""Run component ablation experiments for the multi-base procurement GA.

The script is intentionally an experiment layer around the existing project
code. It reuses the current greedy decoder, mathematical-programming decoder,
mutation operators, and local-search neighborhood logic, while adding
configuration switches, instrumentation, reproducible seeds, and paper-ready
outputs.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCUREMENT_DIR = ROOT / "src" / "ga_hybrid"
sys.path.insert(0, str(PROCUREMENT_DIR))

from decode_yijt import solve_procument_plan as solve_solver_decode  # noqa: E402
from greedy_decode_yijt import solve_procument_plan_greedy  # noqa: E402
from lowerbound_gurobi import solve_procument_plan as solve_lower_bound_model  # noqa: E402
from read_excel_file import (  # noqa: E402
    read_xlsx_file_base,
    read_xlsx_file_supplier,
    read_xlsx_file_transport,
)
import GA_updated_crossover as ga_core  # noqa: E402


CASE_DEFINITIONS: dict[str, tuple[int, int]] = {
    "case_2": (2, 20),
    "case_10": (2, 25),
    "case_13": (4, 30),
    "case_20": (4, 40),
}


@dataclass(frozen=True)
class ExperimentConfig:
    algorithm_name: str
    enable_staged_decoding: bool = True
    decoder_mode: str = "hybrid"  # hybrid / greedy_only / solver_only
    enable_local_search: bool = True
    mutation_mode: str = "mutate1+mutate2"  # mutate1 / mutate2 / mutate1+mutate2
    enabled_neighborhoods: list[str] = field(
        default_factory=lambda: ["single_flip", "single_base_consolidation", "cross_period_consolidation"]
    )
    population_size: int = 80
    max_iter: int = 500
    local_search_period: int = 50
    local_search_elite_count: int = 5
    time_limit: float = 3600.0
    random_seed: int = 0
    solver_time_limit: float = 60.0


@dataclass
class Individual:
    yijt: np.ndarray
    cost: float = 1e12
    decode: np.ndarray | None = None
    feasible: bool = False
    lowerbound: float | None = None
    fitness: float = 0.0

    def clone_empty(self) -> "Individual":
        return Individual(self.yijt.copy())


@dataclass
class RunStats:
    num_greedy_decode_calls: int = 0
    num_solver_decode_calls: int = 0
    total_greedy_decode_time_sec: float = 0.0
    total_solver_decode_time_sec: float = 0.0
    num_local_search_calls: int = 0
    num_local_search_improvements: int = 0
    total_local_search_improvement: float = 0.0


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_data() -> dict[str, Any]:
    base_file = ROOT / "data" / "raw_excel_optional" / "base_data.xlsx"
    supplier_file = ROOT / "data" / "raw_excel_optional" / "supplier_data.xlsx"
    transport_file = ROOT / "data" / "raw_excel_optional" / "transport_data.xlsx"
    n, I, beta, W, w0, e0, H, f, D, S, alpha, L = read_xlsx_file_base(base_file)
    m, J, R, J1, J2, Jsp, z, l, u, r, Q, P = read_xlsx_file_supplier(supplier_file)
    G, cjg, cgi, cij = read_xlsx_file_transport(transport_file)
    return locals()


def case_to_sets(case_id: str) -> tuple[list[int], list[int]]:
    if case_id not in CASE_DEFINITIONS:
        raise ValueError(f"Unknown case_id={case_id}. Known cases: {sorted(CASE_DEFINITIONS)}")
    n_bases, n_suppliers = CASE_DEFINITIONS[case_id]
    return list(range(n_bases)), list(range(n_suppliers))


def configure_local_search_globals(data: dict[str, Any], I: list[int], J: list[int]) -> None:
    """The existing LocalSearch class reads module globals; set them per case."""
    for name in [
        "n", "beta", "W", "w0", "e0", "H", "f", "D", "S", "alpha", "L",
        "m", "R", "J1", "J2", "Jsp", "z", "l", "u", "r", "Q", "P",
        "G", "cjg", "cgi", "cij",
    ]:
        setattr(ga_core, name, data[name])
    ga_core.I = I
    ga_core.J = J
    ga_core.t = 0
    ga_core.K = 1
    ga_core.T = [0]
    ga_core.tau_list = [0, 1]


class ConfigurableLocalSearch(ga_core.LocalSearch):
    """Enable/disable the three existing neighborhoods without changing core code."""

    def __init__(self, ind: Individual, enabled_neighborhoods: list[str], max_iter: int = 5):
        super().__init__(ind, max_iter=max_iter)
        self.enabled_neighborhoods = set(enabled_neighborhoods)

    def run(self, max_neighbors: int = 100) -> Individual:  # noqa: ARG002 - kept for compatibility
        improved = True
        iteration = 0
        while improved and iteration < self.max_iter:
            improved = False
            iteration += 1
            neighbor_solutions = []
            if "single_flip" in self.enabled_neighborhoods:
                neighbor_solutions.extend(self.generate_neighbors_1())
            if "single_base_consolidation" in self.enabled_neighborhoods:
                neighbor_solutions.extend(self.generate_neighbors_2())
            if "cross_period_consolidation" in self.enabled_neighborhoods:
                neighbor_solutions.extend(self.generate_neighbors_3())
            if not neighbor_solutions:
                break
            neighbor_solutions.sort(key=lambda x: x.cost)
            if neighbor_solutions[0].cost + 1e-6 < self.best_solution.cost:
                self.best_solution = neighbor_solutions[0]
                improved = True
            else:
                break
        return self.best_solution


class AblationGA:
    def __init__(self, data: dict[str, Any], case_id: str, config: ExperimentConfig):
        self.data = data
        self.case_id = case_id
        self.I, self.J = case_to_sets(case_id)
        self.config = config
        self.t = 0
        self.K = 1
        self.T = [0]
        self.tau_list = [0, 1]
        self.stats = RunStats()
        self.start_time = 0.0

    def timed_out(self) -> bool:
        return time.time() - self.start_time >= self.config.time_limit

    def random_yijt(self) -> np.ndarray:
        shape = (len(self.I), len(self.J), len(self.T), len(self.tau_list))
        Jsp = set(int(x) for x in self.data["Jsp"])
        for _ in range(3000):
            yijt = np.random.randint(0, 2, size=shape, dtype=int)
            for i in self.I:
                for j in self.J:
                    if self.data["alpha"][i][j] < self.data["beta"][i]:
                        yijt[i, j, 0, :] = 0
            ok = True
            for j in self.J:
                if j in Jsp and np.sum(yijt[:, j, 0, :]) == 0:
                    ok = False
                    break
            if ok:
                return yijt
        raise RuntimeError("Failed to generate an initial chromosome satisfying basic supplier rules")

    def decoder_for_current_iter(self, iteration: int) -> str:
        if self.config.decoder_mode == "greedy_only":
            return "greedy"
        if self.config.decoder_mode == "solver_only":
            return "solver"
        if self.config.enable_staged_decoding and iteration < max(1, self.config.max_iter // 2):
            return "greedy"
        return "solver"

    def evaluate(self, ind: Individual, iteration: int) -> Individual:
        decoder = self.decoder_for_current_iter(iteration)
        before = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            if decoder == "greedy":
                feasible, cost, decode, lower = solve_procument_plan_greedy(
                    self.I, self.data["beta"], self.data["W"], self.data["w0"],
                    self.data["e0"], self.data["H"], self.data["f"], self.data["D"],
                    self.data["S"], self.data["alpha"], self.data["L"], self.J,
                    self.data["R"], self.data["Jsp"], self.data["z"], self.data["l"],
                    self.data["u"], self.data["r"], self.data["Q"], self.data["P"],
                    self.data["cij"], self.t, self.K, self.tau_list, self.T, ind.yijt,
                )
                self.stats.num_greedy_decode_calls += 1
                self.stats.total_greedy_decode_time_sec += time.time() - before
            else:
                feasible, cost, decode, lower = solve_solver_decode(
                    self.data["n"], self.I, self.data["beta"], self.data["W"],
                    self.data["w0"], self.data["e0"], self.data["H"], self.data["f"],
                    self.data["D"], self.data["S"], self.data["alpha"], self.data["L"],
                    self.data["m"], self.J, self.data["R"], self.data["J1"],
                    self.data["J2"], self.data["Jsp"], self.data["z"], self.data["l"],
                    self.data["u"], self.data["r"], self.data["Q"], self.data["P"],
                    self.data["G"], self.data["cjg"], self.data["cgi"], self.data["cij"],
                    self.t, self.K, self.tau_list, self.T, ind.yijt,
                    time_limit=self.config.solver_time_limit, outputflag=0,
                )
                self.stats.num_solver_decode_calls += 1
                self.stats.total_solver_decode_time_sec += time.time() - before
        ind.feasible = bool(feasible)
        ind.cost = float(cost) if cost is not None else 1e12
        ind.decode = decode
        ind.lowerbound = float(lower) if lower is not None else None
        return ind

    def initialize_population(self) -> list[Individual]:
        population: list[Individual] = []
        attempts = 0
        while len(population) < self.config.population_size:
            if self.timed_out():
                break
            attempts += 1
            if attempts > self.config.population_size * 200:
                raise RuntimeError("Too many failed attempts while building initial population")
            ind = self.evaluate(Individual(self.random_yijt()), iteration=0)
            if ind.feasible:
                population.append(ind)
        if not population:
            raise RuntimeError("No feasible individual found during initialization")
        return population

    @staticmethod
    def update_fitness(population: list[Individual]) -> None:
        costs = [ind.cost for ind in population]
        max_cost, min_cost = max(costs), min(costs)
        for ind in population:
            ind.fitness = (max_cost - ind.cost) / (max_cost - min_cost + 1e-4)

    def tournament_selection(self, population: list[Individual], k: int = 3) -> Individual:
        k = min(k, len(population))
        contestants = np.random.choice(population, k, replace=False)
        return min(contestants, key=lambda x: x.cost)

    def crossover(self, parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
        child1 = parent1.clone_empty()
        child2 = parent2.clone_empty()
        for j in self.J:
            if np.random.rand() < 0.5:
                child1.yijt[:, j, :, :] = parent1.yijt[:, j, :, :]
                child2.yijt[:, j, :, :] = parent2.yijt[:, j, :, :]
            else:
                child1.yijt[:, j, :, :] = parent2.yijt[:, j, :, :]
                child2.yijt[:, j, :, :] = parent1.yijt[:, j, :, :]
        return child1, child2

    def mutate1(self, ind: Individual) -> None:
        for i in self.I:
            for j in self.J:
                for tau in self.tau_list:
                    if np.random.rand() < 0.15:
                        ind.yijt[i, j, 0, tau] = 1 - ind.yijt[i, j, 0, tau]
                    if self.data["alpha"][i][j] < self.data["beta"][i]:
                        ind.yijt[i, j, 0, tau] = 0

    def mutate2(self, ind: Individual) -> None:
        i = random.randint(0, len(self.I) - 1)
        tau = random.randint(0, len(self.tau_list) - 1)
        current_suppliers = [j for j in self.J if ind.yijt[i, j, 0, tau] == 1]
        total_decisions = max(1, int(np.sum(ind.yijt)))
        ave_supplier_num = max(1, int(total_decisions / (len(self.tau_list) * len(self.I))))
        if current_suppliers and len(current_suppliers) >= ave_supplier_num:
            new_count = random.randint(1, max(1, len(current_suppliers) - 1))
            new_suppliers = random.sample(current_suppliers, min(new_count, len(current_suppliers)))
        else:
            candidates = [j for j in self.J if self.data["alpha"][i][j] >= self.data["beta"][i]]
            new_count = random.randint(1, min(ave_supplier_num, len(candidates)))
            new_suppliers = random.sample(candidates, new_count)
        ind.yijt[i, :, 0, tau] = 0
        for j in new_suppliers:
            ind.yijt[i, j, 0, tau] = 1

    def mutate(self, ind: Individual, population: list[Individual], iteration: int) -> None:
        self.update_fitness(population)
        f_ave = sum(x.fitness for x in population) / len(population)
        f_max = max(x.fitness for x in population)
        pm = 0.5 * (f_max - ind.fitness) / (f_max - f_ave + 1e-6) if ind.fitness >= f_ave else 0.5
        if np.random.rand() >= pm:
            return
        mode = self.config.mutation_mode
        if mode == "mutate1":
            self.mutate1(ind)
        elif mode == "mutate2":
            self.mutate2(ind)
        else:
            ra = 0.5 + 0.3 * (1 - iteration / max(1, self.config.max_iter))
            self.mutate2(ind) if np.random.rand() < ra else self.mutate1(ind)

    @staticmethod
    def duplicate(ind: Individual, population: list[Individual]) -> bool:
        return any(np.array_equal(ind.yijt, old.yijt) for old in population)

    def maybe_local_search(self, population: list[Individual], iteration: int) -> None:
        if not self.config.enable_local_search:
            return
        if iteration <= 0 or iteration % self.config.local_search_period != 0:
            return
        configure_local_search_globals(self.data, self.I, self.J)
        population.sort(key=lambda x: x.cost)
        elite_count = min(self.config.local_search_elite_count, len(population))
        for idx in range(elite_count):
            if self.timed_out():
                return
            before_cost = population[idx].cost
            self.stats.num_local_search_calls += 1
            with contextlib.redirect_stdout(io.StringIO()):
                improved = ConfigurableLocalSearch(
                    population[idx],
                    self.config.enabled_neighborhoods,
                    max_iter=5,
                ).run()
            if improved.cost + 1e-6 < before_cost:
                self.stats.num_local_search_improvements += 1
                self.stats.total_local_search_improvement += before_cost - improved.cost
                population[idx] = improved

    def run(self) -> tuple[Individual, pd.DataFrame, str]:
        self.start_time = time.time()
        population = self.initialize_population()
        best = min(population, key=lambda x: x.cost)
        time_to_best = time.time() - self.start_time
        iter_to_best = 0
        history = []
        status = "success"

        for iteration in range(1, self.config.max_iter + 1):
            if self.timed_out():
                status = "timeout"
                break
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)
            children = list(self.crossover(parent1, parent2))
            accepted = []
            for child in children:
                self.mutate(child, population, iteration)
                child = self.evaluate(child, iteration)
                if child.feasible and not self.duplicate(child, population):
                    accepted.append(child)
            if accepted:
                population.sort(key=lambda x: x.cost)
                population = population[: max(1, len(population) - len(accepted))] + accepted
            self.maybe_local_search(population, iteration)
            current = min(population, key=lambda x: x.cost)
            if current.cost + 1e-6 < best.cost:
                best = current
                time_to_best = time.time() - self.start_time
                iter_to_best = iteration
            avg_cost = float(np.mean([x.cost for x in population]))
            feasible_count = sum(1 for x in population if x.feasible)
            history.append({
                "iteration": iteration,
                "best_cost": best.cost,
                "avg_cost": avg_cost,
                "feasible_count": feasible_count,
                "elapsed_sec": time.time() - self.start_time,
            })

        history_df = pd.DataFrame(history)
        history_df.attrs["time_to_best_sec"] = time_to_best
        history_df.attrs["iter_to_best"] = iter_to_best
        return best, history_df, status


def make_variants(args: argparse.Namespace, seed: int) -> list[ExperimentConfig]:
    base = dict(
        population_size=args.population_size,
        max_iter=args.max_iter,
        local_search_period=args.local_search_period,
        local_search_elite_count=args.local_search_elite_count,
        time_limit=args.time_limit,
        random_seed=seed,
        solver_time_limit=args.solver_time_limit,
    )
    all_neighborhoods = ["single_flip", "single_base_consolidation", "cross_period_consolidation"]
    return [
        ExperimentConfig("Full-GA-hybrid", True, "hybrid", True, "mutate1+mutate2", all_neighborhoods, **base),
        ExperimentConfig("w/o staged decoding", False, "greedy_only", True, "mutate1+mutate2", all_neighborhoods, **base),
        ExperimentConfig("solver-only decoding", False, "solver_only", True, "mutate1+mutate2", all_neighborhoods, **base),
        ExperimentConfig("w/o local search", True, "hybrid", False, "mutate1+mutate2", all_neighborhoods, **base),
        ExperimentConfig("adaptive mutation only", True, "hybrid", True, "mutate2", all_neighborhoods, **base),
        ExperimentConfig("random mutation only", True, "hybrid", True, "mutate1", all_neighborhoods, **base),
        ExperimentConfig("w/o single_flip neighborhood", True, "hybrid", True, "mutate1+mutate2", all_neighborhoods[1:], **base),
        ExperimentConfig("w/o single_base_consolidation neighborhood", True, "hybrid", True, "mutate1+mutate2", [all_neighborhoods[0], all_neighborhoods[2]], **base),
        ExperimentConfig("w/o cross_period_consolidation neighborhood", True, "hybrid", True, "mutate1+mutate2", all_neighborhoods[:2], **base),
    ]


def compute_case_lower_bound(data: dict[str, Any], case_id: str, time_limit: float) -> float | None:
    I, J = case_to_sets(case_id)
    with contextlib.redirect_stdout(io.StringIO()):
        result = solve_lower_bound_model(
            data["n"], I, data["beta"], data["W"], data["w0"], data["e0"], data["H"],
            data["f"], data["D"], data["S"], data["alpha"], data["L"], data["m"], J,
            data["R"], data["J1"], data["J2"], data["Jsp"], data["z"], data["l"],
            data["u"], data["r"], data["Q"], data["P"], data["G"], data["cjg"],
            data["cgi"], data["cij"], 0, 1, [0, 1], [0],
            time_limit=time_limit, outputflag=0, return_results=True,
        )
    return result.get("lower_bound")


def solution_metrics(best: Individual, data: dict[str, Any], J: list[int]) -> dict[str, Any]:
    if best.decode is None:
        return {
            "best_solution_supplier_count": pd.NA,
            "best_solution_total_purchase_quantity": pd.NA,
            "best_solution_discount_saving": pd.NA,
        }
    quantities = np.sum(best.decode, axis=(0, 2, 3))
    supplier_count = int(np.sum(quantities > 1e-6))
    total_qty = float(np.sum(best.decode))
    saving = 0.0
    for j in J:
        for tau in [0, 1]:
            q = float(np.sum(best.decode[:, j, 0, tau]))
            saving += float(data["r"][j] * min(q, data["Q"][j]) * q)
    return {
        "best_solution_supplier_count": supplier_count,
        "best_solution_total_purchase_quantity": total_qty,
        "best_solution_discount_saving": saving,
    }


def run_one(data: dict[str, Any], case_id: str, config: ExperimentConfig, lower_bound: float | None) -> tuple[dict[str, Any], pd.DataFrame]:
    set_reproducible_seed(config.random_seed)
    ga = AblationGA(data, case_id, config)
    started = time.time()
    error_message = ""
    history = pd.DataFrame()
    try:
        best, history, status = ga.run()
        if not best.feasible:
            status = "infeasible"
    except Exception as exc:  # Keep the full experiment batch alive.
        best = Individual(np.zeros((len(ga.I), len(ga.J), 1, 2), dtype=int))
        status = "error"
        error_message = f"{type(exc).__name__}: {str(exc)[:220]}"
        history = pd.DataFrame()
        traceback.print_exc()
    runtime = time.time() - started
    lb = lower_bound if lower_bound and lower_bound > 0 else best.lowerbound
    gap = (best.cost - lb) / lb * 100 if lb and lb > 0 and math.isfinite(best.cost) else pd.NA
    row = {
        "case_id": case_id,
        "variant_name": config.algorithm_name,
        "random_seed": config.random_seed,
        "status": status,
        "error_message": error_message,
        "best_cost": best.cost if math.isfinite(best.cost) else pd.NA,
        "lower_bound": lb if lb is not None else pd.NA,
        "gap_to_lower_bound_pct": gap,
        "runtime_sec": runtime,
        "time_to_best_sec": history.attrs.get("time_to_best_sec", pd.NA),
        "iter_to_best": history.attrs.get("iter_to_best", pd.NA),
        "final_feasible": bool(best.feasible),
        "num_greedy_decode_calls": ga.stats.num_greedy_decode_calls,
        "num_solver_decode_calls": ga.stats.num_solver_decode_calls,
        "total_greedy_decode_time_sec": ga.stats.total_greedy_decode_time_sec,
        "total_solver_decode_time_sec": ga.stats.total_solver_decode_time_sec,
        "num_local_search_calls": ga.stats.num_local_search_calls,
        "num_local_search_improvements": ga.stats.num_local_search_improvements,
        "total_local_search_improvement": ga.stats.total_local_search_improvement,
    }
    row.update(solution_metrics(best, data, ga.J))
    if not history.empty:
        history.insert(0, "random_seed", config.random_seed)
        history.insert(0, "variant_name", config.algorithm_name)
        history.insert(0, "case_id", case_id)
    return row, history


def write_outputs(raw: pd.DataFrame, histories: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_dir / "ablation_raw_results.csv", index=False)
    raw_for_summary = raw.copy()
    raw_for_summary["final_feasible"] = raw_for_summary["final_feasible"].astype(float)
    summary = raw_for_summary.groupby(["case_id", "variant_name"], dropna=False).agg(
        best_cost_mean=("best_cost", "mean"),
        best_cost_std=("best_cost", "std"),
        best_cost_min=("best_cost", "min"),
        best_cost_max=("best_cost", "max"),
        gap_mean=("gap_to_lower_bound_pct", "mean"),
        gap_std=("gap_to_lower_bound_pct", "std"),
        gap_min=("gap_to_lower_bound_pct", "min"),
        gap_max=("gap_to_lower_bound_pct", "max"),
        runtime_mean=("runtime_sec", "mean"),
        runtime_std=("runtime_sec", "std"),
        runtime_min=("runtime_sec", "min"),
        runtime_max=("runtime_sec", "max"),
        time_to_best_mean=("time_to_best_sec", "mean"),
        time_to_best_std=("time_to_best_sec", "std"),
        time_to_best_min=("time_to_best_sec", "min"),
        time_to_best_max=("time_to_best_sec", "max"),
        iter_to_best_mean=("iter_to_best", "mean"),
        iter_to_best_std=("iter_to_best", "std"),
        iter_to_best_min=("iter_to_best", "min"),
        iter_to_best_max=("iter_to_best", "max"),
        feasible_rate_mean=("final_feasible", "mean"),
        feasible_rate_std=("final_feasible", "std"),
        feasible_rate_min=("final_feasible", "min"),
        feasible_rate_max=("final_feasible", "max"),
        local_search_improvement_mean=("total_local_search_improvement", "mean"),
        local_search_improvement_std=("total_local_search_improvement", "std"),
        local_search_improvement_min=("total_local_search_improvement", "min"),
        local_search_improvement_max=("total_local_search_improvement", "max"),
    ).reset_index()
    summary.to_csv(output_dir / "ablation_summary.csv", index=False)
    write_latex_tables(summary, output_dir)
    write_convergence_plots(histories, output_dir)
    write_readme(output_dir)


def write_latex_tables(summary: pd.DataFrame, output_dir: Path) -> None:
    pieces = []
    for label, cases in [("small", ["case_2", "case_10"]), ("large", ["case_13", "case_20"])]:
        table = summary[summary["case_id"].isin(cases)].copy()
        if table.empty:
            continue
        table = table[[
            "case_id", "variant_name", "best_cost_mean", "gap_mean",
            "runtime_mean", "time_to_best_mean", "iter_to_best_mean",
        ]]
        table.columns = ["Case", "Method", "Best Cost", "Gap to LB (%)", "Runtime(s)", "Time to Best(s)", "Iter to Best"]
        pieces.append(f"% {label.capitalize()}-scale ablation table\n")
        pieces.append(render_latex_table(table))
        pieces.append("\n")
    (output_dir / "ablation_table_latex.tex").write_text("".join(pieces), encoding="utf-8")


def latex_escape(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def render_latex_table(table: pd.DataFrame) -> str:
    cols = list(table.columns)
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        " & ".join(latex_escape(col) for col in cols) + r" \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                vals.append("" if pd.isna(value) else f"{value:.2f}")
            else:
                vals.append(latex_escape(value))
        lines.append(" & ".join(vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def write_convergence_plots(histories: pd.DataFrame, output_dir: Path) -> None:
    if histories.empty:
        return
    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for case_id, case_df in histories.groupby("case_id"):
        plt.figure(figsize=(10, 5.5))
        for variant_name, var_df in case_df.groupby("variant_name"):
            curve = var_df.groupby("iteration")["best_cost"].mean().reset_index()
            plt.plot(curve["iteration"], curve["best_cost"], label=variant_name, linewidth=1.6)
        plt.xlabel("Iteration")
        plt.ylabel("Best cost")
        plt.title(f"Convergence curves - {case_id}")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(plot_dir / f"convergence_{case_id}.png", dpi=300)
        plt.close()
    histories.to_csv(output_dir / "ablation_convergence_history.csv", index=False)


def write_readme(output_dir: Path) -> None:
    text = """# Ablation Experiment Outputs

This folder contains the one-command ablation results for paper Section 3.4.

- `Full-GA-hybrid`: complete GA-hybrid with staged greedy/solver decoding, mixed mutation, and all local-search neighborhoods.
- `w/o staged decoding`: removes staged decoding and uses greedy decoding only.
- `solver-only decoding`: uses mathematical-programming decoding throughout; use a solver time limit for large cases.
- `w/o local search`: disables the local-search improvement step.
- `adaptive mutation only`: keeps only `mutate2`, the adaptive supplier-transfer mutation.
- `random mutation only`: keeps only `mutate1`, the random bit-flip mutation.
- `w/o single_flip neighborhood`: removes the single purchase-decision flip neighborhood.
- `w/o single_base_consolidation neighborhood`: removes same-base supplier consolidation.
- `w/o cross_period_consolidation neighborhood`: removes cross-period supplier consolidation.

Use `ablation_raw_results.csv` for seed-level reporting, `ablation_summary.csv` for mean/std tables, `ablation_table_latex.tex` directly in the manuscript, and `figures/convergence_*.png` for convergence plots.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GA-hybrid component ablation experiments.")
    parser.add_argument("--cases", nargs="+", default=["case_2", "case_10", "case_13", "case_20"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--population-size", type=int, default=80)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--local-search-period", type=int, default=50)
    parser.add_argument("--local-search-elite-count", type=int, default=5)
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--solver-time-limit", type=float, default=60.0)
    parser.add_argument("--lower-bound-time-limit", type=float, default=300.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "ablation")
    parser.add_argument("--skip-lower-bound", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data()
    raw_rows = []
    histories = []
    lower_bounds: dict[str, float | None] = {}
    for case_id in args.cases:
        if args.skip_lower_bound:
            lower_bounds[case_id] = None
        else:
            try:
                lower_bounds[case_id] = compute_case_lower_bound(data, case_id, args.lower_bound_time_limit)
            except Exception as exc:
                print(f"[WARN] lower bound failed for {case_id}: {exc}")
                lower_bounds[case_id] = None
        for seed in args.seeds:
            for config in make_variants(args, seed):
                print(f"[RUN] case={case_id} variant={config.algorithm_name} seed={seed}")
                row, history = run_one(data, case_id, config, lower_bounds[case_id])
                raw_rows.append(row)
                if not history.empty:
                    histories.append(history)
                pd.DataFrame(raw_rows).to_csv(args.output_dir / "ablation_raw_results.csv", index=False)
    raw = pd.DataFrame(raw_rows)
    history_df = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    write_outputs(raw, history_df, args.output_dir)
    print(f"Ablation outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
