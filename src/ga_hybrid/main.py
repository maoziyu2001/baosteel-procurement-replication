from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))
REPLICATION_ROOT = os.path.abspath(os.path.join(PACKAGE_DIR, "..", ".."))
sys.path.insert(0, PACKAGE_DIR)

from decode_yijt import solve_procument_plan
from greedy_decode_yijt import check_constraints as greedy_check_constraints
from greedy_decode_yijt import solve_procument_plan_greedy
from read_excel_file import (
    read_xlsx_file_base,
    read_xlsx_file_supplier,
    read_xlsx_file_transport,
)


BIG_M = 1e12


def fmt_money(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:,.2f}"


def pct(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-9:
        return 0.0
    return numerator / denominator * 100


@dataclass
class InstanceContext:
    n: int
    I: list[int]
    beta: Any
    W: Any
    w0: Any
    e0: Any
    H: Any
    f: Any
    D: Any
    S: Any
    alpha: Any
    L: Any
    m: int
    J: list[int]
    R: Any
    J1: list[int]
    J2: list[int]
    Jsp: list[int]
    z: Any
    l: Any
    u: Any
    r: Any
    Q: Any
    P: Any
    G: Any
    cjg: Any
    cgi: Any
    cij: Any
    t: int
    K: int
    T: list[int]
    tau_list: list[int]
    instance_name: str


@dataclass
class Individual:
    yijt: np.ndarray
    cost: float = BIG_M
    decode: np.ndarray | None = None
    feasible: bool = False
    lowerbound: float | None = None
    fitness: float = 0.0

    @classmethod
    def empty(cls, ctx: InstanceContext) -> "Individual":
        shape = (len(ctx.I), len(ctx.J), len(ctx.T), len(ctx.tau_list))
        return cls(yijt=np.zeros(shape, dtype=int), decode=np.zeros(shape))


class AcademicTraceLogger:
    """Write a readable paper-style log and a machine-auditable JSONL trace."""

    def __init__(self, output_dir: str, run_id: str):
        os.makedirs(output_dir, exist_ok=True)
        self.run_id = run_id
        self.text_path = os.path.join(output_dir, f"{run_id}.log")
        self.jsonl_path = os.path.join(output_dir, f"{run_id}.jsonl")
        self.start_time = time.time()
        self._text = open(self.text_path, "w", encoding="utf-8")
        self._jsonl = open(self.jsonl_path, "w", encoding="utf-8")

    def close(self):
        self._text.close()
        self._jsonl.close()

    def emit(self, event: str, level: str = "INFO", **fields):
        elapsed = time.time() - self.start_time
        record = {
            "run_id": self.run_id,
            "elapsed_sec": round(elapsed, 3),
            "level": level,
            "event": event,
            **fields,
        }
        self._jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._jsonl.flush()

        parts = [
            f"[{elapsed:08.2f}s]",
            f"[{level}]",
            f"[{event}]",
        ]
        for key, value in fields.items():
            if isinstance(value, float):
                text = f"{value:,.4f}" if abs(value) >= 10000 else f"{value:.4f}"
            else:
                text = str(value)
            parts.append(f"{key}={text}")
        line = " ".join(parts)
        print(line)
        self._text.write(line + "\n")
        self._text.flush()


class TracedLocalSearch:
    def __init__(
        self,
        ctx: InstanceContext,
        ind: Individual,
        logger: AcademicTraceLogger,
        gen: int,
        elite_rank: int,
        max_iter: int = 5,
    ):
        self.ctx = ctx
        self.best_solution = copy.deepcopy(ind)
        self.logger = logger
        self.gen = gen
        self.elite_rank = elite_rank
        self.max_iter = max_iter

    def check_neighbor(self, neighbor: Individual) -> tuple[bool, float]:
        with contextlib.redirect_stdout(io.StringIO()):
            result = greedy_check_constraints(
                neighbor.decode,
                self.ctx.I,
                self.ctx.beta,
                self.ctx.W,
                self.ctx.w0,
                self.ctx.e0,
                self.ctx.H,
                self.ctx.f,
                self.ctx.D,
                self.ctx.S,
                self.ctx.alpha,
                self.ctx.L,
                self.ctx.J,
                self.ctx.R,
                self.ctx.Jsp,
                self.ctx.z,
                self.ctx.l,
                self.ctx.u,
                self.ctx.r,
                self.ctx.Q,
                self.ctx.P,
                self.ctx.cij,
                self.ctx.t,
                self.ctx.K,
                self.ctx.tau_list,
                self.ctx.T,
                neighbor.yijt,
            )
        if not isinstance(result, tuple):
            return False, BIG_M
        return result

    def generate_neighbors_1(self) -> tuple[list[Individual], int]:
        """N1: single inversion. Remove one active base-supplier-period link."""
        candidates = 0
        neighbors: list[Individual] = []
        t = self.ctx.t
        for i in self.ctx.I:
            for j in self.ctx.J:
                for tau in self.ctx.tau_list:
                    if self.best_solution.yijt[i, j, t, tau] == 1:
                        candidates += 1
                        neighbor = copy.deepcopy(self.best_solution)
                        neighbor.yijt[i, j, t, tau] = 0
                        neighbor.decode[i, j, t, tau] = 0
                        flag, goal = self.check_neighbor(neighbor)
                        if flag:
                            neighbor.cost = goal
                            neighbor.lowerbound = min(
                                goal,
                                neighbor.lowerbound if neighbor.lowerbound is not None else goal,
                            )
                            neighbor.feasible = True
                            neighbors.append(neighbor)
        return neighbors, candidates

    def generate_neighbors_2(self) -> tuple[list[Individual], int]:
        """N2: single-base consolidation. Merge one supplier order into another."""
        candidates = 0
        neighbors: list[Individual] = []
        t = self.ctx.t
        for i in self.ctx.I:
            for tau in self.ctx.tau_list:
                for j1_idx in range(0, len(self.ctx.J) - 1):
                    for j2_idx in range(j1_idx + 1, len(self.ctx.J)):
                        j1 = self.ctx.J[j1_idx]
                        j2 = self.ctx.J[j2_idx]
                        if (
                            self.best_solution.yijt[i, j1, t, tau] == 1
                            and self.best_solution.yijt[i, j2, t, tau] == 1
                        ):
                            q1 = self.best_solution.decode[i, j1, t, tau]
                            q2 = self.best_solution.decode[i, j2, t, tau]
                            if q1 <= 0 or q2 <= 0:
                                continue
                            candidates += 1
                            iron1 = q1 * self.ctx.z[j1]
                            iron2 = q2 * self.ctx.z[j2]
                            remove_j, keep_j = (j1, j2) if iron1 <= iron2 else (j2, j1)
                            removed_iron = (
                                self.best_solution.decode[i, remove_j, t, tau]
                                * self.ctx.z[remove_j]
                            )

                            neighbor = copy.deepcopy(self.best_solution)
                            neighbor.yijt[i, remove_j, t, tau] = 0
                            neighbor.decode[i, remove_j, t, tau] = 0
                            neighbor.decode[i, keep_j, t, tau] += removed_iron / self.ctx.z[keep_j]
                            flag, goal = self.check_neighbor(neighbor)
                            if flag:
                                neighbor.cost = goal
                                neighbor.lowerbound = min(
                                    goal,
                                    neighbor.lowerbound if neighbor.lowerbound is not None else goal,
                                )
                                neighbor.feasible = True
                                neighbors.append(neighbor)
        return neighbors, candidates

    def generate_neighbors_3(self) -> tuple[list[Individual], int]:
        """N3: cross-period consolidation. Move later order into an earlier period."""
        candidates = 0
        neighbors: list[Individual] = []
        t = self.ctx.t
        for i in self.ctx.I:
            for j in self.ctx.J:
                for tau1_idx in range(0, len(self.ctx.tau_list) - 1):
                    for tau2_idx in range(tau1_idx + 1, len(self.ctx.tau_list)):
                        tau1 = self.ctx.tau_list[tau1_idx]
                        tau2 = self.ctx.tau_list[tau2_idx]
                        if (
                            self.best_solution.yijt[i, j, t, tau1] == 1
                            and self.best_solution.yijt[i, j, t, tau2] == 1
                            and self.best_solution.decode[i, j, t, tau2] > 0
                        ):
                            candidates += 1
                            neighbor = copy.deepcopy(self.best_solution)
                            neighbor.yijt[i, j, t, tau2] = 0
                            neighbor.decode[i, j, t, tau1] += neighbor.decode[i, j, t, tau2]
                            neighbor.decode[i, j, t, tau2] = 0
                            flag, goal = self.check_neighbor(neighbor)
                            if flag:
                                neighbor.cost = goal
                                neighbor.lowerbound = min(
                                    goal,
                                    neighbor.lowerbound if neighbor.lowerbound is not None else goal,
                                )
                                neighbor.feasible = True
                                neighbors.append(neighbor)
        return neighbors, candidates

    def run(self) -> tuple[Individual, dict[str, Any]]:
        start_cost = self.best_solution.cost
        total_candidates = 0
        total_feasible = 0
        improvements = 0
        self.logger.emit(
            "LOCAL_SEARCH_START",
            gen=self.gen,
            elite_rank=self.elite_rank,
            max_iter=self.max_iter,
            neighborhoods="N1_single_inversion,N2_single_base_consolidation,N3_cross_period_consolidation",
            start_cost=round(start_cost, 4),
        )

        for iteration in range(1, self.max_iter + 1):
            neighborhood_generators = [
                ("N1_single_inversion", self.generate_neighbors_1),
                ("N2_single_base_consolidation", self.generate_neighbors_2),
                ("N3_cross_period_consolidation", self.generate_neighbors_3),
            ]
            all_neighbors: list[tuple[str, Individual]] = []
            iteration_candidates = 0
            iteration_feasible = 0

            for name, generator in neighborhood_generators:
                neighbors, candidates = generator()
                total_candidates += candidates
                total_feasible += len(neighbors)
                iteration_candidates += candidates
                iteration_feasible += len(neighbors)
                best_neighbor_cost = min((x.cost for x in neighbors), default=None)
                self.logger.emit(
                    "LOCAL_SEARCH_NEIGHBORHOOD",
                    gen=self.gen,
                    elite_rank=self.elite_rank,
                    ls_iter=iteration,
                    neighborhood=name,
                    candidate_neighbors=candidates,
                    feasible_neighbors=len(neighbors),
                    best_neighbor_cost=round(best_neighbor_cost, 4)
                    if best_neighbor_cost is not None
                    else "NA",
                )
                all_neighbors.extend((name, neighbor) for neighbor in neighbors)

            if not all_neighbors:
                self.logger.emit(
                    "LOCAL_SEARCH_ITER",
                    gen=self.gen,
                    elite_rank=self.elite_rank,
                    ls_iter=iteration,
                    candidate_neighbors=iteration_candidates,
                    feasible_neighbors=iteration_feasible,
                    accepted="no",
                    reason="no_feasible_neighbor",
                    current_cost=round(self.best_solution.cost, 4),
                )
                break

            best_name, best_neighbor = min(all_neighbors, key=lambda item: item[1].cost)
            if best_neighbor.cost < self.best_solution.cost - 1e-6:
                old_cost = self.best_solution.cost
                self.best_solution = best_neighbor
                improvements += 1
                self.logger.emit(
                    "LOCAL_SEARCH_ITER",
                    gen=self.gen,
                    elite_rank=self.elite_rank,
                    ls_iter=iteration,
                    selected_neighborhood=best_name,
                    accepted="yes",
                    old_cost=round(old_cost, 4),
                    new_cost=round(best_neighbor.cost, 4),
                    improvement=round(old_cost - best_neighbor.cost, 4),
                    improvement_pct=round(pct(old_cost - best_neighbor.cost, old_cost), 4),
                )
            else:
                self.logger.emit(
                    "LOCAL_SEARCH_ITER",
                    gen=self.gen,
                    elite_rank=self.elite_rank,
                    ls_iter=iteration,
                    selected_neighborhood=best_name,
                    accepted="no",
                    reason="no_improving_neighbor",
                    current_cost=round(self.best_solution.cost, 4),
                    best_neighbor_cost=round(best_neighbor.cost, 4),
                )
                break

        self.logger.emit(
            "LOCAL_SEARCH_END",
            gen=self.gen,
            elite_rank=self.elite_rank,
            start_cost=round(start_cost, 4),
            end_cost=round(self.best_solution.cost, 4),
            total_improvement=round(start_cost - self.best_solution.cost, 4),
            total_improvement_pct=round(pct(start_cost - self.best_solution.cost, start_cost), 4),
            total_candidate_neighbors=total_candidates,
            total_feasible_neighbors=total_feasible,
            accepted_improvements=improvements,
        )
        return self.best_solution, {
            "candidate_neighbors": total_candidates,
            "feasible_neighbors": total_feasible,
            "improvements": improvements,
            "improvement": start_cost - self.best_solution.cost,
        }


class AcademicTraceGA:
    def __init__(
        self,
        ctx: InstanceContext,
        logger: AcademicTraceLogger,
        pop_size: int,
        max_gen: int,
        max_time: float,
        k_explore: int,
        k_ls: int,
        elite_ls: int,
        ls_max_iter: int,
        mut_prob: float,
        ra_max: float,
        ra_min: float,
    ):
        self.ctx = ctx
        self.logger = logger
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.max_time = max_time
        self.k_explore = k_explore
        self.k_ls = k_ls
        self.elite_ls = elite_ls
        self.ls_max_iter = ls_max_iter
        self.mut_prob = mut_prob
        self.ra_max = ra_max
        self.ra_min = ra_min
        self.gen = 0
        self.k1 = self.k2 = self.k3 = self.k4 = 0.5
        self.decode_stats = {
            "greedy_calls": 0,
            "solver_calls": 0,
            "greedy_time_sec": 0.0,
            "solver_time_sec": 0.0,
        }

    def decoder_name(self, gen: int) -> str:
        return "greedy" if gen < self.k_explore else "solver"

    def phase_name(self, gen: int) -> str:
        return "exploration" if gen < self.k_explore else "exploitation"

    def repair_long_contract_selection(self, yijt: np.ndarray) -> tuple[np.ndarray, int]:
        repairs = 0
        for jsp in self.ctx.Jsp:
            if jsp not in self.ctx.J:
                continue
            eligible_bases = [i for i in self.ctx.I if self.ctx.alpha[i][jsp] >= self.ctx.beta[i]]
            if not eligible_bases:
                continue
            for tau in self.ctx.tau_list:
                if sum(yijt[i, jsp, self.ctx.t, tau] for i in eligible_bases) == 0:
                    i = random.choice(eligible_bases)
                    yijt[i, jsp, self.ctx.t, tau] = 1
                    repairs += 1
        return yijt, repairs

    def initialize_population(self) -> list[Individual]:
        population: list[Individual] = []
        total_attempts = 0
        total_repairs = 0
        shape = (len(self.ctx.I), len(self.ctx.J), len(self.ctx.T), len(self.ctx.tau_list))
        max_attempts = 2000
        for idx in range(self.pop_size):
            flag = False
            attempts = 0
            while not flag and attempts < max_attempts:
                attempts += 1
                total_attempts += 1
                yijt = np.random.randint(0, 2, size=shape, dtype=int)
                for i in self.ctx.I:
                    for j in self.ctx.J:
                        if self.ctx.alpha[i][j] < self.ctx.beta[i]:
                            for tau in self.ctx.tau_list:
                                yijt[i, j, self.ctx.t, tau] = 0
                yijt, repairs = self.repair_long_contract_selection(yijt)
                total_repairs += repairs
                ind = Individual(yijt=yijt, decode=np.zeros(shape))
                self.evaluate(ind, gen=0, log_detail=False)
                flag = ind.feasible
            if not flag:
                raise RuntimeError(
                    f"Failed to initialize individual {idx}; no feasible chromosome after {max_attempts} attempts."
                )
            population.append(ind)
        best = min(population, key=lambda x: x.cost)
        avg_cost = sum(ind.cost for ind in population) / len(population)
        self.logger.emit(
            "INIT",
            instance=self.ctx.instance_name,
            population=len(population),
            feasible=sum(1 for ind in population if ind.feasible),
            init_decoder="greedy",
            total_attempts=total_attempts,
            long_contract_repairs=total_repairs,
            best_cost=round(best.cost, 4),
            avg_cost=round(avg_cost, 4),
        )
        return population

    def evaluate(self, individual: Individual, gen: int, log_detail: bool = False, child_label: str = ""):
        individual.yijt, repairs = self.repair_long_contract_selection(individual.yijt)
        decoder = self.decoder_name(gen)
        tic = time.time()
        if decoder == "greedy":
            with contextlib.redirect_stdout(io.StringIO()):
                feasible, cost, decode, lowerbound = solve_procument_plan_greedy(
                    self.ctx.I,
                    self.ctx.beta,
                    self.ctx.W,
                    self.ctx.w0,
                    self.ctx.e0,
                    self.ctx.H,
                    self.ctx.f,
                    self.ctx.D,
                    self.ctx.S,
                    self.ctx.alpha,
                    self.ctx.L,
                    self.ctx.J,
                    self.ctx.R,
                    self.ctx.Jsp,
                    self.ctx.z,
                    self.ctx.l,
                    self.ctx.u,
                    self.ctx.r,
                    self.ctx.Q,
                    self.ctx.P,
                    self.ctx.cij,
                    self.ctx.t,
                    self.ctx.K,
                    self.ctx.tau_list,
                    self.ctx.T,
                    individual.yijt,
                )
            self.decode_stats["greedy_calls"] += 1
            self.decode_stats["greedy_time_sec"] += time.time() - tic
        else:
            feasible, cost, decode, lowerbound = solve_procument_plan(
                self.ctx.n,
                self.ctx.I,
                self.ctx.beta,
                self.ctx.W,
                self.ctx.w0,
                self.ctx.e0,
                self.ctx.H,
                self.ctx.f,
                self.ctx.D,
                self.ctx.S,
                self.ctx.alpha,
                self.ctx.L,
                self.ctx.m,
                self.ctx.J,
                self.ctx.R,
                self.ctx.J1,
                self.ctx.J2,
                self.ctx.Jsp,
                self.ctx.z,
                self.ctx.l,
                self.ctx.u,
                self.ctx.r,
                self.ctx.Q,
                self.ctx.P,
                self.ctx.G,
                self.ctx.cjg,
                self.ctx.cgi,
                self.ctx.cij,
                self.ctx.t,
                self.ctx.K,
                self.ctx.tau_list,
                self.ctx.T,
                individual.yijt,
            )
            self.decode_stats["solver_calls"] += 1
            self.decode_stats["solver_time_sec"] += time.time() - tic

        individual.feasible = bool(feasible)
        individual.cost = cost if feasible else BIG_M
        individual.decode = decode
        individual.lowerbound = lowerbound
        if log_detail:
            self.logger.emit(
                "CHILD_EVAL",
                gen=gen,
                phase=self.phase_name(gen),
                decoder=decoder,
                child=child_label,
                feasible=individual.feasible,
                cost=round(individual.cost, 4),
                lower_bound=round(lowerbound, 4) if lowerbound is not None else "NA",
                decode_time_sec=round(time.time() - tic, 4),
                long_contract_repairs=repairs,
            )

    def update_fitness(self, population: list[Individual]):
        max_cost = max(ind.cost for ind in population)
        min_cost = min(ind.cost for ind in population)
        for ind in population:
            ind.fitness = (max_cost - ind.cost) / (max_cost - min_cost + 1e4)

    def tournament_selection(self, population: list[Individual], k: int = 3) -> Individual:
        contestants = np.random.choice(population, k)
        return min(contestants, key=lambda x: x.cost)

    def adaptive_pc(self, population: list[Individual], parent1: Individual, parent2: Individual) -> float:
        f_bar = max(parent1.fitness, parent2.fitness)
        f_ave = sum(ind.fitness for ind in population) / len(population)
        f_max = max(ind.fitness for ind in population)
        if f_bar >= f_ave:
            pc = self.k1 * (f_max - f_bar) / (f_max - f_ave + 1e-6)
        else:
            pc = self.k3
        return min(max(pc, 0.0), 1.0)

    def adaptive_pm(self, population: list[Individual], child: Individual) -> float:
        f_ave = sum(ind.fitness for ind in population) / len(population)
        f_max = max(ind.fitness for ind in population)
        if child.fitness >= f_ave:
            pm = self.k2 * (f_max - child.fitness) / (f_max - f_ave + 1e-6)
        else:
            pm = self.k4
        return min(max(pm, 0.0), 1.0)

    def crossover(self, parent1: Individual, parent2: Individual, pc: float) -> tuple[Individual, Individual]:
        child1 = Individual.empty(self.ctx)
        child2 = Individual.empty(self.ctx)
        for j in self.ctx.J:
            if np.random.rand() < pc:
                child1.yijt[:, j, :, :] = parent1.yijt[:, j, :, :].copy()
                child2.yijt[:, j, :, :] = parent2.yijt[:, j, :, :].copy()
            else:
                child1.yijt[:, j, :, :] = parent2.yijt[:, j, :, :].copy()
                child2.yijt[:, j, :, :] = parent1.yijt[:, j, :, :].copy()
        return child1, child2

    def mutate_random_flip(self, individual: Individual) -> int:
        changed = 0
        for i in self.ctx.I:
            for j in self.ctx.J:
                for tau in self.ctx.tau_list:
                    if np.random.rand() < 0.15:
                        individual.yijt[i, j, self.ctx.t, tau] = 1 - individual.yijt[i, j, self.ctx.t, tau]
                        changed += 1
        for i in self.ctx.I:
            for j in self.ctx.J:
                if self.ctx.alpha[i][j] < self.ctx.beta[i]:
                    for tau in self.ctx.tau_list:
                        individual.yijt[i, j, self.ctx.t, tau] = 0
        return changed

    def mutate_adaptive_transfer(self, individual: Individual) -> int:
        i = random.choice(self.ctx.I)
        tau = random.choice(self.ctx.tau_list)
        current_suppliers = [j for j in self.ctx.J if individual.yijt[i, j, self.ctx.t, tau] == 1]
        total_decisions = np.sum(individual.yijt)
        ave_supplier_num = max(1, int(total_decisions / (len(self.ctx.tau_list) * len(self.ctx.I))))

        if len(current_suppliers) >= ave_supplier_num:
            if len(current_suppliers) > 1:
                num_to_choose = random.randint(1, len(current_suppliers) - 1)
            else:
                num_to_choose = 1
            new_suppliers = random.sample(current_suppliers, num_to_choose)
        else:
            eligible = [j for j in self.ctx.J if self.ctx.alpha[i][j] >= self.ctx.beta[i]]
            if not eligible:
                return 0
            low = max(1, len(current_suppliers) - 1)
            high = min(ave_supplier_num, len(eligible))
            if low > high:
                low = high
            num_to_choose = random.randint(low, high)
            new_suppliers = random.sample(eligible, num_to_choose)

        changed = 0
        before = individual.yijt[i, :, self.ctx.t, tau].copy()
        for j in current_suppliers:
            individual.yijt[i, j, self.ctx.t, tau] = 0
        for j in new_suppliers:
            individual.yijt[i, j, self.ctx.t, tau] = 1
        changed = int(np.sum(before != individual.yijt[i, :, self.ctx.t, tau]))
        return changed

    def is_duplicate(self, new_individual: Individual, population: list[Individual]) -> bool:
        return any(np.array_equal(new_individual.yijt, existing.yijt) for existing in population)

    def run_periodic_local_search(self, population: list[Individual], gen: int):
        population.sort(key=lambda x: x.cost)
        for rank in range(min(self.elite_ls, len(population))):
            before = population[rank].cost
            local_search = TracedLocalSearch(
                self.ctx,
                population[rank],
                self.logger,
                gen,
                rank + 1,
                self.ls_max_iter,
            )
            improved_solution, _ = local_search.run()
            population[rank] = improved_solution
            if improved_solution.cost < before - 1e-6:
                self.logger.emit(
                    "ELITE_IMPROVED_BY_LS",
                    gen=gen,
                    elite_rank=rank + 1,
                    old_cost=round(before, 4),
                    new_cost=round(improved_solution.cost, 4),
                    improvement=round(before - improved_solution.cost, 4),
                    improvement_pct=round(pct(before - improved_solution.cost, before), 4),
                )

    def run(self) -> tuple[Individual, list[dict[str, Any]]]:
        start_time = time.time()
        self.logger.emit(
            "RUN_START",
            instance=self.ctx.instance_name,
            pop_size=self.pop_size,
            max_gen=self.max_gen,
            time_limit_sec=self.max_time,
            K_explore=self.k_explore,
            K_ls=self.k_ls,
            elite_ls=self.elite_ls,
            ls_max_iter=self.ls_max_iter,
            decoder_schedule="greedy_before_K_explore;solver_from_K_explore",
            local_search="periodic_three_neighborhood_elite_search",
        )
        population = self.initialize_population()
        best_individual = copy.deepcopy(min(population, key=lambda x: x.cost))
        best_lower_bound = min(
            (ind.lowerbound for ind in population if ind.lowerbound is not None),
            default=None,
        )
        history: list[dict[str, Any]] = []
        decoder_switched = False

        for gen in range(1, self.max_gen):
            if time.time() - start_time >= self.max_time:
                self.logger.emit(
                    "TERMINATE",
                    level="WARN",
                    gen=gen,
                    reason="time_limit",
                    elapsed_sec=round(time.time() - start_time, 4),
                    best_cost=round(best_individual.cost, 4),
                )
                break
            self.gen = gen
            if gen == self.k_explore and not decoder_switched:
                decoder_switched = True
                self.logger.emit(
                    "DECODE_SWITCH",
                    gen=gen,
                    from_decoder="greedy",
                    to_decoder="solver",
                    reason="gen >= K_explore; population enters exploitation stage",
                    best_cost=round(best_individual.cost, 4),
                )

            self.update_fitness(population)
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)
            pc = self.adaptive_pc(population, parent1, parent2)
            child1, child2 = self.crossover(parent1, parent2, pc)
            ra = self.ra_min + (self.ra_max - self.ra_min) * (1 - gen / self.max_gen)

            mutation_logs = []
            for label, child in [("child1", child1), ("child2", child2)]:
                pm = self.adaptive_pm(population, child)
                mutation_type = "none"
                changed = 0
                if np.random.rand() < pm:
                    if np.random.rand() < ra:
                        mutation_type = "adaptive_transfer"
                        changed = self.mutate_adaptive_transfer(child)
                    else:
                        mutation_type = "random_flip"
                        changed = self.mutate_random_flip(child)
                mutation_logs.append((label, pm, mutation_type, changed))

            if gen <= 5 or gen % 100 == 0:
                self.logger.emit(
                    "GA_OP",
                    gen=gen,
                    phase=self.phase_name(gen),
                    decoder=self.decoder_name(gen),
                    parent_costs=f"({fmt_money(parent1.cost)}, {fmt_money(parent2.cost)})",
                    pc=round(pc, 4),
                    RA=round(ra, 4),
                    mutation_child1=mutation_logs[0][2],
                    pm_child1=round(mutation_logs[0][1], 4),
                    changed_child1=mutation_logs[0][3],
                    mutation_child2=mutation_logs[1][2],
                    pm_child2=round(mutation_logs[1][1], 4),
                    changed_child2=mutation_logs[1][3],
                )

            self.evaluate(child1, gen=gen, log_detail=gen <= 5 or gen % 100 == 0, child_label="child1")
            self.evaluate(child2, gen=gen, log_detail=gen <= 5 or gen % 100 == 0, child_label="child2")

            retry_count = 0
            max_retry_count = 100
            while not (child1.feasible and child2.feasible) and retry_count < max_retry_count:
                retry_count += 1
                child1, child2 = self.crossover(parent1, parent2, pc)
                for child in (child1, child2):
                    pm = self.adaptive_pm(population, child)
                    if np.random.rand() < pm:
                        if np.random.rand() < ra:
                            self.mutate_adaptive_transfer(child)
                        else:
                            self.mutate_random_flip(child)
                self.evaluate(child1, gen=gen)
                self.evaluate(child2, gen=gen)

            if retry_count > 0:
                self.logger.emit(
                    "RETRY",
                    level="WARN",
                    gen=gen,
                    decoder=self.decoder_name(gen),
                    retry_count=retry_count,
                    child1_feasible=child1.feasible,
                    child2_feasible=child2.feasible,
                    reason="infeasible_child",
                    action="regenerate_by_crossover_and_mutation",
                )

            if not (child1.feasible and child2.feasible):
                self.logger.emit(
                    "GEN_SKIP",
                    level="WARN",
                    gen=gen,
                    reason="failed_to_generate_feasible_children",
                    retry_count=retry_count,
                )
                continue

            updated = False
            ls_executed = False
            if not self.is_duplicate(child1, population) and not self.is_duplicate(child2, population):
                population.sort(key=lambda x: x.cost)
                removed = population[-2:]
                if child1.cost < removed[0].cost and child2.cost < removed[0].cost:
                    population = population[:-2]
                    population.append(child1)
                    population.append(child2)
                    updated = True
                    self.logger.emit(
                        "POP_UPDATE",
                        gen=gen,
                        decoder=self.decoder_name(gen),
                        action="replace_worst_two",
                        child_costs=f"({fmt_money(child1.cost)}, {fmt_money(child2.cost)})",
                        removed_costs=f"({fmt_money(removed[0].cost)}, {fmt_money(removed[1].cost)})",
                    )

                    if gen % self.k_ls == 0:
                        self.run_periodic_local_search(population, gen)
                        ls_executed = True

            if gen % self.k_ls == 0 and not ls_executed:
                self.run_periodic_local_search(population, gen)
                ls_executed = True

            current_best = min(population, key=lambda x: x.cost)
            if current_best.cost < best_individual.cost - 1e-6:
                old_best = best_individual.cost
                best_individual = copy.deepcopy(current_best)
                self.logger.emit(
                    "BEST_UPDATE",
                    gen=gen,
                    source="local_search" if ls_executed else "offspring",
                    decoder=self.decoder_name(gen),
                    old_best=round(old_best, 4),
                    new_best=round(best_individual.cost, 4),
                    improvement=round(old_best - best_individual.cost, 4),
                    improvement_pct=round(pct(old_best - best_individual.cost, old_best), 4),
                )

            current_lbs = [ind.lowerbound for ind in population if ind.lowerbound is not None]
            if current_lbs:
                candidate_lb = min(current_lbs)
                if best_lower_bound is None or candidate_lb < best_lower_bound:
                    best_lower_bound = candidate_lb

            if gen <= 5 or gen % 25 == 0 or updated:
                avg_cost = sum(ind.cost for ind in population) / len(population)
                gap = pct(best_individual.cost - best_lower_bound, best_individual.cost) if best_lower_bound else None
                self.logger.emit(
                    "GEN_SUMMARY",
                    gen=gen,
                    phase=self.phase_name(gen),
                    decoder=self.decoder_name(gen),
                    updated_population=updated,
                    best_cost=round(best_individual.cost, 4),
                    lower_bound=round(best_lower_bound, 4) if best_lower_bound is not None else "NA",
                    gap_pct=round(gap, 4) if gap is not None else "NA",
                    avg_cost=round(avg_cost, 4),
                    feasible_count=sum(1 for ind in population if ind.feasible),
                )
                history.append(
                    {
                        "gen": gen,
                        "best_cost": best_individual.cost,
                        "lower_bound": best_lower_bound,
                        "avg_cost": avg_cost,
                        "feasible_count": sum(1 for ind in population if ind.feasible),
                    }
                )

        self.logger.emit(
            "RUN_END",
            best_cost=round(best_individual.cost, 4),
            lower_bound=round(best_lower_bound, 4) if best_lower_bound is not None else "NA",
            greedy_decode_calls=self.decode_stats["greedy_calls"],
            solver_decode_calls=self.decode_stats["solver_calls"],
            greedy_decode_time_sec=round(self.decode_stats["greedy_time_sec"], 4),
            solver_decode_time_sec=round(self.decode_stats["solver_time_sec"], 4),
        )
        return best_individual, history


def build_context(args: argparse.Namespace) -> InstanceContext:
    base_filename = args.base_file or os.path.join(REPLICATION_ROOT, "data", "raw_excel_optional", "base_data.xlsx")
    supplier_filename = args.supplier_file or os.path.join(REPLICATION_ROOT, "data", "raw_excel_optional", "supplier_data.xlsx")
    transport_filename = args.transport_file or os.path.join(REPLICATION_ROOT, "data", "raw_excel_optional", "transport_data.xlsx")

    n, I_all, beta, W, w0, e0, H, f, D, S, alpha, L = read_xlsx_file_base(base_filename)
    m, J_all, R, J1, J2, Jsp, z, l, u, r, Q, P = read_xlsx_file_supplier(supplier_filename)
    G, cjg, cgi, cij = read_xlsx_file_transport(transport_filename)

    t = args.t
    K = args.K
    T = [t]
    tau_list = list(range(t, t + K + 1))
    I = list(range(args.num_bases)) if args.num_bases is not None else I_all
    J = list(range(args.num_suppliers)) if args.num_suppliers is not None else J_all
    instance_name = args.instance_name or f"case_{len(I)}x{len(J)}"
    return InstanceContext(
        n=n,
        I=I,
        beta=beta,
        W=W,
        w0=w0,
        e0=e0,
        H=H,
        f=f,
        D=D,
        S=S,
        alpha=alpha,
        L=L,
        m=m,
        J=J,
        R=R,
        J1=J1,
        J2=J2,
        Jsp=Jsp,
        z=z,
        l=l,
        u=u,
        r=r,
        Q=Q,
        P=P,
        G=G,
        cjg=cjg,
        cgi=cgi,
        cij=cij,
        t=t,
        K=K,
        T=T,
        tau_list=tau_list,
        instance_name=instance_name,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paper-style GA-hybrid algorithm and generate academic event logs."
    )
    parser.add_argument("--base-file", default=None)
    parser.add_argument("--supplier-file", default=None)
    parser.add_argument("--transport-file", default=None)
    parser.add_argument("--instance-name", default=None)
    parser.add_argument("--num-bases", type=int, default=4)
    parser.add_argument("--num-suppliers", type=int, default=35)
    parser.add_argument("--t", type=int, default=0)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--pop-size", type=int, default=200)
    parser.add_argument("--max-gen", type=int, default=2000)
    parser.add_argument("--max-time", type=float, default=3600)
    parser.add_argument("--k-explore", type=int, default=None)
    parser.add_argument("--k-ls", type=int, default=50)
    parser.add_argument("--elite-ls", type=int, default=10)
    parser.add_argument("--ls-max-iter", type=int, default=5)
    parser.add_argument("--mut-prob", type=float, default=0.2)
    parser.add_argument("--ra-max", type=float, default=0.8)
    parser.add_argument("--ra-min", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(REPLICATION_ROOT, "logs", "generated"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.k_explore is None:
        args.k_explore = args.max_gen // 2

    ctx = build_context(args)
    run_id = (
        f"{ctx.instance_name}_seed{args.seed}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    logger = AcademicTraceLogger(args.output_dir, run_id)
    try:
        ga = AcademicTraceGA(
            ctx=ctx,
            logger=logger,
            pop_size=args.pop_size,
            max_gen=args.max_gen,
            max_time=args.max_time,
            k_explore=args.k_explore,
            k_ls=args.k_ls,
            elite_ls=args.elite_ls,
            ls_max_iter=args.ls_max_iter,
            mut_prob=args.mut_prob,
            ra_max=args.ra_max,
            ra_min=args.ra_min,
        )
        best, _ = ga.run()
        logger.emit(
            "ARTIFACT",
            text_log=logger.text_path,
            jsonl_log=logger.jsonl_path,
            final_best_cost=round(best.cost, 4),
            feasible=best.feasible,
        )
    finally:
        logger.close()


if __name__ == "__main__":
    main()
