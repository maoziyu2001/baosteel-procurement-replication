"""Procurement result standardization and analysis exports.

This module is deliberately read-only with respect to the algorithms: it takes
the final solution produced by GA-greedy or manual heuristics, recomputes
descriptive statistics using the existing objective components, and writes CSV
files for paper tables and figures.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
EPS = 1e-12


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if abs(float(denominator)) > EPS else np.nan


def _get(data: dict[str, Any], key: str, default: Any = None) -> Any:
    if key not in data:
        LOGGER.warning("Missing instance field %s; related outputs will contain NaN.", key)
    return data.get(key, default)


def _as_list(values: Any, fallback_len: int | None = None) -> list[int]:
    if values is None:
        return list(range(fallback_len or 0))
    return [int(x) for x in list(values)]


def _normalise_x(raw_solution: Any, instance_data: dict[str, Any]) -> np.ndarray:
    """Return purchase quantities as base x supplier x period.

    GA-greedy stores x as base x supplier x T x tau. Manual stores it as
    base x supplier x period. A dict with key ``x`` or ``purchase_qty`` is also
    accepted for batch scripts.
    """
    if isinstance(raw_solution, dict):
        x = raw_solution.get("x", raw_solution.get("purchase_qty", raw_solution.get("procurement")))
    elif hasattr(raw_solution, "decode"):
        x = raw_solution.decode
    else:
        x = raw_solution
    if x is None:
        raise ValueError("raw_solution does not contain purchase quantities")
    x = np.asarray(x, dtype=float)
    if x.ndim == 4:
        # Existing rolling-horizon code has a single current decision period.
        x = x[:, :, 0, :]
    if x.ndim != 3:
        raise ValueError(f"purchase quantity array must be 3D or 4D, got shape={x.shape}")
    return x


def _solution_attr(raw_solution: Any, key: str, default: Any = None) -> Any:
    if isinstance(raw_solution, dict):
        return raw_solution.get(key, default)
    return getattr(raw_solution, key, default)


def standardize_solution(raw_solution: Any, instance_data: dict[str, Any],
                         algorithm_name: str, instance_name: str) -> dict[str, Any]:
    """Convert algorithm-specific final output into one analysis structure."""
    x = _normalise_x(raw_solution, instance_data)
    I = _as_list(instance_data.get("I"), x.shape[0])
    J = _as_list(instance_data.get("J"), x.shape[1])
    periods = _as_list(instance_data.get("tau_list", instance_data.get("periods")), x.shape[2])
    if len(I) != x.shape[0]:
        I = list(range(x.shape[0]))
    if len(J) != x.shape[1]:
        J = list(range(x.shape[1]))
    if len(periods) != x.shape[2]:
        periods = list(range(x.shape[2]))

    sol = {
        "instance_name": instance_name,
        "algorithm_name": algorithm_name,
        "x": x,
        "I": I,
        "J": J,
        "periods": periods,
        "raw_solution": raw_solution,
        "runtime_seconds": _solution_attr(raw_solution, "runtime_seconds", instance_data.get("runtime_seconds", np.nan)),
        "feasibility_status": _solution_attr(raw_solution, "feasible", instance_data.get("feasibility_status", np.nan)),
        "reported_total_cost": _solution_attr(raw_solution, "cost", instance_data.get("total_cost", np.nan)),
        "data": instance_data,
    }
    _derive_core_arrays(sol)
    return sol


def _derive_core_arrays(solution: dict[str, Any]) -> None:
    """Compute prices, inventory states, and objective component arrays."""
    data = solution["data"]
    x = solution["x"]
    I, J, periods = solution["I"], solution["J"], solution["periods"]
    z = _get(data, "z", np.full(max(J) + 1, np.nan))
    P = _get(data, "P", np.full((max(J) + 1, max(periods) + 1), np.nan))
    r = _get(data, "r", np.zeros(max(J) + 1))
    Q = _get(data, "Q", np.zeros(max(J) + 1))
    cij = _get(data, "cij", np.full((max(I) + 1, max(J) + 1), np.nan))
    D = _get(data, "D", np.zeros((max(I) + 1, max(periods) + 1)))
    w0 = _get(data, "w0", np.zeros(max(I) + 1))
    e0 = _get(data, "e0", np.zeros(max(I) + 1))
    f = _get(data, "f", np.zeros(max(I) + 1))
    R = _get(data, "R", np.zeros(max(J) + 1))

    n_i, n_j, n_p = x.shape
    q_supplier_period = x.sum(axis=0)
    actual_price = np.zeros((n_j, n_p), dtype=float)
    for jp, j in enumerate(J):
        for pp, period in enumerate(periods):
            q = q_supplier_period[jp, pp]
            # If the solver did not save dynamic prices, reproduce the existing
            # piecewise discount rule: P[j,t] - r[j] * min(q, Q[j]).
            actual_price[jp, pp] = P[j][period] - r[j] * min(q, Q[j])

    beginning_w = np.zeros((n_i, n_p))
    beginning_e = np.zeros((n_i, n_p))
    ending_w = np.zeros((n_i, n_p))
    ending_e = np.zeros((n_i, n_p))
    for ip, i in enumerate(I):
        for pp, period in enumerate(periods):
            purchase_qty = x[ip, :, pp].sum()
            purchase_iron = sum(x[ip, jp, pp] * z[j] for jp, j in enumerate(J))
            if pp == 0:
                beginning_w[ip, pp] = w0[i]
                beginning_e[ip, pp] = e0[i]
            else:
                beginning_w[ip, pp] = ending_w[ip, pp - 1]
                beginning_e[ip, pp] = ending_e[ip, pp - 1]
            available_iron = beginning_e[ip, pp] + purchase_iron
            ending_e[ip, pp] = available_iron - D[i][period]
            denominator = beginning_w[ip, pp] + purchase_qty
            ending_w[ip, pp] = (1 - safe_div(D[i][period], available_iron)) * denominator if abs(available_iron) > EPS else np.nan

    purchase_cost = np.zeros((n_i, n_j, n_p))
    transport_cost = np.zeros((n_i, n_j, n_p))
    for ip, i in enumerate(I):
        for jp, j in enumerate(J):
            for pp, _period in enumerate(periods):
                purchase_cost[ip, jp, pp] = x[ip, jp, pp] * actual_price[jp, pp]
                transport_cost[ip, jp, pp] = x[ip, jp, pp] * cij[i][j]

    inventory_cost_by_base = np.zeros(n_i)
    inventory_cost_by_base_period = np.zeros((n_i, n_p))
    for ip, i in enumerate(I):
        period_terms = np.array([
            ending_w[ip, pp] + 0.5 * x[ip, :, pp].sum()
            for pp in range(n_p)
        ])
        adjustment = 0.5 * w0[i] - 0.5 * ending_w[ip, -1]
        inventory_cost_by_base[ip] = f[i] * (np.nansum(period_terms) + adjustment)
        if n_p:
            inventory_cost_by_base_period[ip, :] = f[i] * period_terms
            inventory_cost_by_base_period[ip, -1] += f[i] * adjustment

    selected_supplier = q_supplier_period.sum(axis=1) > EPS
    fixed_supplier_cost = np.array([R[j] if selected_supplier[jp] else 0.0 for jp, j in enumerate(J)])

    solution.update({
        "actual_price": actual_price,
        "beginning_inventory_weight": beginning_w,
        "beginning_inventory_iron": beginning_e,
        "ending_inventory_weight": ending_w,
        "ending_inventory_iron": ending_e,
        "purchase_cost_array": purchase_cost,
        "transport_cost_array": transport_cost,
        "inventory_cost_by_base": inventory_cost_by_base,
        "inventory_cost_by_base_period": inventory_cost_by_base_period,
        "fixed_supplier_cost_array": fixed_supplier_cost,
    })


def build_purchase_detail(solution: dict[str, Any]) -> pd.DataFrame:
    rows = []
    data = solution["data"]
    x = solution["x"]
    I, J, periods = solution["I"], solution["J"], solution["periods"]
    z, cij, Jsp = data["z"], data["cij"], set(int(v) for v in data.get("Jsp", []))
    for ip, i in enumerate(I):
        for jp, j in enumerate(J):
            for pp, period in enumerate(periods):
                qty = x[ip, jp, pp]
                price = solution["actual_price"][jp, pp]
                trans = cij[i][j]
                delivered = price + trans
                iron_qty = qty * z[j]
                rows.append({
                    "instance_name": solution["instance_name"],
                    "algorithm_name": solution["algorithm_name"],
                    "period": period,
                    "base_id": i,
                    "supplier_id": j,
                    "purchase_qty": qty,
                    "supplier_grade": z[j],
                    "purchase_price": price,
                    "transport_cost_per_ton": trans,
                    "delivered_unit_cost": delivered,
                    "iron_qty": iron_qty,
                    "unit_iron_cost": safe_div(delivered, z[j]),
                    "purchase_cost": qty * price,
                    "transport_cost": qty * trans,
                    "is_selected": int(qty > EPS),
                    "is_long_contract_supplier": int(j in Jsp),
                })
    return pd.DataFrame(rows)


def compute_cost_summary(solution: dict[str, Any]) -> pd.DataFrame:
    detail = build_purchase_detail(solution)
    purchase_cost = detail["purchase_cost"].sum()
    transport_cost = detail["transport_cost"].sum()
    inventory_cost = float(np.nansum(solution["inventory_cost_by_base"]))
    fixed_cost = float(np.nansum(solution["fixed_supplier_cost_array"]))
    total_cost = purchase_cost + transport_cost + inventory_cost + fixed_cost
    total_qty = detail["purchase_qty"].sum()
    total_iron = detail["iron_qty"].sum()
    return pd.DataFrame([{
        "instance_name": solution["instance_name"],
        "algorithm_name": solution["algorithm_name"],
        "purchase_cost": purchase_cost,
        "transport_cost": transport_cost,
        "inventory_cost": inventory_cost,
        "fixed_supplier_cost": fixed_cost,
        "total_cost": total_cost,
        "reported_total_cost": solution.get("reported_total_cost", np.nan),
        "total_purchase_qty": total_qty,
        "total_iron_qty": total_iron,
        "average_purchase_price_per_ton": safe_div(purchase_cost, total_qty),
        "average_delivered_cost_per_ton": safe_div(purchase_cost + transport_cost, total_qty),
        "average_unit_iron_cost": safe_div(purchase_cost + transport_cost, total_iron),
        "runtime_seconds": solution.get("runtime_seconds", np.nan),
        "feasibility_status": solution.get("feasibility_status", np.nan),
    }])


def compute_supplier_summary(solution: dict[str, Any]) -> pd.DataFrame:
    detail = build_purchase_detail(solution)
    data = solution["data"]
    Jsp = set(int(v) for v in data.get("Jsp", []))
    total_qty_all = detail["purchase_qty"].sum()
    total_iron_all = detail["iron_qty"].sum()
    rows = []
    for j, group in detail.groupby("supplier_id", sort=True):
        active = group[group["purchase_qty"] > EPS]
        total_qty = group["purchase_qty"].sum()
        total_iron = group["iron_qty"].sum()
        weighted_trans = safe_div(group["transport_cost"].sum(), total_qty)
        delivered_cost = group["purchase_cost"].sum() + group["transport_cost"].sum()
        min_qty = len(solution["periods"]) * data["l"][j] if j in Jsp else np.nan
        rows.append({
            "instance_name": solution["instance_name"],
            "algorithm_name": solution["algorithm_name"],
            "supplier_id": j,
            "total_purchase_qty": total_qty,
            "total_iron_qty": total_iron,
            "purchase_share": safe_div(total_qty, total_qty_all),
            "iron_share": safe_div(total_iron, total_iron_all),
            "served_base_count": active["base_id"].nunique(),
            "served_period_count": active["period"].nunique(),
            "average_purchase_qty_per_active_period": safe_div(total_qty, active["period"].nunique()),
            "supplier_grade": data["z"][j],
            "average_price": safe_div(group["purchase_cost"].sum(), total_qty),
            "average_transport_cost_weighted": weighted_trans,
            "average_unit_iron_cost_weighted": safe_div(delivered_cost, total_iron),
            "is_long_contract_supplier": int(j in Jsp),
            "long_contract_min_qty": min_qty,
            "long_contract_surplus": total_qty - min_qty if j in Jsp else np.nan,
            "reached_discount_threshold": int(total_qty >= data["Q"][j]) if not pd.isna(data["Q"][j]) else 0,
        })
    return pd.DataFrame(rows)


def compute_supplier_structure_metrics(solution: dict[str, Any]) -> pd.DataFrame:
    supplier = compute_supplier_summary(solution)
    selected = supplier[supplier["total_purchase_qty"] > EPS].copy()
    total_qty = supplier["total_purchase_qty"].sum()
    shares = selected["purchase_share"].sort_values(ascending=False).to_numpy()
    multi = selected[selected["served_base_count"] >= 2]
    single = selected[selected["served_base_count"] == 1]
    long_qty = selected.loc[selected["is_long_contract_supplier"] == 1, "total_purchase_qty"].sum()
    return pd.DataFrame([{
        "instance_name": solution["instance_name"],
        "algorithm_name": solution["algorithm_name"],
        "selected_supplier_count": len(selected),
        "avg_purchase_qty_per_selected_supplier": safe_div(total_qty, len(selected)),
        "top1_supplier_share": shares[:1].sum() if len(shares) else np.nan,
        "top3_supplier_share": shares[:3].sum() if len(shares) else np.nan,
        "top5_supplier_share": shares[:5].sum() if len(shares) else np.nan,
        "herfindahl_index": float(np.sum(shares ** 2)) if len(shares) else np.nan,
        "multi_base_supplier_count": len(multi),
        "multi_base_supplier_purchase_share": safe_div(multi["total_purchase_qty"].sum(), total_qty),
        "single_base_supplier_count": len(single),
        "long_contract_supplier_purchase_share": safe_div(long_qty, total_qty),
        "non_long_contract_supplier_purchase_share": safe_div(total_qty - long_qty, total_qty),
    }])


def compute_base_supplier_matrix(solution: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = build_purchase_detail(solution)
    grouped = detail.groupby(["base_id", "supplier_id"], as_index=False).agg(
        total_purchase_qty=("purchase_qty", "sum"),
        total_iron_qty=("iron_qty", "sum"),
    )
    base_total = grouped.groupby("base_id")["total_purchase_qty"].transform("sum")
    grouped.insert(0, "algorithm_name", solution["algorithm_name"])
    grouped.insert(0, "instance_name", solution["instance_name"])
    grouped["purchase_share_in_base"] = grouped["total_purchase_qty"] / base_total.replace(0, np.nan)
    grouped["is_selected"] = (grouped["total_purchase_qty"] > EPS).astype(int)
    wide = grouped.pivot(index="base_id", columns="supplier_id", values="total_purchase_qty").fillna(0.0)
    wide.insert(0, "algorithm_name", solution["algorithm_name"])
    wide.insert(0, "instance_name", solution["instance_name"])
    return grouped, wide.reset_index()


def compute_cross_base_metrics(solution: dict[str, Any]) -> pd.DataFrame:
    supplier = compute_supplier_summary(solution)
    selected = supplier[supplier["total_purchase_qty"] > EPS]
    multi = selected[selected["served_base_count"] >= 2]
    total_qty = supplier["total_purchase_qty"].sum()
    shared = [str(int(v)) for v in multi["supplier_id"].tolist()]
    base_supplier_long, _ = compute_base_supplier_matrix(solution)
    per_base_counts = base_supplier_long[base_supplier_long["is_selected"] == 1].groupby("base_id")["supplier_id"].nunique()
    return pd.DataFrame([{
        "instance_name": solution["instance_name"],
        "algorithm_name": solution["algorithm_name"],
        "base_count": len(solution["I"]),
        "supplier_count": len(solution["J"]),
        "selected_supplier_count": len(selected),
        "multi_base_supplier_count": len(multi),
        "multi_base_supplier_ratio": safe_div(len(multi), len(selected)),
        "multi_base_purchase_qty": multi["total_purchase_qty"].sum(),
        "multi_base_purchase_share": safe_div(multi["total_purchase_qty"].sum(), total_qty),
        "avg_supplier_count_per_base": per_base_counts.mean() if len(per_base_counts) else 0,
        "avg_base_count_per_selected_supplier": selected["served_base_count"].mean() if len(selected) else np.nan,
        "max_base_count_per_supplier": selected["served_base_count"].max() if len(selected) else 0,
        "shared_supplier_list": ";".join(shared),
    }])


def compute_base_summary(solution: dict[str, Any]) -> pd.DataFrame:
    detail = build_purchase_detail(solution)
    data = solution["data"]
    high_grade_threshold, low_cost_threshold = _thresholds(solution)
    rows = []
    for ip, i in enumerate(solution["I"]):
        group = detail[detail["base_id"] == i]
        active = group[group["purchase_qty"] > EPS]
        total_qty = group["purchase_qty"].sum()
        total_iron = group["iron_qty"].sum()
        delivered = group["purchase_cost"].sum() + group["transport_cost"].sum()
        long_qty = active.loc[active["is_long_contract_supplier"] == 1, "purchase_qty"].sum()
        rows.append({
            "instance_name": solution["instance_name"],
            "algorithm_name": solution["algorithm_name"],
            "base_id": i,
            "total_purchase_qty": total_qty,
            "total_iron_qty": total_iron,
            "production_demand_total": sum(data["D"][i][p] for p in solution["periods"]),
            "selected_supplier_count": active["supplier_id"].nunique(),
            "long_contract_purchase_qty": long_qty,
            "long_contract_purchase_share": safe_div(long_qty, total_qty),
            "weighted_average_grade": safe_div(total_iron, total_qty),
            "average_purchase_price_per_ton": safe_div(group["purchase_cost"].sum(), total_qty),
            "average_transport_cost_per_ton": safe_div(group["transport_cost"].sum(), total_qty),
            "average_delivered_cost_per_ton": safe_div(delivered, total_qty),
            "average_unit_iron_cost": safe_div(delivered, total_iron),
            "high_grade_purchase_share": safe_div(group.loc[group["supplier_grade"] > high_grade_threshold, "purchase_qty"].sum(), total_qty),
            "low_unit_iron_cost_purchase_share": safe_div(group.loc[group["unit_iron_cost"] < low_cost_threshold, "purchase_qty"].sum(), total_qty),
            "ending_inventory_weight": solution["ending_inventory_weight"][ip, -1],
            "ending_inventory_iron": solution["ending_inventory_iron"][ip, -1],
            "average_inventory_weight": np.nanmean(solution["ending_inventory_weight"][ip, :]),
            "average_inventory_iron": np.nanmean(solution["ending_inventory_iron"][ip, :]),
            "safety_stock_redundancy_avg": np.nanmean([
                safe_div(solution["ending_inventory_iron"][ip, pp] - data["S"][i][period], data["S"][i][period])
                for pp, period in enumerate(solution["periods"])
            ]),
            "capacity_utilization_avg": safe_div(np.nanmean(solution["ending_inventory_weight"][ip, :]), data["H"][i]),
        })
    return pd.DataFrame(rows)


def compute_inventory_period_summary(solution: dict[str, Any]) -> pd.DataFrame:
    data = solution["data"]
    x = solution["x"]
    rows = []
    for ip, i in enumerate(solution["I"]):
        for pp, period in enumerate(solution["periods"]):
            purchase_qty = x[ip, :, pp].sum()
            purchase_iron = sum(x[ip, jp, pp] * data["z"][j] for jp, j in enumerate(solution["J"]))
            ending_iron = solution["ending_inventory_iron"][ip, pp]
            ending_weight = solution["ending_inventory_weight"][ip, pp]
            rows.append({
                "instance_name": solution["instance_name"],
                "algorithm_name": solution["algorithm_name"],
                "base_id": i,
                "period": period,
                "beginning_inventory_weight": solution["beginning_inventory_weight"][ip, pp],
                "beginning_inventory_iron": solution["beginning_inventory_iron"][ip, pp],
                "purchase_qty": purchase_qty,
                "purchase_iron_qty": purchase_iron,
                "production_demand": data["D"][i][period],
                "ending_inventory_weight": ending_weight,
                "ending_inventory_iron": ending_iron,
                "safety_stock": data["S"][i][period],
                "safety_stock_redundancy": safe_div(ending_iron - data["S"][i][period], data["S"][i][period]),
                "capacity": data["H"][i],
                "capacity_utilization": safe_div(ending_weight, data["H"][i]),
                "inventory_cost": solution["inventory_cost_by_base_period"][ip, pp],
                "weighted_average_grade_after_purchase": safe_div(solution["beginning_inventory_iron"][ip, pp] + purchase_iron, solution["beginning_inventory_weight"][ip, pp] + purchase_qty),
            })
    return pd.DataFrame(rows)


def compute_inventory_strategy_metrics(solution: dict[str, Any]) -> pd.DataFrame:
    inv = compute_inventory_period_summary(solution)
    cost = compute_cost_summary(solution).iloc[0]
    # Descriptive early-purchase proxy: a period is flagged when purchase iron
    # exceeds current demand, the next period purchases less, and ending iron
    # inventory increases. This is not a feasibility or optimality judgment.
    possible_early = 0.0
    for base_id, group in inv.sort_values("period").groupby("base_id"):
        rows = group.to_dict("records")
        for current, nxt in zip(rows, rows[1:]):
            if (current["purchase_iron_qty"] > current["production_demand"]
                    and nxt["purchase_qty"] < current["purchase_qty"]
                    and current["ending_inventory_iron"] > current["beginning_inventory_iron"]):
                possible_early += max(0.0, current["purchase_qty"] - nxt["purchase_qty"])
    return pd.DataFrame([{
        "instance_name": solution["instance_name"],
        "algorithm_name": solution["algorithm_name"],
        "avg_ending_inventory_weight": inv["ending_inventory_weight"].mean(),
        "avg_ending_inventory_iron": inv["ending_inventory_iron"].mean(),
        "avg_safety_stock_redundancy": inv["safety_stock_redundancy"].mean(),
        "max_safety_stock_redundancy": inv["safety_stock_redundancy"].max(),
        "min_safety_stock_redundancy": inv["safety_stock_redundancy"].min(),
        "avg_capacity_utilization": inv["capacity_utilization"].mean(),
        "max_capacity_utilization": inv["capacity_utilization"].max(),
        "total_inventory_cost": cost["inventory_cost"],
        "inventory_cost_share": safe_div(cost["inventory_cost"], cost["total_cost"]),
        "possible_early_purchase_qty": possible_early,
    }])


def _thresholds(solution: dict[str, Any]) -> tuple[float, float]:
    detail = build_purchase_detail(solution)
    return float(np.nanmean([solution["data"]["z"][j] for j in solution["J"]])), float(np.nanmean(detail["unit_iron_cost"]))


def compute_grade_cost_summary(solution: dict[str, Any]) -> pd.DataFrame:
    detail = build_purchase_detail(solution)
    data = solution["data"]
    high_grade_threshold, unit_iron_cost_threshold = _thresholds(solution)
    rows = []
    groups = [("ALL", "ALL", detail)]
    groups.extend((i, "ALL", g) for i, g in detail.groupby("base_id"))
    groups.extend(((i, p, g) for (i, p), g in detail.groupby(["base_id", "period"])))
    for base_id, period, group in groups:
        total_qty = group["purchase_qty"].sum()
        total_iron = group["iron_qty"].sum()
        delivered = group["purchase_cost"].sum() + group["transport_cost"].sum()
        min_required = data["W"][base_id] if base_id != "ALL" else np.nan
        rows.append({
            "instance_name": solution["instance_name"],
            "algorithm_name": solution["algorithm_name"],
            "base_id": base_id,
            "period": period,
            "total_purchase_qty": total_qty,
            "total_iron_qty": total_iron,
            "weighted_average_grade": safe_div(total_iron, total_qty),
            "min_required_grade": min_required,
            "grade_surplus": safe_div(total_iron, total_qty) - min_required if base_id != "ALL" else np.nan,
            "average_unit_iron_cost": safe_div(delivered, total_iron),
            "high_grade_purchase_share": safe_div(group.loc[group["supplier_grade"] > high_grade_threshold, "purchase_qty"].sum(), total_qty),
            "low_grade_purchase_share": safe_div(group.loc[group["supplier_grade"] <= high_grade_threshold, "purchase_qty"].sum(), total_qty),
            "low_unit_iron_cost_purchase_share": safe_div(group.loc[group["unit_iron_cost"] < unit_iron_cost_threshold, "purchase_qty"].sum(), total_qty),
            "high_unit_iron_cost_purchase_share": safe_div(group.loc[group["unit_iron_cost"] >= unit_iron_cost_threshold, "purchase_qty"].sum(), total_qty),
            "high_grade_threshold": high_grade_threshold,
            "unit_iron_cost_threshold": unit_iron_cost_threshold,
        })
    return pd.DataFrame(rows)


def build_algorithm_comparison(manual_stats: dict[str, pd.DataFrame],
                               ga_stats: dict[str, pd.DataFrame]) -> pd.DataFrame:
    def scalar(stats: dict[str, pd.DataFrame], table: str, column: str) -> float:
        df = stats.get(table, pd.DataFrame())
        return df[column].iloc[0] if column in df.columns and not df.empty else np.nan

    metric_map = {
        "total_cost": ("cost_summary", "total_cost"),
        "purchase_cost": ("cost_summary", "purchase_cost"),
        "transport_cost": ("cost_summary", "transport_cost"),
        "inventory_cost": ("cost_summary", "inventory_cost"),
        "fixed_supplier_cost": ("cost_summary", "fixed_supplier_cost"),
        "selected_supplier_count": ("supplier_structure_metrics", "selected_supplier_count"),
        "avg_supplier_count_per_base": ("cross_base_metrics", "avg_supplier_count_per_base"),
        "top5_supplier_share": ("supplier_structure_metrics", "top5_supplier_share"),
        "herfindahl_index": ("supplier_structure_metrics", "herfindahl_index"),
        "multi_base_supplier_count": ("cross_base_metrics", "multi_base_supplier_count"),
        "multi_base_purchase_share": ("cross_base_metrics", "multi_base_purchase_share"),
        "weighted_average_grade": ("grade_cost_summary", "weighted_average_grade"),
        "average_unit_iron_cost": ("cost_summary", "average_unit_iron_cost"),
        "avg_safety_stock_redundancy": ("inventory_strategy_metrics", "avg_safety_stock_redundancy"),
        "avg_capacity_utilization": ("inventory_strategy_metrics", "avg_capacity_utilization"),
        "long_contract_supplier_purchase_share": ("supplier_structure_metrics", "long_contract_supplier_purchase_share"),
    }
    rows = []
    for metric, (table, column) in metric_map.items():
        manual_value = scalar(manual_stats, table, column)
        ga_value = scalar(ga_stats, table, column)
        rows.append({
            "metric_name": metric,
            "manual_value": manual_value,
            "ga_greedy_value": ga_value,
            "absolute_change": ga_value - manual_value if pd.notna(manual_value) and pd.notna(ga_value) else np.nan,
            "relative_change": safe_div(ga_value - manual_value, manual_value) if pd.notna(manual_value) and pd.notna(ga_value) else np.nan,
        })
    return pd.DataFrame(rows)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.6f")


def _write_or_update_by_algorithm(df: pd.DataFrame, path: Path) -> None:
    """Update shared instance-level files without duplicating an algorithm row."""
    if path.exists():
        old = pd.read_csv(path, encoding="utf-8-sig")
        if "algorithm_name" in old.columns and "algorithm_name" in df.columns:
            old = old[~old["algorithm_name"].isin(df["algorithm_name"].unique())]
        df = pd.concat([old, df], ignore_index=True)
    _write_csv(df, path)


def _read_algorithm_row(path: Path, algorithm: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "algorithm_name" in df.columns:
        df = df[df["algorithm_name"] == algorithm]
    return df.reset_index(drop=True)


def _try_export_existing_algorithm_comparison(instance: str, output_dir: Path) -> None:
    """Create comparison CSV automatically once both manual and GA outputs exist."""
    manual_stats = {
        "cost_summary": _read_algorithm_row(output_dir / f"{instance}_cost_summary.csv", "manual"),
        "supplier_structure_metrics": _read_algorithm_row(output_dir / f"{instance}_manual_supplier_structure_metrics.csv", "manual"),
        "cross_base_metrics": _read_algorithm_row(output_dir / f"{instance}_manual_cross_base_metrics.csv", "manual"),
        "grade_cost_summary": _read_algorithm_row(output_dir / f"{instance}_manual_grade_cost_summary.csv", "manual"),
        "inventory_strategy_metrics": _read_algorithm_row(output_dir / f"{instance}_manual_inventory_strategy_metrics.csv", "manual"),
    }
    ga_stats = {
        "cost_summary": _read_algorithm_row(output_dir / f"{instance}_cost_summary.csv", "ga_greedy"),
        "supplier_structure_metrics": _read_algorithm_row(output_dir / f"{instance}_ga_greedy_supplier_structure_metrics.csv", "ga_greedy"),
        "cross_base_metrics": _read_algorithm_row(output_dir / f"{instance}_ga_greedy_cross_base_metrics.csv", "ga_greedy"),
        "grade_cost_summary": _read_algorithm_row(output_dir / f"{instance}_ga_greedy_grade_cost_summary.csv", "ga_greedy"),
        "inventory_strategy_metrics": _read_algorithm_row(output_dir / f"{instance}_ga_greedy_inventory_strategy_metrics.csv", "ga_greedy"),
    }
    if all(not df.empty for df in manual_stats.values()) and all(not df.empty for df in ga_stats.values()):
        export_algorithm_comparison(instance, manual_stats, ga_stats, output_dir)


def export_all_analysis_outputs(solution: dict[str, Any], output_dir: str | Path = "outputs") -> dict[str, pd.DataFrame]:
    """Write all paper-analysis CSV outputs for one standardized solution."""
    output_dir = Path(output_dir)
    instance = solution["instance_name"]
    algorithm = solution["algorithm_name"]
    prefix = f"{instance}_{algorithm}"

    detail = build_purchase_detail(solution)
    cost_summary = compute_cost_summary(solution)
    supplier_summary = compute_supplier_summary(solution)
    supplier_structure = compute_supplier_structure_metrics(solution)
    base_supplier_long, base_supplier_wide = compute_base_supplier_matrix(solution)
    cross_base = compute_cross_base_metrics(solution)
    base_summary = compute_base_summary(solution)
    inventory_period = compute_inventory_period_summary(solution)
    inventory_strategy = compute_inventory_strategy_metrics(solution)
    grade_cost = compute_grade_cost_summary(solution)

    supplier_pareto = supplier_summary.sort_values("total_purchase_qty", ascending=False).copy()
    supplier_pareto["rank"] = np.arange(1, len(supplier_pareto) + 1)
    supplier_pareto["cumulative_purchase_share"] = supplier_pareto["purchase_share"].cumsum()
    supplier_pareto = supplier_pareto[[
        "instance_name", "algorithm_name", "supplier_id", "total_purchase_qty",
        "purchase_share", "cumulative_purchase_share", "rank",
    ]]

    heatmap = base_supplier_long[["instance_name", "algorithm_name", "base_id", "supplier_id", "total_purchase_qty"]].copy()
    inventory_line = inventory_period[[
        "instance_name", "algorithm_name", "base_id", "period", "ending_inventory_iron",
        "safety_stock", "ending_inventory_weight", "capacity",
    ]].copy()
    cost_bar = cost_summary.melt(
        id_vars=["instance_name", "algorithm_name"],
        value_vars=["purchase_cost", "transport_cost", "inventory_cost", "fixed_supplier_cost"],
        var_name="cost_component",
        value_name="cost_value",
    )

    outputs = {
        "purchase_detail": detail,
        "cost_summary": cost_summary,
        "supplier_summary": supplier_summary,
        "supplier_structure_metrics": supplier_structure,
        "base_supplier_matrix_long": base_supplier_long,
        "base_supplier_matrix_wide": base_supplier_wide,
        "cross_base_metrics": cross_base,
        "base_summary": base_summary,
        "inventory_period_summary": inventory_period,
        "inventory_strategy_metrics": inventory_strategy,
        "grade_cost_summary": grade_cost,
        "supplier_pareto_data": supplier_pareto,
        "heatmap_base_supplier": heatmap,
        "inventory_line_data": inventory_line,
        "cost_component_bar_data": cost_bar,
    }

    _write_csv(detail, output_dir / f"{prefix}_purchase_detail.csv")
    _write_or_update_by_algorithm(cost_summary, output_dir / f"{instance}_cost_summary.csv")
    _write_csv(supplier_summary, output_dir / f"{prefix}_supplier_summary.csv")
    _write_csv(supplier_structure, output_dir / f"{prefix}_supplier_structure_metrics.csv")
    _write_csv(base_supplier_long, output_dir / f"{prefix}_base_supplier_matrix_long.csv")
    _write_csv(base_supplier_wide, output_dir / f"{prefix}_base_supplier_matrix_wide.csv")
    _write_csv(cross_base, output_dir / f"{prefix}_cross_base_metrics.csv")
    _write_csv(base_summary, output_dir / f"{prefix}_base_summary.csv")
    _write_csv(inventory_period, output_dir / f"{prefix}_inventory_period_summary.csv")
    _write_csv(inventory_strategy, output_dir / f"{prefix}_inventory_strategy_metrics.csv")
    _write_csv(grade_cost, output_dir / f"{prefix}_grade_cost_summary.csv")
    _write_csv(supplier_pareto, output_dir / f"{prefix}_supplier_pareto_data.csv")
    _write_csv(heatmap, output_dir / f"{prefix}_heatmap_base_supplier.csv")
    _write_csv(inventory_line, output_dir / f"{prefix}_inventory_line_data.csv")
    _write_or_update_by_algorithm(cost_bar, output_dir / f"{instance}_cost_component_bar_data.csv")
    _try_export_existing_algorithm_comparison(instance, output_dir)
    return outputs


def export_algorithm_comparison(instance_name: str, manual_stats: dict[str, pd.DataFrame],
                                ga_stats: dict[str, pd.DataFrame],
                                output_dir: str | Path = "outputs") -> pd.DataFrame:
    comparison = build_algorithm_comparison(manual_stats, ga_stats)
    _write_csv(comparison, Path(output_dir) / f"{instance_name}_algorithm_comparison_summary.csv")
    return comparison
