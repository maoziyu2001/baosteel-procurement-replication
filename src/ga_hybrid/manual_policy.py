import numpy as np
import pandas as pd
import time
import random
import copy
import os

from read_excel_file import read_xlsx_file_transport, read_xlsx_file_base, read_xlsx_file_supplier
from collections import defaultdict
from decode_yijt import solve_procument_plan
from result_analysis import standardize_solution, export_all_analysis_outputs

class HeuristicProcurementAlgorithm:
    def __init__(self, I, J, tau_list, D, z, H, l, u, P, r, R, Q):
        self.bases = I
        self.suppliers = J
        self.periods = tau_list
        self.demand = D.copy()
        self.quality = z
        # 每个周期的仓储容量上限相同，但不能跨周期累计扣减。
        self.capacity = np.tile(H.copy()[:, None], (1, len(self.periods)))
        self.min_order = l
        self.max_order = u.copy()
        self.price = P
        self.price_discount = r
        self.max_price_discount = Q
        self.cooperation_cost = R

        # 初始化采购量矩阵 (基地×供应商×周期)
        self.procurement = np.zeros((len(self.bases), len(self.suppliers), len(self.periods)))

    def execute(self):
        """执行算法"""
        # 步骤0: 分配战略供应商
        self._satisfy_strategy_supplier()

        # 步骤1: 优先满足高品位需求
        self._satisfy_high_quality_demand()

        # 步骤2: 确保最低采购量要求
        self._ensure_minimum_order_quantity()

        # 修复安全库存
        self._fix_safety_storage()
        # 步骤3: 验证解可行性并计算成本
        feasible = self._validate_solution()
        total_cost = self._calculate_total_cost()
        # print(self.procurement)
        return self.procurement, feasible, total_cost

    def _satisfy_strategy_supplier(self):
        n_bases = len(self.bases)
        n_periods = len(self.periods)
        n_suppliers = len(self.suppliers)

        for index in range(n_suppliers):
            j = self.suppliers[index]
            if j in Jsp:
                contract_qty = self.min_order[j] * n_periods
                eligible_bases = [i for i in range(n_bases) if alpha[i][j] > beta[i]]
                if not eligible_bases:
                    continue
                allocatable = contract_qty / (len(eligible_bases) * n_periods)
                for t in range(n_periods):
                    for i in eligible_bases:
                        self.procurement[i, index, t] = allocatable
                        self.demand[i][t] -= allocatable * self.quality[j]
                        self.max_order[j] -= allocatable
                        self.capacity[i][t] -= allocatable

    def _satisfy_high_quality_demand(self):
        """优先满足高品位需求"""
        n_bases = len(self.bases)
        n_periods = len(self.periods)
        n_suppliers = len(self.suppliers)

        for t in range(n_periods):
            # 按铁品位对供应商进行降序排序 (高品位优先)
            # sorted_suppliers = sorted(self.suppliers,
            #                           key=lambda j: (self.price[self.suppliers[j]][t] / self.quality[self.suppliers[j]]),
            #                           reverse=False)
            sorted_suppliers_with_index = sorted(enumerate(self.suppliers),
                                                 key=lambda j: (self.price[self.suppliers[j[0]]][t] / self.quality[self.suppliers[j[0]]]),
                                                reverse=False)

            for i in range(n_bases):
                for index in range(n_suppliers):
                    j = sorted_suppliers_with_index[index][1]
                    if alpha[i][j] > beta[i]:
                        if self.demand[i][t] <= 0:
                            continue  # 需求已满足

                        if self.max_order[j] <= 0:
                            continue  # 供应商能力已耗尽

                        # 计算可采购量
                        allocatable = min(self.demand[i][t]/self.quality[j], self.max_order[j])

                        # 分配采购量
                        self.procurement[i, sorted_suppliers_with_index[index][0], t] = allocatable
                        self.demand[i][t] -= allocatable * self.quality[j]
                        self.max_order[j] -= allocatable
                        self.capacity[i][t] -= allocatable


    def _ensure_minimum_order_quantity(self):
        """确保满足最低采购量要求"""
        n_periods = len(self.periods)
        n_suppliers = len(self.suppliers)
        n_bases = len(self.bases)

        for index in range(n_suppliers):
            j = self.suppliers[index]
            for t in range(n_periods):
                # 计算当前总采购量
                total_procurement = np.sum(self.procurement[:, index, t])

                # 检查是否满足最低采购量要求
                if total_procurement > 0 and total_procurement < self.min_order[j]:
                    deficit = self.min_order[j] - total_procurement

                    # 如果供应商还有剩余能力
                    for i in range(n_bases):
                        if alpha[i][j] > beta[i]:
                            if self.capacity[i][t] > 0:
                                # 计算可增加的采购量
                                additional = min(deficit, self.capacity[i][t])

                                self.procurement[i, index, t] += additional
                                self.capacity[i][t] -= additional

    def _fix_safety_storage(self):
        n_suppliers = len(self.suppliers)
        condition_satisfied = defaultdict(dict)

        # 计算物料流平衡
        e = np.zeros((len(I), len(T), len(tau_list)))
        w = np.zeros((len(I), len(T), len(tau_list)))
        for tau in tau_list:
            for i in I:
                if tau == t:
                    e[i][t][tau] = e0[i] + sum(
                        self.procurement[i][j][tau] * self.quality[self.suppliers[j]] for j in range(n_suppliers)) - \
                                   D[i][tau]
                    w[i][t][tau] = (1 - (
                            D[i][tau] / (e0[i] + sum(
                        self.procurement[i][j][tau] * self.quality[self.suppliers[j]] for j in
                        range(n_suppliers))))) * (
                                           sum(self.procurement[i][j][tau] for j in range(n_suppliers)) + w0[i])
                else:
                    e[i][t][tau] = e[i][t][tau - 1] + sum(
                        self.procurement[i][j][tau] * self.quality[self.suppliers[j]] for j in range(n_suppliers)) - \
                                   D[i][tau]
                    w[i][t][tau] = (1 - (D[i][tau] / (
                            e[i][t][tau - 1] + sum(
                        self.procurement[i][j][tau] * self.quality[self.suppliers[j]] for j in
                        range(n_suppliers))))) * (
                                           sum(self.procurement[i][j][tau] for j in range(n_suppliers)) + w[i][t][
                                       tau - 1])
        for i in I:
            for j in J:
                for tmp in range(100):
                    key = time = tmp * 0.01

                    # 筛选出满足条件的ĵ (L[i][ĵ] <= L[i][j])
                    valid_js = [j_hat for j_hat in J if L[i][j_hat] <= time]
                    condition_satisfied[key] = valid_js

        for tmp in range(100):
            key = time = tmp * 0.01
            # 获取满足条件的ĵ集合
            valid_js = condition_satisfied[key]
            condition_satisfied[key] = valid_js
            for i in I:
                for tau in tau_list:
                    if tau == t:
                        s_tmp = e0[i] + sum(
                            self.procurement[i][j][tau] * self.quality[self.suppliers[j]] for j in range(n_suppliers) if
                            self.suppliers[j] in valid_js)

                    else:
                        s_tmp = e[i, t, tau - 1] + sum(
                            self.procurement[i][j][tau] * self.quality[self.suppliers[j]] for j in range(n_suppliers) if
                            self.suppliers[j] in valid_js)

                    ss = s_tmp - time * D[i][tau]
                    while ss < S[i][tau]:
                        quantity = S[i][tau] - ss
                        for index in range(n_suppliers):
                            if alpha[i][index] > beta[i]:
                                if (self.suppliers[index] in valid_js) and self.max_order[index] > 0 and ss < S[i][tau]:

                                    additional = min(self.max_order[index], self.capacity[i][tau], quantity/self.quality[self.suppliers[index]])

                                    self.procurement[i, index, t] += additional
                                    self.capacity[i][tau] -= additional
                                    ss += additional * self.quality[self.suppliers[index]]


    def _validate_solution(self):

        T = [0]
        t = 0
        n_suppliers = len(self.suppliers)

        # 计算物料流平衡
        e = np.zeros((len(I), len(T), len(tau_list)))
        w = np.zeros((len(I), len(T), len(tau_list)))
        for tau in tau_list:
            for i in I:
                if tau == t:
                    e[i][t][tau] = e0[i] + sum(self.procurement[i][j][tau] * self.quality[self.suppliers[j]]  for j in range(n_suppliers)) - D[i][tau]
                    w[i][t][tau] = (1 - (
                                D[i][tau] / (e0[i] + sum(self.procurement[i][j][tau] * self.quality[self.suppliers[j]] for j in range(n_suppliers))))) * (
                                               sum(self.procurement[i][j][tau] for j in range(n_suppliers)) + w0[i])
                else:
                    e[i][t][tau] = e[i][t][tau - 1] + sum(self.procurement[i][j][tau] * self.quality[self.suppliers[j]] for j in range(n_suppliers)) - D[i][tau]
                    w[i][t][tau] = (1 - (D[i][tau] / (
                                e[i][t][tau - 1] + sum(self.procurement[i][j][tau] * self.quality[self.suppliers[j]] for j in range(n_suppliers))))) * (
                                               sum(self.procurement[i][j][tau] for j in range(n_suppliers)) + w[i][t][tau - 1])

        # 1. 基地与供应商契合度约束：只从满足契合度要求的供应商处采购
        for i in I:
            for index in range(n_suppliers):
                j = self.suppliers[index]
                if alpha[i][j] < beta[i]:
                    for tau in tau_list:
                        if self.procurement[i][index][tau] > 0:
                            print(f"基地{i}与供应商{j}契合度不符合:契合度{alpha[i][j]},下限要求{beta[i]}")
                            return False

        # 2. 采购量上下限约束：长协供应商满足滚动周期总量承诺，普通供应商当期启用时满足最小起购量
        for index in range(n_suppliers):
            j = self.suppliers[index]
            if j in Jsp:
                total_q = 0
                for tau in tau_list:
                    q = sum(self.procurement[i][index][tau] for i in I)
                    total_q += q
                    if q > u[j] + 1:
                        print(f"战略供应商{j}周期{tau}采购量不符合:实际采购量{q},采购上限{u[j]}")
                        return False
                if total_q < len(tau_list) * l[j] - 1:
                    print(f"战略供应商{j}滚动周期采购总量不符合:实际采购量{total_q},采购总量要求{len(tau_list) * l[j]}")
                    return False
            elif j not in Jsp:
                for tau in tau_list:
                    q = sum(self.procurement[i][index][tau] for i in I)
                    if q > 0 and (q > u[j] + 1 or q < l[j] - 1):
                        print(f"普通供应商{j}采购量不符合:实际采购量{q},采购要求[{l[j]},{u[j]}]")
                        return False

        # 3.采购量及配矿约束
        for tau in tau_list:
            if tau == t:  # （决策周期开始时）
                for i in I:
                    x_tmp = e0[i] + sum(self.procurement[i][j][tau] *  self.quality[self.suppliers[j]] for j in range(n_suppliers))
                    w_tmp = x_tmp / (w0[i] + sum(self.procurement[i][j][tau] for j in range(n_suppliers)))
                    if (x_tmp < D[i][tau] - 1):
                        print(f"基地{i}周期{tau}采购量不符合:实际采购量{x_tmp},采购要求{D[i][tau]}")
                        return False
                    elif (w_tmp < W[i]):
                        print(f"基地{i}周期{tau}配矿比例不符合:实际比例{w_tmp},配矿要求{W[i]}")
                        return False
            else:  # （后续决策周期）
                for i in I:
                    x_tmp = e[i][t][tau - 1] + sum(self.procurement[i][j][tau] *  self.quality[self.suppliers[j]] for j in range(n_suppliers))
                    w_tmp = x_tmp / (w[i][t][tau - 1] + sum(self.procurement[i][j][tau] for j in range(n_suppliers)))
                    if (x_tmp < D[i][tau] - 1):
                        print(f"基地{i}周期{tau}采购量不符合:实际采购量{x_tmp},采购要求{D[i][tau]}")
                        return False
                    elif (w_tmp < W[i]):
                        print(f"基地{i}周期{tau}配矿比例不符合:实际比例{w_tmp},配矿要求{W[i]}")
                        return False

        # 4. 安全库存及仓储能力上限约束
        condition_satisfied = defaultdict(dict)
        for i in I:
            for j in J:
                for tmp in range(100):
                    key = time = tmp * 0.01

                    # 筛选出满足条件的ĵ (L[i][ĵ] <= L[i][j])
                    valid_js = [j_hat for j_hat in J if L[i][j_hat] <= time]
                    condition_satisfied[key] = valid_js

        for tmp in range(100):
            key = time = tmp * 0.01
            # 获取满足条件的ĵ集合
            valid_js = condition_satisfied[key]
            condition_satisfied[key] = valid_js
            for i in I:
                for tau in tau_list:
                    if tau == t:
                        s_tmp = e0[i] + sum(self.procurement[i][j][tau] * self.quality[self.suppliers[j]] for j in range(n_suppliers) if self.suppliers[j] in valid_js)

                    else:
                        s_tmp = e[i, t, tau - 1]+ sum(self.procurement[i][j][tau] * self.quality[self.suppliers[j]] for j in range(n_suppliers) if self.suppliers[j] in valid_js)

                    ss = s_tmp - time * D[i][tau]
                    if ss < S[i][tau] - 1:
                        print(
                            f"基地{i}对于供应商{j}在周期{tau}提前期{time}内安全库存不满足:最小库存量{ss},安全库存{S[i][tau]}")
                        return False

        return True

    def _calculate_total_cost(self):
        T = [t]
        n_suppliers = len(self.suppliers)
        # 计算物料流平衡
        e = np.zeros((len(I), len(T), len(tau_list)))
        w = np.zeros((len(I), len(T), len(tau_list)))
        for tau in tau_list:
            for i in I:
                if tau == t:
                    e[i][t][tau] = e0[i] + sum(self.procurement[i][j][tau] * self.quality[self.suppliers[j]] for j in range(n_suppliers)) - D[i][tau]
                    w[i][t][tau] = (1 - (
                            D[i][tau] / (e0[i] + sum(
                        self.procurement[i][j][tau] * self.quality[self.suppliers[j]] for j in
                        range(n_suppliers))))) * (
                                           sum(self.procurement[i][j][tau] for j in range(n_suppliers)) + w0[i])
                else:
                    e[i][t][tau] = e[i][t][tau - 1] + sum(
                        self.procurement[i][j][tau] * self.quality[self.suppliers[j]] for j in range(n_suppliers)) - \
                                   D[i][tau]
                    w[i][t][tau] = (1 - (D[i][tau] / (
                            e[i][t][tau - 1] + sum(
                        self.procurement[i][j][tau] * self.quality[self.suppliers[j]] for j in
                        range(n_suppliers))))) * (
                                           sum(self.procurement[i][j][tau] for j in range(n_suppliers)) + w[i][t][
                                       tau - 1])

        # q[j,t,tau]
        q = np.zeros((len(J), len(T), len(tau_list)))
        for j in range(n_suppliers):
            for tau in tau_list:
                q[j, t, tau] = sum(self.procurement[i][j][tau] for i in I)

        # p[j,t,tau]
        p = np.zeros((len(J), len(T), len(tau_list)))
        for index in range(n_suppliers):
            j = self.suppliers[index]
            for tau in tau_list:
                if q[index, t, tau] < Q[j]:
                    p[index, t, tau] = P[j][tau] - r[j] * q[index, t, tau]
                else:
                    p[index, t, tau] = P[j][tau] - r[j] * Q[j]

        # y[j]
        y2 = np.zeros((len(J)))
        for j in range(n_suppliers):
            if sum(q[j, t, tau] for tau in tau_list) > 0:
                y2[j] = 1

        goal = \
            sum(sum(p[j, t, tau] * q[j, t, tau] for j in range(n_suppliers)) for tau in tau_list) \
            + sum(f[i] * (
                        sum(w[i, t, tau] + 1 / 2 * sum(self.procurement[i][j][tau] for j in range(n_suppliers)) for tau in tau_list) + 1 / 2 *
                        w0[i] - 1 / 2 * w[i, t, t + K]) for i in I) \
            + sum(sum(sum(cij[i][self.suppliers[j]] * self.procurement[i][j][tau] for tau in tau_list) for j in range(n_suppliers)) for i in I) \
            + sum(y2[j] * R[self.suppliers[j]] for j in range(n_suppliers))

        # print(goal)
        return goal
if __name__ == '__main__':

    base_filename = r'C:\Users\maoziyu\Desktop\baosteel\data\base_data.xlsx'
    supplier_filename = r'C:\Users\maoziyu\Desktop\baosteel\data\supplier_data.xlsx'
    transport_filename = r'C:\Users\maoziyu\Desktop\baosteel\data\transport_data.xlsx'
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if not os.path.exists(base_filename):
        base_filename = os.path.join(project_root, 'data', 'base_data.xlsx')
        supplier_filename = os.path.join(project_root, 'data', 'supplier_data.xlsx')
        transport_filename = os.path.join(project_root, 'data', 'transport_data.xlsx')
    n, I, beta, W, w0, e0, H, f, D, S, alpha, L = read_xlsx_file_base(base_filename)
    m, J, R, J1, J2, Jsp, z, l, u, r, Q, P = read_xlsx_file_supplier(supplier_filename)
    G, cjg, cgi, cij = read_xlsx_file_transport(transport_filename)

    t = 0  # 当前决策时期
    K = 1  # 滚动周期数量
    T = [t]
    tau_list = list(range(t, t + K + 1))
    # print(I,J)
    # J = [0,1,2,3,4,5,6]
    # print(D)
    # ----小规模算例----
    I = [0, 1, 2, 3]
    J = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
         21, 22, 23, 24, 25, 26, 27, 28, 29]
    # 创建算法实例并执行
    manual_start_time = time.time()
    algorithm = HeuristicProcurementAlgorithm(
        I, J, tau_list, D, z, H, l, u, P, r, R, Q
    )

    procurement_plan, feasible, total_cost = algorithm.execute()
    runtime_seconds = time.time() - manual_start_time

    # 输出结果
    print("采购计划生成完成:\n ", procurement_plan[:,:,0], procurement_plan[:,:,1])
    print(f"可行性: {'可行' if feasible else '不可行'}")
    print(f"总成本: {total_cost:.2f} 元")

    instance_data = {
        "n": n, "I": I, "beta": beta, "W": W, "w0": w0, "e0": e0, "H": H,
        "f": f, "D": D, "S": S, "alpha": alpha, "L": L, "m": m, "J": J,
        "R": R, "J1": J1, "J2": J2, "Jsp": Jsp, "z": z, "l": l, "u": u,
        "r": r, "Q": Q, "P": P, "G": G, "cjg": cjg, "cgi": cgi, "cij": cij,
        "t": t, "K": K, "T": T, "tau_list": tau_list,
    }
    raw_solution = {
        "procurement": procurement_plan,
        "cost": total_cost,
        "feasible": feasible,
        "runtime_seconds": runtime_seconds,
    }
    instance_name = f"case_{len(I)}x{len(J)}"
    standardized = standardize_solution(raw_solution, instance_data, "manual", instance_name)
    export_all_analysis_outputs(standardized, output_dir=os.path.join(project_root, "outputs"))
    print(f"Analysis CSV files exported to {os.path.join(project_root, 'outputs')}")
