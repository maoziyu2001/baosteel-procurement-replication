import sys
import os
from copy import deepcopy

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from read_excel_file import read_xlsx_file_transport, read_xlsx_file_base, read_xlsx_file_supplier
from collections import defaultdict
from decode_yijt_baseline import solve_procument_plan


def solve_procument_plan_greedy(I, beta, W, w0, e0, H, f, Demand, S, alpha, L,
                                J, R, Jsp, z, l, u, r, Q, P, cij,
                                t, K, tau_list, T, yijt):

    """
    return individual.feasible, individual.cost, individual.decode, individual.lowerbound
    """
    x = np.zeros((len(I), len(J), len(T), len(tau_list)))  # 解码后的具体采购量
    min_price = np.zeros((len(I), len(J), len(tau_list)))
    to_serve = np.zeros((len(J), len(tau_list)))
    # remaining_D is only used to guide greedy construction. Feasibility
    # checking and objective evaluation must use the original Demand.
    remaining_D = deepcopy(Demand)

    for tau in tau_list:
        for i in I:
            for j in J:
                if yijt[i][j][0][tau] == 1:
                    min_price[i][j][tau] = (cij[i][j] + (P[j][tau] - r[j] * Q[j]) + f[i] / 2) / z[j]
                else:
                    min_price[i][j][tau] = 1e9

    for j in J:
        for tau in tau_list:
            to_serve[j][tau] = u[j]

    for tau in tau_list:
        flag = True
        while flag:
            flag = False
            min_i = -1
            min_j = -1
            min_p = 1e8
            for i in I:
                for j in J:
                    if min_price[i][j][tau] < min_p:
                        min_i = i
                        min_p = min_price[i][j][tau]
                        min_j = j

            if min_i > -1 and min_j > -1:
                flag = True
                to_buy = remaining_D[min_i][tau] / z[min_j]  # 计算要买的铁重量
                if to_buy > 0 and to_buy + sum(x[min_i][j][0][tau] for j in J) < H[min_i]: # 如果满足仓储上限
                    to_buy = max(l[min_j], min(to_buy, to_serve[min_j][tau]))
                elif sum(x[tmp][min_j][0][tau] for tmp in I) >= l[min_j]:
                    to_buy = 1
                else:
                    to_buy = l[min_j] - sum(x[tmp][min_j][0][tau] for tmp in I)

                x[min_i][min_j][0][tau] = to_buy
                remaining_D[min_i][tau] -= to_buy * z[min_j]
                to_serve[min_j][tau] -= to_buy
                min_price[min_i][min_j][tau] = 1e9


    for i in I:
        for tau in tau_list:
            if remaining_D[i][tau] < -0: # 采购量太多
                max_j = -1
                max_to_buy = 0
                for j in J:
                    if x[i][j][0][tau] > max_to_buy:
                        max_j = j
                        max_to_buy = x[i][j][0][tau]
                to_decrease = -remaining_D[i][tau] / z[max_j]
                max_to_decrease = (x[i][max_j][0][tau] - l[max_j])
                save = min(to_decrease, max_to_decrease)
                remaining_D[i][tau] += save * z[max_j]
                x[i][max_j][0][tau] -= save

    for i in I:
        for tau in tau_list:
            for j in J:
                if x[i][j][0][tau] > 0:
                    buy = sum(x[i][index][0][tau] * z[index] for index in J)
                    supplier = sum(x[index][j][0][tau] for index in I)
                    to_decrease = min((buy-Demand[i][tau]) / z[j], supplier - l[j], x[i][j][0][tau] )
                    if to_decrease > 0:
                        remaining_D[i][tau] += to_decrease * z[j]
                        x[i][j][0][tau] -= to_decrease


    # for i in I:
    #     for tau in tau_list:
    #         tmp = sum(x[i][j][0][tau] * z[j] for j in J)
    #         print(i, tau, tmp, Demand[i][tau], tmp-Demand[i][tau])

    flag, goal = check_constraints(x, I, beta, W, w0, e0, H, f, Demand, S, alpha, L,
                                   J, R, Jsp, z, l, u, r, Q, P, cij,
                                   t, K, tau_list, T, yijt)
    if flag :
        return flag, goal, x, goal
    else:
        return flag, 1e11, x, 1e11

def check_constraints(x, I, beta, W, w0, e0, H, f, D, S, alpha, L,
                                   J, R, Jsp, z, l, u, r, Q, P, cij,
                                   t, K, tau_list, T, yijt):
    t = 0
    T = [0]
    # 计算物料流平衡
    e = np.zeros((len(I), len(T), len(tau_list)))
    w = np.zeros((len(I), len(T), len(tau_list)))
    for tau in tau_list:
        for i in I:
            if tau == t:
                e[i][t][tau] = e0[i] + sum(x[i][j][t][tau] * z[j] for j in J) - D[i][tau]
                w[i][t][tau] = (1 - (
                            D[i][tau] / (e0[i] + sum(x[i][j][t][tau] * z[j] for j in J)))) * (
                                       sum(x[i][j][t][tau] for j in J) + w0[i])
            else:
                e[i][t][tau] = e[i][t][tau - 1] + sum(x[i][j][t][tau] * z[j] for j in J) - D[i][tau]
                w[i][t][tau] = (1 - (D[i][tau] / (
                            e[i][t][tau - 1] + sum(x[i][j][t][tau] * z[j] for j in J)))) * (
                                       sum(x[i][j][t][tau] for j in J) + w[i][t][tau - 1])

    # 1. 基地与供应商契合度约束：只从满足契合度要求的供应商处采购
    for i in I:
        for j in J:
            if alpha[i][j] < beta[i]:
                for tau in tau_list:
                    if x[i][j][t][tau] > 0:
                        print(f"基地{i}与供应商{j}契合度不符合:契合度{alpha[i][j]},下限要求{beta[i]}")
                        return False, 0

    # 2. 采购量上下限约束：长协供应商满足滚动周期总量承诺，普通供应商当期启用时满足最小起购量
    for j in J:
        if j in Jsp:
            total_q = 0
            for tau in tau_list:
                q = sum(x[i][j][t][tau] for i in I)
                total_q += q
                if q > u[j] + 1:
                    print(f"战略供应商{j}周期{tau}采购量不符合:实际采购量{q},采购上限{u[j]}")
                    return False, 0
            if total_q < len(tau_list) * l[j] - 1:
                print(f"战略供应商{j}滚动周期采购总量不符合:实际采购量{total_q},采购总量要求{len(tau_list) * l[j]}")
                return False, 0

        elif j not in Jsp:
            for tau in tau_list:
                q = sum(x[i][j][t][tau] for i in I)
                if q > 0 and (q > u[j] + 1 or q < l[j] - 1):
                    print(f"普通供应商{j}采购量不符合:实际采购量{q},采购要求[{l[j]},{u[j]}]")
                    return False, 0

    # 3.	采购量及配矿约束
    for tau in tau_list:
        if tau == t:  # （决策周期开始时）
            for i in I:
                x_tmp = e0[i] + sum(x[i][j][t][tau] * z[j] for j in J)
                w_tmp = x_tmp / (w0[i] + sum(x[i][j][t][tau] for j in J))
                if (x_tmp < D[i][tau] - 1):
                    print(f"基地{i}周期{tau}采购量不符合:实际采购量{x_tmp},采购要求{D[i][tau]}")
                    return False, 0
                elif (w_tmp < W[i]):
                    print(f"基地{i}周期{tau}配矿比例不符合:实际比例{w_tmp},配矿要求{W[i]}")
                    return False, 0
        else:  # （后续决策周期）
            for i in I:
                x_tmp = e[i][t][tau - 1] + sum(x[i][j][t][tau] * z[j] for j in J)
                w_tmp = x_tmp / (w[i][t][tau - 1] + sum(x[i][j][t][tau] for j in J))
                if (x_tmp < D[i][tau] - 1):
                    print(f"基地{i}周期{tau}采购量不符合:实际采购量{x_tmp},采购要求{D[i][tau]}")
                    return False, 0
                elif (w_tmp < W[i]):
                    print(f"基地{i}周期{tau}配矿比例不符合:实际比例{w_tmp},配矿要求{W[i]}")
                    return False, 0

    # 4. 安全库存及仓储能力上限约束
    condition_satisfied = defaultdict(dict)

    for i in I:
        for j in J:
            tmp = 99
            key = time = tmp * 0.01

            # 筛选出满足条件的ĵ (L[i][ĵ] <= L[i][j])
            valid_js = [j_hat for j_hat in J if L[i][j_hat] <= time]
            condition_satisfied[key] = valid_js
    for i in I:
        for tau in tau_list:
            tmp = 99
            key = time = tmp * 0.01
            # 获取满足条件的ĵ集合
            valid_js = condition_satisfied[key]
            if tau == t:
                s_tmp = e0[i] + sum(x[i][j_hat][t][tau] * z[j_hat] for j_hat in valid_js)
            else:
                s_tmp = e[i, t, tau - 1] + sum(
                    x[i][j_hat][t][tau] * z[j_hat] for j_hat in valid_js)
            ss = s_tmp - time * D[i][tau]
            if ss < S[i][tau] - 1:
                print(f"基地{i}对于供应商{j}在周期{tau}提前期{L[i][j]}内安全库存不满足:最小库存量{ss},安全库存{S[i][tau]}")
                return False, 0

    # # 5. 仓储能力上限约束
    for i in I:
        for tau in tau_list:
            if tau == t:
                w_tmp = w0[i] + sum(x[i][j][t][tau] for j in J)
            else:
                w_tmp = w[i][t][tau-1] + sum(x[i][j][t][tau] for j in J)
            if w_tmp > H[i] + 1:
                print(f"基地{i}在周期{tau}仓储量超出上限:仓储量{w_tmp},仓储上限{H[i]}")
                return False

    q = np.zeros((len(J), len(T), len(tau_list)))
    for j in J:
        for tau in tau_list:
            q[j, t, tau] = sum(x[i][j][t][tau] for i in I)

    # p[j,t,tau]
    p = np.zeros((len(J), len(T), len(tau_list)))
    for j in J:
        for tau in tau_list:
            if q[j, t, tau] < Q[j]:
                p[j, t, tau] = P[j][tau] - r[j] * q[j, t, tau]
            else:
                p[j, t, tau] = P[j][tau] - r[j] * Q[j]

    y2 = np.zeros((len(J)))
    for j in J:
        if sum(q[j, t, tau] for tau in tau_list) > 0:
            y2[j] = 1

    goal = \
        sum(sum(p[j, t, tau] * q[j, t, tau] for j in J) for tau in tau_list) \
        + sum(f[i] * (sum(
            w[i, t, tau] + 1 / 2 * sum(x[i, j, t, tau] for j in J) for tau in tau_list) + 1 / 2 * w0[
                          i] - 1 / 2 * w[i, t, t + K]) for i in I) \
        + sum(sum(sum(cij[i][j] * x[i, j, t, tau] for tau in tau_list) for j in J) for i in I) \
        + sum(y2[j] * R[j] for j in J)
    return True, goal


if __name__ == '__main__':
    base_filename = r'C:\Users\maoziyu\Desktop\baosteel\data\base_data.xlsx'
    supplier_filename = r'C:\Users\maoziyu\Desktop\baosteel\data\supplier_data.xlsx'
    transport_filename = r'C:\Users\maoziyu\Desktop\baosteel\data\transport_data.xlsx'

    n, I, beta, W, w0, e0, H, f, D, S, alpha, L = read_xlsx_file_base(base_filename)
    m, J, R, J1, J2, Jsp, z, l, u, r, Q, P = read_xlsx_file_supplier(supplier_filename)
    G, cjg, cgi, cij = read_xlsx_file_transport(transport_filename)

    t = 0  # 当前决策时期
    K = 1  # 滚动周期数量
    T = [t]
    tau_list = list(range(t, t + K + 1))

    # ----小规模算例----
    I = [0, 1, 2, 3]
    J = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

    flag = False
    while not flag:
        yijt = np.random.randint(0, 2, size=(len(I), len(J), len(T), len(tau_list)), dtype=int)
        flag = True

        for i in I:
            for j in J:
                if alpha[i][j] < beta[i]:
                    for tau in tau_list:
                        yijt[i][j][t][tau] = 0

        for jsp in Jsp:
            if jsp in J and sum(yijt[i,jsp,0,tau] for i in I for tau in tau_list) == 0:
                flag = False

    _, _, x, _ = solve_procument_plan_greedy(I, beta, W, w0, e0, H, f, D, S, alpha, L,
                                    J, R, Jsp, z, l, u, r, Q, P, cij,
                                    t, K, tau_list, T, yijt)

    flag, goal = check_constraints(x, I, beta, W, w0, e0, H, f, D, S, alpha, L,
                                   J, R, Jsp, z, l, u, r, Q, P, cij,
                                   t, K, tau_list, T, yijt)

    print(flag, goal)

    feasible, cost, decode, lowerbound = solve_procument_plan(n, I, beta, W,
                                                              w0, e0, H, f,
                                                              D, S, alpha,
                                                              L,
                                                              m, J, R, J1,
                                                              J2, Jsp, z, l,
                                                              u, r, Q, P,
                                                              G, cjg, cgi,
                                                              cij,
                                                              t, K,
                                                              tau_list, T,
                                                              yijt)
    print(feasible, cost)

    for i in I:
        for j in J:
            for tau in tau_list:
                if x[i][j][0][tau] > 0:
                    print(i,j,tau, x[i][j][0][tau], decode[i][j][0][tau])
    print(goal, cost)
