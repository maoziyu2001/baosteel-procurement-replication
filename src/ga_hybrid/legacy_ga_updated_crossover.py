import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gurobipy import Model, GRB, quicksum
import pandas as pd
from read_excel_file import read_xlsx_file_transport, read_xlsx_file_base, read_xlsx_file_supplier
from collections import defaultdict
from decode_yijt import solve_procument_plan
import time
import random 
import copy

class Individual:
    """表示单个采购计划方案"""
    def __init__(self):
        # 决策变量：是否采购0-1矩阵,IxJxTxtau
        self.yijt = np.zeros((len(I), len(J), len(T), len(tau_list)))
        
        # 其他辅助变量+
        self.cost = 1e12  # 目标函数值
        self.decode = np.zeros((len(I), len(J), len(T), len(tau_list)))  # 解码后的具体采购量
        self.feasible = False     # 可行性标志
        self.lowerbound = 1e10  # gurobi得到的下界
        self.fitness = 0


class GA:
    """遗传算法主框架"""
    def __init__(self, pop_size=2, max_gen=2, mut_prob=0.2, max_time = 36000, RA_max = 0.8, RA_min = 0.5):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.mut_prob = mut_prob
        self.max_time = max_time
        self.RA_max = RA_max
        self.RA_min = RA_min
        self.gen = 0
        self.k1 = 0.5
        self.k2 = 0.5
        self.k3 = 0.5
        self.k4 = 0.5

    def initialize_population(self) -> list[Individual]:
        """初始化种群：生成随机满足基本约束的个体"""
        population = []
        for _ in range(self.pop_size):
            # print(f"子代数量:{len(population)}")
            ind = Individual()
            # 随机生成采购方案（需满足供应商能力约束）
            flag = False
            while not flag:
                flag = True
                yijt = np.random.randint(0, 2, size=(len(I), len(J), len(T), len(tau_list)), dtype=int)

                for i in I:
                    for j in J:
                        if alpha[i][j] < beta[i]:
                            for tau in tau_list:
                                yijt[i][j][t][tau] = 0

                flag, result, x2, lower_bound = solve_procument_plan(n, I, beta, W, w0, e0, H, f, D, S, alpha, L,
                                        m, J, R, J1, J2, Jsp, z, l, u, r, Q, P, 
                                        G, cjg, cgi, cij, 
                                        t, K, tau_list, T, yijt)

            ind.yijt = yijt
            ind.cost = result
            ind.feasible = flag
            ind.decode = x2
            ind.lowerbound = lower_bound
            population.append(ind)
        return population
    
    def evaluate(self, individual: Individual):
        # 对于给定的个体，解码得到具体的采购方案
        individual.feasible, individual.cost, individual.decode, individual.lowerbound = solve_procument_plan(n, I, beta, W, w0, e0, H, f, D, S, alpha, L,
                                        m, J, R, J1, J2, Jsp, z, l, u, r, Q, P, 
                                        G, cjg, cgi, cij, 
                                        t, K, tau_list, T, individual.yijt)
        
    def tournament_selection(self, population: list[Individual], k=3) -> Individual:
        """锦标赛选择"""
        contestants = np.random.choice(population, k)
        # 选择表现最好的个体 (成本最低)
        return min(contestants, key=lambda x: (x.cost))
    
    def crossover(self, parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
        """交叉操作 - 对于每个供应商，随机从两个父代中选择一个采购方案遗传"""
        child1, child2 = Individual(), Individual()

        pc = 0.5
        for j in J:
            if np.random.rand() < pc:
                child1.yijt[:, j, :, :] = parent1.yijt[:, j, :, :].copy()
                child2.yijt[:, j, :, :] = parent2.yijt[:, j, :, :].copy()
            else:
                child1.yijt[:, j, :, :] = parent2.yijt[:, j, :, :].copy()
                child2.yijt[:, j, :, :] = parent1.yijt[:, j, :, :].copy()

        return child1, child2
    
    # def crossover(self, parent1, parent2):
    #     """按供应商-基地组合进行块交叉"""
    #     child1 = Individual()
    #     child2 = Individual()

    #     # 随机选择一些供应商-基地组合
    #     num_blocks = random.randint(1, min(len(I), len(J)))
    #     selected_pairs = []

    #     for _ in range(num_blocks):
    #         i = random.randint(0, len(I)-1)
    #         j = random.randint(0, len(J)-1)
    #         selected_pairs.append((i, j))

    #     # 交换选定的供应商-基地组合的所有决策
    #     for i, j in selected_pairs:
    #         child1.yijt[i, j, :, :] = parent2.yijt[i, j, :, :]
    #         child2.yijt[i, j, :, :] = parent1.yijt[i, j, :, :]

    #     # 复制其余部分
    #     for i in range(len(I)):
    #         for j in range(len(J)):
    #             if (i, j) not in selected_pairs:
    #                 child1.yijt[i, j, :, :] = parent1.yijt[i, j, :, :]
    #                 child2.yijt[i, j, :, :] = parent2.yijt[i, j, :, :]

    #     return child1, child2

    def update_fitness(self, population):
        max_cost = 0
        min_cost = 1e12
        for ind in population:
            if ind.cost > max_cost:
                max_cost = ind.cost
            if ind.cost < min_cost:
                min_cost = ind.cost
        for ind in population:
            ind.fitness = (max_cost - ind.cost) / (max_cost - min_cost + 1e4)

    def adapative_pc(self, population, parent1, parent2):
        f_bar = max(parent1.fitness, parent2.fitness)
        f_ave = sum([ind.fitness for ind in population]) / len(population)
        f_max = max(ind.fitness for ind in population)
        if f_bar >= f_ave:
            pc = self.k1 * (f_max-f_bar) / (f_max - f_ave + 1e-6)
        else:
            pc = self.k3
        return pc

    def adapative_pm(self, population, child):
        f_ave = sum([ind.fitness for ind in population]) / len(population)
        f_max = max(ind.fitness for ind in population)
        if child.fitness >= f_ave:
            pm = self.k2 * (f_max-child.fitness) / (f_max - f_ave + 1e-6)
        else:
            pm = self.k4
        self.mut_prob = pm

    def mutate1(self, individual: Individual):
        """变异操作 - 随机选择某个点位进行反转"""
        for i in I:
            for j in J:
                for tau in tau_list:
                    if np.random.rand() < 0.15:
                        individual.yijt[i, j, 0, tau] = 1 - individual.yijt[i, j, 0, tau]
        for i in I:
            for j in J:
                if alpha[i][j] < beta[i]:
                    for tau in tau_list:
                        individual.yijt[i, j, 0, tau] = 0
        return individual
    
    def mutate2(self,individual):
        """将某个基地的采购从当前供应商转移到另一个供应商"""
        # 减少某给定基地在某给定时期选择进行采购的供应商数量
        i = random.randint(0, len(I)-1)
        tau = random.randint(0, len(tau_list)-1)
        
        # 找出当前有采购的供应商
        current_suppliers = []
        for j in range(len(J)):
            if individual.yijt[i, j, 0, tau] == 1:
                current_suppliers.append(j)
                break
        
        # 计算平均每个基地每个时期选择的供应商数量
        total_decisions = np.sum(individual.yijt)
        ave_supplier_num = int(total_decisions / (len(tau_list) * len(I)))

        # 供应商数量超过平均数时，减少供应商
        if len(current_suppliers) >= ave_supplier_num:
            if len(current_suppliers)>1:
                num_to_choose = random.randint(1, len(current_suppliers) - 1)
            else:
                num_to_choose = 1
            new_j = random.sample(current_suppliers, num_to_choose)
            
        else:
            # 随机选择新供应商
            selected_j = [j for j in J if alpha[i][j] > beta[i]]
            # num_to_choose = random.randint(len(current_suppliers), min(ave_supplier_num ,len(selected_j)))
            num_to_choose = random.randint(max(1, len(current_suppliers)-1), min(ave_supplier_num, len(selected_j)))
            new_j = random.sample(selected_j, num_to_choose)

        # 转移采购决策
        for j in current_suppliers:
            individual.yijt[i, j, t, tau] = 0
        for j in new_j:
            individual.yijt[i, j, t, tau] = 1
        
        return individual

    def is_individual_in_population(self,new_individual, population):
        """检查新个体是否已存在于种群中（基于 yijt 内容比较）"""
        for existing_individual in population:
            if np.array_equal(new_individual.yijt, existing_individual.yijt):
                return True
        return False

    def run(self):
        """主优化循环"""
        start_time = time.time()

        # 初始化种群
        population = self.initialize_population()
        print(">>> Initialization completed.")


        # for i in range(len(population)):
        #     # print("generation:",i)
        #     best_solution = population[i]
        #     localsearch = LocalSearch(best_solution, max_iter = 5)
        #     sol = localsearch.run()
        #     # print(sol.cost)
        #     population[i] = sol
        #     print(2)
        # 记录最佳解
        best_individual = min(population, key=lambda x: (x.cost))
        best_lower_bound = min(population, key=lambda x: (x.lowerbound)).lowerbound
        history = []
        print(f"Gen {0}: Best Cost={best_individual.cost:.2f}\n"
              f"Lower bound={best_lower_bound:.2f}\n"
              f"Gap:{(best_individual.cost-best_individual.lowerbound)/best_individual.cost * 100:.2f}%")
        print("Best 5 individuals", end=':')
        for ind in population[:5]:
            print(f"{ind.cost:.2f},", end = ' ')
        print('')
        print(f'Iteration start. Time limit:{self.max_time}')
        # 主循环
        for gen in range(self.max_gen):
            print(f"\n>>> Generation {gen}")
            # ------【限时检测】---------------------------------
            if time.time() - start_time >= self.max_time:
                print(f"\n>>> Reach time limit ({self.max_time}s). Stop at generation {gen}.")
                break

            self.gen = gen
            # 选择
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)
            # print(parent1.cost,parent2.cost)

            # 交叉
            child1, child2 = self.crossover(parent1, parent2)
            RA = self.RA_min + (self.RA_max - self.RA_min) * (1 - self.gen / self.max_gen)
            # 变异
            self.adapative_pm(population, child1)
            if np.random.rand() < self.mut_prob:
                if np.random.rand() < RA:
                    self.mutate2(child1)  # 自适应
                else:
                    self.mutate1(child1)  # 随机
            self.adapative_pm(population, child2)
            if np.random.rand() < self.mut_prob:
                if np.random.rand() < RA:
                    self.mutate2(child2)  # 自适应
                else:
                    self.mutate1(child2)  # 随机

            # 评估子代
            self.evaluate(child1)
            self.evaluate(child2)
            # print(child1.cost, child2.cost)

            while not (child1.feasible and child2.feasible):
                # 交叉
                child1, child2 = self.crossover(parent1, parent2)
                # 变异
                self.adapative_pm(population, child1)
                if np.random.rand() < self.mut_prob:
                    if np.random.rand() < RA:
                        self.mutate2(child2)  # 自适应
                    else:
                        self.mutate1(child2)  # 随机

                self.adapative_pm(population, child2)
                if np.random.rand() < self.mut_prob:
                    if np.random.rand() < RA:
                        self.mutate2(child2)  # 自适应
                    else:
                        self.mutate1(child2)  # 随机

                # 评估子代
                self.evaluate(child1)
                self.evaluate(child2)

            # 更新种群
            if  (not self.is_individual_in_population(child1, population)) and (not self.is_individual_in_population(child2, population)):
                population.sort(key=lambda x: x.cost)  # 原地排序

                if child1.cost < population[-2].cost and child2.cost < population[-2].cost:
                    population = population[:-2]           # 移除最后两个（成本最高的）
                    population.append(child1)
                    population.append(child2)
                    
                    for i in [-1,-2]:
                        best_solution = population[i]
                        localsearch = LocalSearch(best_solution, max_iter = 5)
                        sol = localsearch.run()
                        # print(sol.cost)
                        population[i] = sol

                    # 更新最佳解
                    current_best = min(population, key=lambda x: (x.cost))
                    if current_best.cost < best_individual.cost:
                        best_individual = current_best
                    
                    # 更新下界
                    current_bound = min(population, key=lambda x: (x.lowerbound))
                    if current_bound.lowerbound < best_lower_bound:
                        best_lower_bound = current_bound.lowerbound

                    # 记录迭代信息
                    avg_cost = sum(ind.cost for ind in population) / len(population)
                    feasible_count = sum(1 for ind in population if ind.feasible)
                    history.append((gen, best_individual.cost,best_lower_bound, avg_cost, feasible_count))
                    
                    # 输出进度
                    print(f"Best Cost={best_individual.cost:.2f}\n"
                        f"Lower bound={best_lower_bound:.2f}\n"
                        f"Gap:{(best_individual.cost-best_lower_bound)/best_individual.cost * 100:.2f}%\n"
                        f"Time={time.time()-start_time:.2f}s")
                    print("Best 5 individuals", end=':')

                    for ind in population[:5]:
                        print(f"{ind.cost:.2f},", end=' ')
                    print('')
                else:
                    print("No update found.")
            else:
                print("No update found.")

            # else:
            #     print(f"重复{child1.cost,child2.cost}")
            
            
        return best_individual, history

class LocalSearch:
    def __init__(self, ind, max_iter = 10):
        self.ind = ind  # 每次对一个个体进行局部搜索
        self.best_solution = copy.deepcopy(ind)
        self.max_iter = max_iter
        
    def generate_neighbors_1(self):
        neighbors = []
        # count = 0
        
        # 遍历所有可能的决策点 (i, j, t, tau)
        for i in I:
            for j in J:
                for tau in tau_list:
                    # 只考虑当前为1的点位
                    if self.best_solution.yijt[i, j, 0, tau] == 1:
                        # 创建新解
                        neighbor = copy.deepcopy(self.best_solution)
                        
                        # 步骤1: 将点位设置为0，采购量设置为0
                        neighbor.yijt[i, j, 0, tau] = 0
                        
                        # 更新采购量
                        # original_qty = self.best_solution.decode[i, j, 0, tau]
                        neighbor.decode[i, j, 0, tau] = 0
                        
                        # 检查约束是否违反
                        flag, goal = self.check_constraints(neighbor)
                        if flag:
                            # 如果没有违反约束，添加到候选列表
                            # print(type(neighbor.decode))
                            neighbor.cost = goal
                            neighbors.append(neighbor)

        return neighbors

    def generate_neighbors_2(self):
        neighbors = []
        # 遍历所有可能的决策点 (i, j, t, tau)
        for i in I:
            for tau in tau_list:
                for j1 in range(0, len(J)-1):
                    for j2 in range(j1+1, len(J)):
                        # 只考虑当前为1的点位
                        if self.best_solution.yijt[i, j1, 0, tau] == 1 and self.best_solution.yijt[i, j2, 0, tau] == 1:
                            # 创建新解
                            neighbor = copy.deepcopy(self.best_solution)

                            iron1 = neighbor.decode[i, j1, 0, tau] * z[j1]
                            iron2 = neighbor.decode[i, j2, 0, tau] * z[j2]

                            if iron1 > iron2: # 保证j1是铁含量较小的供应商
                                tmp = j1
                                j1 = j2
                                j2 = tmp

                            # 步骤1: 将点位设置为0，采购量设置为0
                            neighbor.yijt[i, j1, 0, tau] = 0

                            # 更新采购量
                            neighbor.decode[i, j1, 0, tau] = 0
                            neighbor.decode[i, j2, 0, tau] += iron1 / z[j2]

                            flag, goal = self.check_constraints(neighbor)
                            # 检查约束是否违反
                            if flag:
                                # 如果没有违反约束，添加到候选列表
                                neighbor.cost = goal
                                neighbors.append(neighbor)

        return neighbors

    def generate_neighbors_3(self):
        neighbors = []
        # 遍历所有可能的决策点 (i, j, t, tau)
        for i in I:
            for j in J:
                for tau1 in range(0, len(tau_list)-1):
                    for tau2 in range(tau1+1, len(tau_list)):
                        # 只考虑当前为1的点位
                        if self.best_solution.yijt[i, j, 0, tau1] == 1 and self.best_solution.yijt[i, j, 0, tau2] == 1:
                            # 创建新解
                            neighbor = copy.deepcopy(self.best_solution)

                            # 步骤1: 将点位设置为0，采购量设置为0
                            neighbor.yijt[i, j, 0, tau2] = 0

                            # 更新采购量

                            neighbor.decode[i, j, 0, tau1] += neighbor.decode[i, j, 0, tau2]
                            neighbor.decode[i, j, 0, tau2] = 0

                            flag, goal = self.check_constraints(neighbor)
                            # 检查约束是否违反
                            if flag:
                                # 如果没有违反约束，添加到候选列表
                                neighbor.cost = goal
                                neighbors.append(neighbor)

        return neighbors
    
    def check_constraints(self, neighbor):
        
        T = [0]
        # 计算物料流平衡
        e = np.zeros((len(I), len(T), len(tau_list)))
        w = np.zeros((len(I), len(T), len(tau_list)))   
        for tau in tau_list:
            for i in I:
                if tau == t:
                    e[i][t][tau] = e0[i] + sum(neighbor.decode[i][j][t][tau] * z[j] for j in J) - D[i][tau]
                    w[i][t][tau] = (1 - (D[i][tau] / (e0[i] + sum(neighbor.decode[i][j][t][tau] * z[j] for j in J))))*(sum(neighbor.decode[i][j][t][tau] for j in J) + w0[i])
                else:
                    e[i][t][tau] = e[i][t][tau-1] + sum(neighbor.decode[i][j][t][tau] * z[j] for j in J) - D[i][tau]
                    w[i][t][tau] = (1 - (D[i][tau] / (e[i][t][tau-1] + sum(neighbor.decode[i][j][t][tau] * z[j] for j in J))))*(sum(neighbor.decode[i][j][t][tau] for j in J) + w[i][t][tau-1])
            
        # 1. 基地与供应商契合度约束：只从满足契合度要求的供应商处采购
        for i in I:
            for j in J:
                if alpha[i][j] < beta[i]:
                    for tau in tau_list:
                        if neighbor.decode[i][j][t][tau] > 0: 
                            print(f"基地{i}与供应商{j}契合度不符合:契合度{alpha[i][j]},下限要求{beta[i]}")
                            return False,0

        # 2. 采购量上下限约束：长协供应商满足滚动周期总量承诺，普通供应商当期启用时满足最小起购量
        for j in J:
            if j in Jsp:
                total_q = 0
                for tau in tau_list:
                    q = sum(neighbor.decode[i][j][t][tau] for i in I)
                    total_q += q
                    if q > u[j] + 1:
                        # print(f"战略供应商{j}周期{tau}采购量不符合:实际采购量{q},采购上限{u[j]}")
                        return False,0
                if total_q < len(tau_list) * l[j] - 1:
                    # print(f"战略供应商{j}滚动周期采购总量不符合:实际采购量{total_q},采购总量要求{len(tau_list) * l[j]}")
                    return False,0
            elif j not in Jsp:
                for tau in tau_list:
                    q = sum(neighbor.decode[i][j][t][tau] for i in I)
                    if q > 0 and (q > u[j] + 1 or q < l[j] - 1):
                        # print(f"普通供应商{j}采购量不符合:实际采购量{q},采购要求[{l[j]},{u[j]}]")
                        return False,0

        #3.	采购量及配矿约束
        for tau in tau_list:
            if tau == t:  # （决策周期开始时）
                for i in I:
                    x_tmp = e0[i] + sum(neighbor.decode[i][j][t][tau] * z[j] for j in J)
                    w_tmp = x_tmp / (w0[i] + sum(neighbor.decode[i][j][t][tau] for j in J))
                    if (x_tmp < D[i][tau] - 1):
                        # print(f"基地{i}周期{tau}采购量不符合:实际采购量{x_tmp},采购要求{D[i][tau]}")
                        return False,0
                    elif (w_tmp  < W[i]):
                        # print(f"基地{i}周期{tau}配矿比例不符合:实际比例{w_tmp},配矿要求{W[i]}")
                        return False,0
            else: # （后续决策周期）
                for i in I:
                    x_tmp = e[i][t][tau-1] + sum(neighbor.decode[i][j][t][tau] * z[j] for j in J)
                    w_tmp = x_tmp / (w[i][t][tau-1] + sum(neighbor.decode[i][j][t][tau] for j in J))
                    if (x_tmp < D[i][tau] - 1):
                        # print(f"基地{i}周期{tau}采购量不符合:实际采购量{x_tmp},采购要求{D[i][tau]}")
                        return False,0
                    elif (w_tmp < W[i]):
                        # print(f"基地{i}周期{tau}配矿比例不符合:实际比例{w_tmp},配矿要求{W[i]}")
                        return False,0
                    
        # 4. 安全库存及仓储能力上限约束
        condition_satisfied = defaultdict(dict)
        # for i in I:
        #     for j in J:
        #         key = (i, j)
        #         # 筛选出满足条件的ĵ (L[i][ĵ] <= L[i][j])
        #         valid_js = [j_hat for j_hat in J if L[i][j_hat] <= L[i][j]]
        #         condition_satisfied[key] = valid_js
        for i in I:
            for j in J:
                for tmp in range(100):
                    key = time = tmp * 0.01

                    # 筛选出满足条件的ĵ (L[i][ĵ] <= L[i][j])
                    valid_js = [j_hat for j_hat in J if L[i][j_hat] <= time]
                    condition_satisfied[key] = valid_js
        for i in I:
            for tau in tau_list:
                for tmp in range(100):
                    key = time = tmp * 0.01
                    # 获取满足条件的ĵ集合
                    valid_js = condition_satisfied[key]
                    if tau == t:
                        s_tmp =  e0[i] + sum(neighbor.decode[i][j_hat][t][tau] * z[j_hat] for j_hat in valid_js ) 
                    else:
                        s_tmp =  e[i,t,tau-1] + sum(neighbor.decode[i][j_hat][t][tau] * z[j_hat] for j_hat in valid_js ) 
                    ss = s_tmp - time * D[i][tau]
                    if ss < S[i][tau] - 1:
                        # print(f"基地{i}对于供应商{j}在周期{tau}提前期{L[i][j]}内安全库存不满足:最小库存量{ss},安全库存{S[i][tau]}")
                        return False, 0

        # # 5. 仓储能力上限约束
        # for i in I:
        #     for tau in tau_list:
        #         if tau == t:
        #             w_tmp = w0[i] + sum(neighbor.decode[i][j][t][tau] for j in J)
        #         else:
        #             w_tmp = w[i][t][tau-1] + sum(neighbor.decode[i][j][t][tau] for j in J)
        #         if w_tmp > H[i] + 1:
        #             # print(f"基地{i}在周期{tau}仓储量超出上限:仓储量{w_tmp},仓储上限{H[i]}") 
        #             return False
        # print("当前解为可行解")


        q = np.zeros((len(J), len(T), len(tau_list)))
        for j in J:
            for tau in tau_list:
                q[j, t, tau] = sum(neighbor.decode[i][j][t][tau] for i in I)

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
                w[i, t, tau] + 1 / 2 * sum(neighbor.decode[i, j, t, tau] for j in J) for tau in tau_list) + 1 / 2 * w0[
                              i] - 1 / 2 * w[i, t, t + K]) for i in I) \
            + sum(sum(sum(cij[i][j] * neighbor.decode[i, j, t, tau] for tau in tau_list) for j in J) for i in I) \
            + sum(y2[j] * R[j] for j in J)
        return True, goal

    def evaluate(self,solution):
        """评估解决方案的总成本"""
        T = [t]
        # 计算物料流平衡
        e = np.zeros((len(I), len(T), len(tau_list)))
        w = np.zeros((len(I), len(T), len(tau_list)))   
        for tau in tau_list:
            for i in I:
                if tau == t:
                    e[i][t][tau] = e0[i] + sum(solution[i][j][t][tau] * z[j] for j in J) - D[i][tau]
                    w[i][t][tau] = (1 - (D[i][tau] / (e0[i] + sum(solution[i][j][t][tau] * z[j] for j in J))))*(sum(solution[i][j][t][tau] for j in J) + w0[i])
                else:
                    e[i][t][tau] = e[i][t][tau-1] + sum(solution[i][j][t][tau] * z[j] for j in J) - D[i][tau]
                    w[i][t][tau] = (1 - (D[i][tau] / (e[i][t][tau-1] + sum(solution[i][j][t][tau] * z[j] for j in J))))*(sum(solution[i][j][t][tau] for j in J) + w[i][t][tau-1])
        
        # q[j,t,tau]
        q = np.zeros((len(J), len(T), len(tau_list)))
        for j in J:
            for tau in tau_list:
                q[j,t,tau] = sum(solution[i][j][t][tau] for i in I)

        # p[j,t,tau]
        p = np.zeros((len(J), len(T), len(tau_list)))
        for j in J:
            for tau in tau_list:
                if q[j,t,tau] < Q[j]:
                    p[j,t,tau] = P[j][tau]-r[j]*q[j,t,tau]
                else:
                    p[j,t,tau] = P[j][tau]-r[j]*Q[j]

        # y[j]
        y2 = np.zeros((len(J)))
        for j in J:
            if sum(q[j,t,tau] for tau in tau_list) > 0:
                y2[j] = 1

        goal = \
        sum(sum( p[j,t,tau] * q[j,t,tau] for j in J)for tau in tau_list) \
            + sum(f[i]*(sum(w[i,t,tau]+1/2*sum(solution[i,j,t,tau] for j in J) for tau in tau_list) + 1/2*w0[i] - 1/2*w[i,t,t+K]) for i in I) \
                + sum(sum(sum(cij[i][j] * solution[i,j,t,tau] for tau in tau_list) for j in J) for i in I) \
                    + sum(y2[j] * R[j] for j in J )
        
        # print(goal)
        return goal
    
    def run(self, max_neighbors=100):
        '''
        暂未考虑gurobi验证的情况
        '''
        improved = True
        iteration = 0
        # print(f"第{iteration}次搜索:{self.best_solution.cost}")

        while improved and iteration < self.max_iter:
            improved = False
            iteration += 1
            neighbor_solutions = []
            for num in range(3):
                # 步骤1: 生成邻域解
                # print(num)
                neighbor_solution = []
                if num == 0:
                    neighbor_solution = self.generate_neighbors_1()  # 邻域解的集合
                # elif num == 1:
                #     neighbor_solution = self.generate_neighbors_2()  # 邻域解的集合
                # else:
                #     neighbor_solution = self.generate_neighbors_3()  # 邻域解的集合
                neighbor_solutions.extend(neighbor_solution)

            # 按估计成本排序
            if neighbor_solutions:
                neighbor_solutions.sort(key=lambda x: x.cost)

                 # 步骤3: 检查并应用改进
                if not neighbor_solutions or neighbor_solutions[0].cost >= self.best_solution.cost:
                    # print(f"第{iteration}次搜索:{self.best_solution.cost}")
                    break  # 估计成本没有改进，跳过
                else:
                    self.best_solution = neighbor_solutions[0]
                    # print(f"第{iteration}次搜索:{self.best_solution.cost}")
                    improved = True

        return self.best_solution



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
    # print(I,J)
    # J = [0,1,2,3,4,5,6]

    # ----小规模算例----
    I = [0, 1, 2, 3]
    J = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
         30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]


    filename = r'\optimization_history_4x50-4h.xlsx'
    # ------------------
    ga = GA(pop_size=150, max_gen=200000, mut_prob=0.2,
            max_time = 3600*4, RA_max = 0.8, RA_min = 0.5)
    best_solution, history = ga.run()
    
    # 创建列名列表
    columns = ["Generation", "Best Cost", "Lower Bound", "Average Cost", "Feasible Count"]

    # 将history转换为DataFrame
    df = pd.DataFrame(history, columns=columns)
    df.to_excel(r"C:\Users\maoziyu\Desktop\baosteel\procurement-v3\history"+filename, index=False,
                engine="openpyxl")

    # 输出结果和分析
    print(f"\nOptimization Completed!")
    print(f"Best Solution Cost: {best_solution.cost}")
    print(f"Feasible: {'Yes' if best_solution.feasible else 'No'}")

    localsearch = LocalSearch(best_solution, max_iter = 100)
    localsearch.run()
