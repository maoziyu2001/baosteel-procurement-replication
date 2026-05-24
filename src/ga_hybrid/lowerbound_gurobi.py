import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gurobipy import Model, GRB, quicksum
import pandas as pd
from read_excel_file import read_xlsx_file_transport, read_xlsx_file_base, read_xlsx_file_supplier
from collections import defaultdict


def solve_procument_plan(n, I, beta, W, w0, e0, H, f, D, S, alpha, L,
                         m, J, R, J1, J2, Jsp, z, l, u, r, Q, P, 
                         G, cjg, cgi, cij, 
                         t, K, tau_list, T,
                         time_limit=None, outputflag=1, return_results=False):
    
    # 结果存储路径
    folder_name = r'C:\Users\maoziyu\Desktop\baosteel'

    # ---------------
    # 建立模型
    model = Model('solve_procument')

    # 大M
    M = 10e9
    
    # ---------------
    # 决策变量

    # 决策周期t，当前及未来周期τ时基地i从供应商j采购并通过中转海港（或虚拟海港）g转运的铁矿石原料重量
    # x1 = model.addVars(I, J, T, tau_list, G, vtype=GRB.CONTINUOUS, name = 'x1[i,j,t,tau,g]')
    x2 = model.addVars(I, J, T, tau_list, vtype=GRB.CONTINUOUS, name = 'x2[i,j,t,tau]')
    
    # t月末（t+1月初），基地i库存原材料总重量
    w = model.addVars(I, T, tau_list, vtype=GRB.CONTINUOUS, name = 'w[i,t,tau]')
    # t月末（t+1月初），基地i库存原材料有效成分含量
    e = model.addVars(I, T, tau_list, vtype=GRB.CONTINUOUS, name = 'e[i,t,tau]')

    # 0-1变量，y1_jt,tau=1表示在周期t决策时，企业在周期τ从供应商j采购
    y1 = model.addVars(J, T, tau_list, vtype=GRB.BINARY, name = 'y1[j,t,tau]')
    # 0-1变量，y2_j=1表示集团当年将从供应商j处采购原材料
    y2 = model.addVars(J, vtype=GRB.BINARY, name = 'y2[j]')
    # 0-1变量，y3_j=1表示基地i当年将从供应商j处采购原材料
    y3 = model.addVars(I, J, vtype=GRB.BINARY, name = 'y3[i,j]')
    
    # 供应商j的的原材料单价
    p = model.addVars(J, T, tau_list, vtype=GRB.CONTINUOUS, name = 'p[j,t,tau]')
    # 所有基地全年从供应商j采购的所有原材料总重量
    q = model.addVars(J, T, tau_list, vtype=GRB.CONTINUOUS, name = 'q[j,t,tau]')


    # ---------------
    # 优化目标: 最小化整个周期内的采购总成本（原料成本+库存成本+运输成本+合作成本）

    model.setObjective(
        quicksum(quicksum( p[j,t,tau] * q[j,t,tau] for j in J)for tau in tau_list)
                        + quicksum(f[i]*(quicksum(e[i,t,tau]+1/2*quicksum(x2[i,j,t,tau] for j in J) for tau in tau_list) + 1/2*e0[i] - 1/2*e[i,t,t+K]) for i in I)
                        + quicksum(quicksum(quicksum(cij[i][j] * x2[i,j,t,tau] for tau in tau_list) for j in J) for i in I)
                        + quicksum(y2[j] * R[j] for j in J )
                       , GRB.MINIMIZE)
    # ---------------
    # 约束条件

    # # 1. 铁矿石中转海港选择约束
    # for g in G:
    #     if g == 0:
    #         model.addConstrs(x1[i,j,t,tau,g] == 0 for i in I for j in J1 for tau in tau_list)
    #     else:
    #         model.addConstrs(x1[i,j,t,tau,g] == 0 for i in I for j in J2 for tau in tau_list)

    # -----无变量替换-----
    # 2. 铁矿石库存量和铁含量平衡约束
    for tau in tau_list:
        if tau == t:
            model.addConstrs(e[i,t,tau]  == e0[i] + quicksum(x2[i,j,t,tau]*z[j] for j in J) - D[i][tau] for i in I)
            # model.addConstrs(w[i,t,tau] * (e0[i] + quicksum(x2[i,j,t,tau]*z[j] for j in J))  == ((e0[i] + quicksum(x2[i,j,t,tau]*z[j] for j in J))-D[i][tau])*(quicksum(x2[i,j,t,tau] for j in J) + w0[i]) for i in I)
            # model.addConstrs(e[i,t,tau]  >= e0[i] + quicksum(x2[i,j,t,tau]*z[j] for j in J) - D[i][tau] for i in I)
            # model.addConstrs(w[i,t,tau] * (e0[i] + quicksum(x2[i,j,t,tau]*z[j] for j in J))  >= ((e0[i] + quicksum(x2[i,j,t,tau]*z[j] for j in J))-D[i][tau])*(quicksum(x2[i,j,t,tau] for j in J) + w0[i]) for i in I)
        
        else:
            model.addConstrs(e[i,t,tau]  == e[i,t,tau-1] + quicksum(x2[i,j,t,tau]*z[j] for j in J) - D[i][tau] for i in I)
            # model.addConstrs(w[i,t,tau] * (e[i,t,tau-1] + quicksum(x2[i,j,t,tau]*z[j] for j in J))  == ((e[i,t,tau-1] + quicksum(x2[i,j,t,tau]*z[j] for j in J))-D[i][tau])*(quicksum(x2[i,j,t,tau] for j in J) + w[i,t,tau-1]) for i in I)

    # 3. 采购量及配矿约束
    for tau in tau_list:
        if tau == t:
            model.addConstrs(e0[i] + quicksum(x2[i,j,t,tau]*z[j] for j in J) >= D[i][tau] for i in I)
            model.addConstrs((e0[i] + quicksum(x2[i,j,t,tau]*z[j] for j in J)) >= W[i] * (w0[i] + quicksum(x2[i,j,t,tau] for j in J)) for i in I)
        else:
            model.addConstrs(e[i,t,tau-1] + quicksum(x2[i,j,t,tau]*z[j] for j in J) >= D[i][tau] for i in I)     
            model.addConstrs((e[i,t,tau-1] + quicksum(x2[i,j,t,tau]*z[j] for j in J)) >= W[i] * (w[i,t,tau-1] + quicksum(x2[i,j,t,tau] for j in J)) for i in I)

    # -----有变量替换-----
    # 2. 铁矿石库存量和铁含量平衡约束
    # e1 = model.addVars(I, T, tau_list, vtype=GRB.CONTINUOUS, name = 'e1[i,t,tau]')
    # e2 = model.addVars(I, T, tau_list, vtype=GRB.CONTINUOUS, name = 'e2[i,t,tau]')
    # e3 = model.addVars(I, T, tau_list, vtype=GRB.CONTINUOUS, name = 'e3[i,t,tau]')
    # e4 = model.addVars(I, T, tau_list, vtype=GRB.CONTINUOUS, name = 'e4[i,t,tau]')
    # e5 = model.addVars(I, T, tau_list, vtype=GRB.CONTINUOUS, name = 'e5[i,t,tau]')

    # for tau in tau_list:
    #     if tau == t:
    #         model.addConstrs(e1[i,t,tau] == e0[i] + quicksum(x2[i,j,t,tau]*z[j] for j in J) for i in I)
    #         model.addConstrs(e3[i,t,tau] == (e1[i,t,tau]-D[i][tau]) for i in I)
    #         model.addConstrs(e[i,t,tau]  == e3[i,t,tau]  for i in I)
    #         model.addConstrs(e4[i,t,tau] == w0[i] + quicksum(x2[i,j,t,tau] for j in J)  for i in I)
    #         model.addConstrs(e5[i,t,tau] == e3[i,t,tau]*e4[i,t,tau] for i in I)
    #         model.addConstrs(e2[i,t,tau] == w[i,t,tau] * e1[i,t,tau] for i in I)
    #         model.addConstrs(e2[i,t,tau] == e5[i,t,tau] for i in I)
            
    #     else:
    #         model.addConstrs(e1[i,t,tau] == e[i,t,tau-1] + quicksum(x2[i,j,t,tau]*z[j] for j in J) for i in I)
    #         model.addConstrs(e3[i,t,tau] == (e1[i,t,tau]-D[i][tau]) for i in I)
    #         model.addConstrs(e[i,t,tau]  == e3[i,t,tau]  for i in I)           
    #         model.addConstrs(e2[i,t,tau] == w[i,t,tau] * e1[i,t,tau] for i in I)
    #         model.addConstrs(e4[i,t,tau] == w[i,t,tau-1] + quicksum(x2[i,j,t,tau] for j in J)  for i in I)         
    #         model.addConstrs(e5[i,t,tau] == e3[i,t,tau]*e4[i,t,tau] for i in I)
    #         model.addConstrs(e2[i,t,tau] == e5[i,t,tau]for i in I)
  

    # # 3. 采购量及配矿约束
    # model.addConstrs(e1[i,t,tau] >= D[i][tau] for i in I for tau in tau_list)     
    # model.addConstrs(e1[i,t,tau] >= W[i] * e4[i,t,tau] for i in I for tau in tau_list)

    # ----------
    # 4.价格折扣约束
    for j in J:
        for tau in tau_list:
            model.addGenConstrPWL(q[j,t,tau], p[j,t,tau], [0, Q[j], u[j]],[P[j][tau], P[j][tau]-r[j]*Q[j], P[j][tau]-r[j]*Q[j]])

    # 5. 安全库存及存储能力上限约束
    condition_satisfied = defaultdict(dict)
    for i in I:
        for j in J:
            for tmp in range(100):
                key = time = tmp * 0.01
                
                # 筛选出满足条件的ĵ (L[i][ĵ] <= L[i][j])
                valid_js = [j_hat for j_hat in J if L[i][j_hat] <= time]
                condition_satisfied[key] = valid_js
    
    for i in I:
        for j in J:
            for tau in tau_list:
                for tmp in range(100):
                    key = time = tmp * 0.01
                    # 获取满足条件的ĵ集合
                    valid_js = condition_satisfied[key]
                    if tau == t:
                        model.addConstr(e0[i] +  quicksum(x2[i, j_hat, t, tau] * z[j_hat] for j_hat in valid_js ) - time * D[i][tau] >= S[i][tau])
                    else:
                        model.addConstr(e[i,t,tau-1] +  quicksum(x2[i, j_hat, t, tau] * z[j_hat] for j_hat in valid_js ) - time * D[i][tau] >= S[i][tau])
        
    for tau in tau_list:
        if tau == t:
            model.addConstrs(e0[i] + quicksum(x2[i,j,t,tau] for j in J) <= H[i] for i in I)
        else:
            model.addConstrs(e[i,t,tau-1] + quicksum(x2[i,j,t,tau] for j in J) <= H[i] for i in I)


    # 6. 基地与供应商契合度约束
    model.addConstrs((1-y3[i,j])*M+alpha[i][j] >= beta[i] for i in I for j in J )
    model.addConstrs(y3[i,j]*M >= quicksum(x2[i,j,t,tau] for tau in tau_list) for i in I for j in J )
    model.addConstrs(y3[i,j] <= quicksum(x2[i,j,t,tau] for tau in tau_list) for i in I for j in J )

    # 7. 采购量上下限约束
    # 普通供应商按单周期启用状态约束最小起购量；长协供应商按整个滚动采购周期约束承诺总量。
    model.addConstrs(quicksum(x2[i,j,t,tau] for i in I) >= l[j]*y1[j,t,tau] for tau in tau_list for j in J if j not in Jsp )
    model.addConstrs(quicksum(x2[i,j,t,tau] for i in I) <= u[j]*y1[j,t,tau] for tau in tau_list for j in J )
    model.addConstrs(quicksum(quicksum(x2[i,jsp,t,tau] for i in I) for tau in tau_list) >= len(tau_list)*l[jsp] for jsp in Jsp if jsp in J )
    
    # 8. 其他均衡和线性化约束
    # model.addConstrs(quicksum(x1[i,j,t,tau,g] for g in G) == x2[i,j,t,tau] for i in I for j in J for tau in tau_list)
    # model.addConstrs(quicksum(x2[i,j,t,tau] for i in I) == q[j,t,tau] for j in J for tau in tau_list)
    # model.addConstrs(M*y2[j] >= quicksum(y3[i,j] for i in I) for j in J )
    # model.addConstrs(y2[j] <= quicksum(y3[i,j] for i in I) for j in J )
    
    model.addConstrs(quicksum(x2[i,j,t,tau] for i in I) == q[j,t,tau] for j in J for tau in tau_list)

    model.addConstrs(M*y2[j] >= quicksum(y3[i,j] for i in I) for j in J )
    model.addConstrs(y2[j] <= quicksum(y3[i,j] for i in I) for j in J )

    model.addConstrs(M * quicksum(y1[j,t,tau] for tau in tau_list) >= quicksum(y3[i,j] for i in I) for j in J)
    model.addConstrs(M * quicksum(y3[i,j] for i in I) >= quicksum(y1[j,t,tau] for tau in tau_list) for j in J)

    # ---------------
    # 设置求解时间上限。默认保持原脚本行为；消融实验可传入限时和静默输出。
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    model.Params.OutputFlag = outputflag

    # ---------------
    # model.setParam(GRB.Param.StartNodeLimit, 1000)
    model.params.NonConvex = 2
    model.optimize()  # 求解


    # ----------------
    # 如果模型不可行则输出冲突约束
    if return_results:
        return {
            "status": int(model.Status),
            "objective": float(model.objVal) if model.SolCount > 0 else None,
            "lower_bound": float(model.ObjBound) if hasattr(model, "ObjBound") else None,
            "runtime": float(model.Runtime) if hasattr(model, "Runtime") else None,
        }

    if model.Status == GRB.INFEASIBLE or model.Status == GRB.Status.INF_OR_UNBD:
        model.computeIIS()
        for c in model.getConstrs():
            if c.IISConstr:  # 如果约束在 IIS 中
                print(f"冲突约束: {c.ConstrName} = {model.getRow(c)} {c.Sense} {c.RHS}")    
    
    # ----------------
    # 输出求解结果
    else:
        for t in T: # 依次输出每个原材料的计划
            filename = folder_name+"\month_"+str(t)+"plan.xlsx"

            with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
                for tau in tau_list:
                    x_result = pd.DataFrame(data=None, columns=J, index = I)

                    for i in I:
                        for j in J:
                            x_result.loc[i, j] = x2[i,j,t,tau].x
                    
                    st = str(tau+1)+'月'
                    x_result.loc['总计'] =x_result.sum()  # 添加一行，存储每列的总和
                    x_result['总计'] = x_result.sum(axis=1)  # 添加一列，存储每行的总和
                    x_result.to_excel(writer, sheet_name=st)

                iron_result = pd.DataFrame(data=None, columns=tau_list, index = I)
                for i in I:
                    for tau in tau_list:
                        iron_result.loc[i, tau] = e[i,t, tau].x
                iron_result.to_excel(writer, sheet_name="每月剩余库存含量")
    

    #             q_result = pd.DataFrame(data=None, index = J)
    #             for j in J:
    #                 q_result.loc[j, g] = q[j,g].x
    #             q_result.loc['总计'] =q_result.sum()  
    #             q_result.to_excel(writer, sheet_name="供应商订购总量")

    #             iron_result = pd.DataFrame(data=None, columns=T, index = I)
    #             for i in I:
    #                 for t in T:
    #                     iron_result.loc[i, t] = e[i,g,t].x
    #             iron_result.to_excel(writer, sheet_name="每月剩余库存含量")

    #             st_result = pd.DataFrame(data=None, columns=T, index = I)
    #             for i in I:
    #                 for t in T:
    #                     st_result.loc[i, t] = w[i,g,t].x
    #             st_result.to_excel(writer, sheet_name="每月剩余重量")


    return folder_name


if __name__ == '__main__':
    base_filename = r'/Users/yuyuyu/Desktop/260413/baosteel/data/base_data.xlsx'
    supplier_filename = r'/Users/yuyuyu/Desktop/260413/baosteel/data/supplier_data.xlsx'
    transport_filename = r'/Users/yuyuyu/Desktop/260413/baosteel/data/transport_data.xlsx'

    n, I, beta, W, w0, e0, H, f, D, S, alpha, L = read_xlsx_file_base(base_filename)
    m, J, R, J1, J2, Jsp, z, l, u, r, Q, P = read_xlsx_file_supplier(supplier_filename)
    G, cjg, cgi, cij = read_xlsx_file_transport(transport_filename)

    print("基地总数:", n)
    print("供应商总数:", m)
    print(W, D)
    t = 0  # 当前决策时期
    K = 1  # 滚动周期数量
    T = [t]
    # J = [0,1,2,3,4,5,6]

    # ----小规模算例----
    I = [0, 1, 2, 3,4]
    J = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
         21, 22, 23, 24, 25, 26, 27, 28, 29]

    # ------------------

    tau_list = list(range(t, t + K + 1))
    print(alpha[1][5], beta[1])
    solve_procument_plan(n, I, beta, W, w0, e0, H, f, D, S, alpha, L,
                         m, J, R, J1, J2, Jsp, z, l, u, r, Q, P, 
                         G, cjg, cgi, cij, 
                         t, K, tau_list, T)
    
    
    # excel_folder = r"C:\Users\admin\Desktop\baosteel\procurement\update"
    # x_init, w_init, e_init = single_material_combination(excel_folder, n, m, t, b, I, J)
    # solve_procument_plan(I, J, T, K, G, o, cor, sp, w0, e0, h, f, s, z, d, alpha, alpha0, l, u, pj, rp, qj_max, c, R, Gr, x_init, w_init, e_init)

    
