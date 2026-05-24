import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def read_xlsx_file_base(base_filename):
    
    # 指定Excel文件路径
    base_file_path = base_filename  
    
    # 基地信息
    df_base = pd.read_excel(base_file_path, sheet_name='base', index_col=0)  
    n = df_base.shape[0]                             # 计算基地的总数
    I = df_base['index_i'].to_numpy()                    # 基地编号
    beta = df_base['fitness_threshold_beta'].to_numpy()  # 最小契合度要求
    W =  df_base['content_threshold_G'].to_numpy()    # 最小成分含量要求
    w0 = df_base["material_init_stock_w"].to_numpy()     # 原材料起始库存   
    e0 = df_base["material_init_use_e"].to_numpy()     # 原材料起始有效含量
    H = df_base["storage_cap_H"].to_numpy()   # 存储能力
    f = df_base["storage_cost_f"].to_numpy()   # 存储成本

    # 各生产需求信息
    df_demand = pd.read_excel(base_file_path, sheet_name='demand', index_col=0) 
    D = df_demand.to_numpy()

    # 安全库存信息
    df_safetystock = pd.read_excel(base_file_path, sheet_name='safetystock', index_col=0) 
    S = df_safetystock.to_numpy()

    # 供应商基地之间契合度
    df_fitness = pd.read_excel(base_file_path, sheet_name='fitness', index_col=0)  
    alpha = df_fitness.to_numpy()

    # 提前期
    df_leadtime = pd.read_excel(base_file_path, sheet_name='leadtime', index_col=0)  
    L = df_leadtime.to_numpy()
    return n, I, beta, W, w0, e0, H, f, D, S, alpha, L

def read_xlsx_file_supplier(supplier_filename):
    supplier_file_path = supplier_filename

    # 供应商信息
    df_supplier_cost = pd.read_excel(supplier_file_path, sheet_name='cost', index_col=0) 
    m = df_supplier_cost.shape[0]                                     # 计算供应商的总数
    J = df_supplier_cost['index'].to_numpy()                          # 供应商编号
    R = df_supplier_cost['cor_cost_R'].to_numpy()                     # 固定合作成本
    J1_lis = []  # 海外供应商
    J2_lis = []  # 进口供应商
    for j in J:
        if df_supplier_cost['site_tag'][j] == 1:
            J1_lis.append(j)
        elif df_supplier_cost['site_tag'][j] == 2:
            J2_lis.append(j)
    J1 = np.array(J1_lis)
    J2 = np.array(J2_lis)
    
    # 原料信息
    df_material = pd.read_excel(supplier_file_path, sheet_name='material', index_col=0) 
    sp_lis = [] # 存储特殊供应商信息
    for j in J:
        if df_material['strategy_tag'][j] == 1:
            sp_lis.append(j)
    Jsp = np.array(sp_lis)
    z = df_material["content_z"].to_numpy()   # 铁含量均值
    l = df_material["lower_limit"].to_numpy()   # 采购下限
    u = df_material["upper_limit"].to_numpy()   # 采购上限
    r = df_material["price_discount"].to_numpy()   # 价格折扣
    Q = df_material["price_discount_threshold"].to_numpy()   # 最大价格折扣
    
    # 价格信息
    df_price = pd.read_excel(supplier_file_path, sheet_name='price', index_col=0) 
    P = df_price.to_numpy()
    
    return m, J, R, J1, J2, Jsp, z, l, u, r, Q, P

def read_xlsx_file_transport(transport_filename):
    transport_file_path = transport_filename
    df_port = pd.read_excel(transport_filename, sheet_name='port', index_col=0)        
    G = df_port['index'].to_numpy()                    # 港口编号

    df_cjg = pd.read_excel(transport_file_path, sheet_name='supplier-port-cjg', index_col=0) 
    cjg = df_cjg.to_numpy()

    df_cgi = pd.read_excel(transport_file_path, sheet_name='port-base-cgi', index_col=0) 
    cgi = df_cgi.to_numpy()

    df_cij = pd.read_excel(transport_file_path, sheet_name='base-supplier-cij', index_col=0) 
    cij = df_cij.to_numpy()

    return G, cjg, cgi, cij

if __name__ == '__main__':
    base_filename = r'C:C:\Users\admin\Desktop\baosteel\procurement-v2\25-07-17\data\base_data.xlsx'
    supplier_filename = r'C:\Users\admin\Desktop\baosteel\procurement-v2\25-07-17\data\supplier_data.xlsx'
    transport_filename = r'C:\Users\admin\Desktop\baosteel\procurement-v2\25-07-17\data\transport_data.xlsx'

    n, I, beta, W, w0, e0, H, f, D, S, alpha, L = read_xlsx_file_base(base_filename)
    m, J, R, J1, J2, Jsp, z, l, u, r, Q, P = read_xlsx_file_supplier(supplier_filename)
    G, cjg, cgi, cij = read_xlsx_file_transport(transport_filename)
