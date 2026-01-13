import os
import csv
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from cluster.cluster_methods import cluster_main1
import time
import json
from tree.online_SL_functions_for_realdata import (parameter_generation, data_generation, data_choose, information_distance,
                                              information_distance_online, information_distance_online_no, SLLT, tree_from_edge, RFdist)
from tree.online_PE_functions import (tree_from_edge, PEMLT_online, parameter_generation_online_static_one, adjust_para_hat,
                                 renumber_latents_df,
                                 data_generation_one, data_choose, loss_fun)
from tree.BF_functions_for_tree import infer_latents_batch
import random

# # #########################################Model-2#############################################
# # 构造 nodes_name: 前 24 个为 "1", "2", ..., "24"，后 8 个为 "H1", ..., "H8"
# nodes_name = [str(i) for i in range(1, 25)] + [f"H{i}" for i in range(1, 9)]
#
# # 构造 tem，效果等同于 R 中 sequence(rep(8,3)) 后，再转换为矩阵（按行填充）并转换为整数后按列展开
# tem_raw = list(range(1, 9)) * 3        # 生成 [1,2,...,8, 1,2,...,8, 1,2,...,8]
# mat = np.reshape(tem_raw, (3, 8), order='C')   # 3行8列，按行填充
# tem = mat.flatten(order='F')           # 按列展开
# tem = tem.astype(int).tolist()         # 转换为整型列表
# # 得到的 tem 为：[1, 1, 1, 2, 2, 2, 3, 3, 3, ..., 8, 8, 8]
#
# # 构造 father_vertice，先由 paste("H", tem, sep="") 得到 24 个字符串，
# # 然后再拼接额外的 ["", "H1", "H1", "H1", "H2", "H3", "H3", "H4"]
# father_vertice = [f"H{x}" for x in tem] + ["", "H1", "H1", "H1", "H2", "H3", "H3", "H4"]
# # 由于 nodes_name 长度为 32，截取前 32 个元素
# father_vertice = father_vertice[:len(nodes_name)]
#
# # 构造 vertice_type 向量
# vertice_type = (["observable_discrete"] * 3 +
#                 ["observable_continuous"] * 6 +
#                 ["observable_discrete"] * 3 +
#                 ["observable_continuous"] * 6 +
#                 ["observable_discrete"] * 6 +
#                 ["latent"] * 8)
#
# # 构造 DataFrame，并将行索引设置为 nodes_name
# model = pd.DataFrame({
#     "vertice": nodes_name,
#     "vertice_type": vertice_type,
#     "father_vertice": father_vertice
# }, index=nodes_name)


###########打印结果
# print(model)


# #######################################Model-3#############################################
# 定义节点名称（1-13 为观测节点，H1-H8 为隐节点）
nodes_name = [str(i) for i in range(1, 14)] + [f"H{i}" for i in range(1, 9)]

# 定义父节点（按照 R 代码顺序）
father_vertice = ["H2", "H2", "H4", "H5", "H5", "H5", "H6", "H6", "H7", "H7", "H8", "H8", "H8",
                  "", "H1", "H1", "H1", "H2", "H3", "H3", "H4"]

# 定义节点类型
vertice_type = (["observable_continuous"] * 13 + ["latent"] * 8)

# 创建 DataFrame
model = pd.DataFrame({"vertice": nodes_name, "vertice_type": vertice_type, "father_vertice": father_vertice})
model.index = nodes_name

#####打印结果
# print("model:")
print(model)

#######################################初始化############################################

# 初始化保存RF距离的列表
rf_dist_real = []     # 演化前后结构之间的RF距离（RF_dist）
rf_dist_real0 = []  # 每次演化与原始结构之间的RF距离（RF_dist）
rf_dist_estimated = [] # 估计结构与真实结构的RF距离（RF_dist1）
rf_dist_estimated0 = [] # 估计结构与原始结构的RF距离（RF_dist1）
# 获取所有连续节点名称
continuous_nodes = model[model['vertice_type'] == 'observable_continuous']['vertice'].tolist()

# 初始化历史记录容器
history = {node: {'mu0': [], 'mu1': []} for node in continuous_nodes}

#=====================================生成参数（种子等）=====================================

# 设置全局种子，确保数据生成的初始条件相同
np.random.seed(123)
random.seed(123)
N_sim = 1  # 重复次数

p_dim = ((model['vertice_type'] == 'observable_continuous')|(model['vertice_type'] == 'observable_discrete')).sum()

#信息距离中使用
EX_inform_online =  np.zeros(p_dim, dtype=np.float32)
EX2_inform_online= np.zeros((p_dim, p_dim), dtype=np.float32)

#PEMLT中使用
EXYZ11_online= np.zeros((p_dim, p_dim, p_dim), dtype=np.float32)
EXYZ12_online= np.zeros((p_dim, p_dim, p_dim), dtype=np.float32)
EXYZ21_online= np.zeros((p_dim, p_dim, p_dim), dtype=np.float32)
EXYZ22_online= np.zeros((p_dim, p_dim, p_dim), dtype=np.float32)
EXY11_online= np.zeros((p_dim, p_dim), dtype=np.float32)
EXY12_online= np.zeros((p_dim, p_dim), dtype=np.float32)
EXY21_online= np.zeros((p_dim, p_dim), dtype=np.float32)
EXY22_online= np.zeros((p_dim, p_dim), dtype=np.float32)
EX_online  = np.zeros(p_dim,dtype=np.float32)
EX2_online = np.zeros(p_dim,dtype=np.float32)

last_model = model
original_model = model
# 使用固定的种子序列，确保每次运行时生成相同的随机数序列
seeds = list(range(1, 101))  # 1到100的种子序列

for sl_i in range(1):
    # print('sl_i:', sl_i)
    t=sl_i/10
    # print('t',t)
    # 记录循环开始前的时间
    start_time = time.time()

    # 设置当前迭代的随机种子，同时控制Python和NumPy的随机数生成器
    current_seed = seeds[sl_i]
    random.seed(current_seed)
    np.random.seed(current_seed)

    if sl_i == 0:
        new_model = model
        new_model = renumber_latents_df(new_model)  # 对隐变量重新排号
    else:
        new_model = last_model

    #=====================================比较结构：rf距离=====================================

    # 去掉model中"vertice_type"这一列，然后带入tree_from_edge构造tree
    # TreeTr = tree_from_edge(model.drop(columns=["vertice_type"]))
    # print(TreeTr)

    RF_dist1 = RFdist(tree_from_edge(new_model.drop(columns=["vertice_type"])), tree_from_edge(last_model.drop(columns=["vertice_type"])))  # 计算 Robinson-Foulds 距离
    rf_dist_real.append(RF_dist1)  # 演化前后
    RF_dist2 = RFdist(tree_from_edge(new_model.drop(columns=["vertice_type"])),
                      tree_from_edge(original_model.drop(columns=["vertice_type"])))  # 计算 Robinson-Foulds 距离
    rf_dist_real0.append(RF_dist2)  # 演化与原始
    # print("时间演化sl_i：", sl_i)
    # print("演化前后RF距离:", RF_dist)
    last_model = new_model

    # =====================================生成模型参数=====================================
    para_seed = 511
    para = parameter_generation_online_static_one(model, t=t, N_sim=N_sim, seed=para_seed)
    # para = parameter_generation_online(model, t=t, N_sim=N_sim, seed =current_seed)

    # 提取并保存 mu0 和 mu1
    for node in continuous_nodes:
        # para[node] 是形状为 (4, N_sim) 的数组，行依次为 [mu0, mu1, sigma0, sigma1]
        history[node]['mu0'].append(para[node][0])  # 提取 mu0
        history[node]['mu1'].append(para[node][1])  # 提取 mu1

    #=====================================生成数据=====================================
    # if t == 0:
    #     data = data_generation(i=0, model=model, para_all=para, n=sample_size1)  # t=1时生成10000个数据
    # else:
    #     data = data_generation(i=0, model=model, para_all=para, n=sample_size2)  # t!=1时生成1000个数据

    if sl_i == 0:
        sample_size=10000
    else:
        sample_size = 10

    data = data_generation_one(i=0, model=model, para_all=para, n=sample_size, seed=para_seed)
    # data = data_generation(i=0, model=model, para_all=para, n=sample_size, seed =current_seed)

    data_obs = data_choose(data, model)  # 提取观测数据

    # 假设 D 是一个 NumPy 数组
    df_data_obs = pd.DataFrame(data_obs)
    # 保存为 CSV 文件
    # df_data_obs.to_csv(f"data_obs_python_{sl_i}.csv", index=True, header=True)

    ####===================================计算信息距离===================================
    if sl_i==0:
        D = information_distance_online_no(data_obs, EX_inform_online, EX2_inform_online)
    else:
        D = information_distance_online(data_obs, EX_inform_online, EX2_inform_online)  # 计算信息距离矩阵,EX_online, EX2_online直接传引用吧

    # print('chulai',EX_inform_online[0], EX2_inform_online[0])
    # D = information_distance(data_obs)  # 计算信息距离矩阵
    # D = information_distance_online(data_obs, EX_inform_online, EX2_inform_online)
    # 假设 D 是一个 NumPy 数组
    df_D = pd.DataFrame(D)

    #=====================================结构恢复=====================================
    tau1 = 2.0
    tau2 = 4.0
    result = SLLT(D, tau1=tau1, tau2=tau2)

    edges_hat = result["edges"]  # 获取边列表
    TreeEs = result["tree"]      # 获取估计的树结构
    # print(TreeEs)
    epson = result["epson"]      # 获取阈值参数

    #=====================================比较结构=====================================

    #去掉model中"vertice_type"这一列，然后带入tree_from_edge构造tree
    TreeTr = tree_from_edge(new_model.drop(columns=["vertice_type"]))
    # print(TreeTr)

    RF_dist3 = RFdist(tree_from_edge(new_model.drop(columns=["vertice_type"])),
                      TreeEs)  # 计算 Robinson-Foulds 距离
    rf_dist_estimated.append(RF_dist3)  # 估计与当前模型
    RF_dist4 = RFdist(tree_from_edge(original_model.drop(columns=["vertice_type"])),
                      TreeEs)  # 计算 Robinson-Foulds 距离
    rf_dist_estimated0.append(RF_dist4)  # 估计与原始模型

    # # # ========================= 参数估计函数 PEMLT =========================
    #利用结构学习得到的edges_hat构建ModelEs
    ModelEs = pd.DataFrame({
        "vertice": edges_hat["nodes"],
        "vertice_type": [None] * len(edges_hat),
        "father_vertice": edges_hat["parent"]
    })
    # 正常设置索引为 edges_hat["nodes"]
    ModelEs.index = edges_hat["nodes"]
    ModelEs.index.name = None

    # 利用 data_obs 第一行判断每个 observable 是否为连续变量（True 为连续，False 为离散）
    tol = 1e-6
    logi = [abs(data_obs[0, u] - int(data_obs[0, u])) > tol for u in range(data_obs.shape[1])]

    # 遍历 DataFrame，根据 vertices 是否为 observable（纯数字）来更新 vertice_type
    obs_counter = 0  # 用于追踪观测变量在 logi 中的位置
    for index, row in ModelEs.iterrows():
        vertex = row["vertice"]
        # print("vertex: ",vertex,"type(vertex): ",type(vertex))
        if vertex.isdigit():
            # 观测变量：使用 logi 对应位置判断连续/离散
            if logi[obs_counter]:
                ModelEs.at[index, "vertice_type"] = "observable_continuous"
            else:
                ModelEs.at[index, "vertice_type"] = "observable_discrete"
            obs_counter += 1
        else:
            # 非观测变量，统一设置为 latent
            ModelEs.at[index, "vertice_type"] = "latent"

    # print("ModelEs:")
    # print(ModelEs)
    # print(logi)

    ModelEs_renum = renumber_latents_df(ModelEs) #对隐变量重新排号

    # ModelEs_renum.to_csv(f"model3_ModelEs_renum_{sample_size}.csv", index=False, header=True)
    ModelEs_renum.to_csv("model3_ModelEs_renum.csv", index=False, header=True)

    #EXYZ_online，EXY_online，EX_online 是否已经完成更新，初始值为0，一但在PEMLT_online取值为1,表示已经更新完毕

    EXYZ_update = np.zeros((p_dim, p_dim, p_dim), dtype=int)
    EXY_update  = np.zeros((p_dim, p_dim),    dtype=int)
    EX_update   = np.zeros(p_dim,         dtype=int)

    para_hat = PEMLT_online(data_obs,D, ModelEs_renum, EXYZ11_online,EXYZ12_online,EXYZ21_online,EXYZ22_online,EXY11_online, EXY12_online,EXY21_online,EXY22_online, EX2_online, EX_online,
                            EXYZ_update, EXY_update, EX_update, tol)

    #################################传播计算#############################################

    # 重新参数化，让意义更加明确一些
    params = {}
    for node, row in ModelEs_renum.iterrows():
        vtype = row["vertice_type"]
        mat   = para_hat[node]
        # mat = para_hat[node].reshape(2, 2)
        # print("mat:", mat.shape)
        # print(para_hat[node])

        if vtype == "latent":
            #—— 处理潜变量
            if node == "H1":
                # 根节点：para_hat["H1"] 是 P(H1=1)
                p1 = float(mat)
                params[node] = {
                    "states": [0, 1],
                    "prior":  np.array([1-p1, p1])
                }
            else:
                # 其它潜变量：CPT P(node | parent)
                params[node] = {
                    "states": [0, 1],
                    "cpt":    np.array(mat)   # shape (2,2)
                }

        elif vtype == "observable_discrete":
            #—— 离散观测节点：CPT P(obs | parent)
            params[node] = {
                "states": [0, 1],
                "cpt":    np.array(mat)   # shape (2,2)
            }

        elif vtype == "observable_continuous":
            #—— 连续观测节点：第一行是均值，第二行是方差
            means = mat[0, :]           # μ|parent=0, μ|parent=1
            vars_ = mat[1, :]           # σ²|parent=0, σ²|parent=1
            params[node] = {
                "mean": means,          # 1D array length=2
                "var":  vars_           # 1D array length=2
            }

        else:
            raise ValueError(f"未知 vertice_type: {vtype}")

    # 保存为 CSV 文件
    # df_data_obs.to_csv(f"model3_data_obs_python_{sample_size}.csv", index=True, index_label='id', header=True)
    df_data_obs.to_csv("model3_data_obs.csv", index=True, index_label='id', header=True)

    # 使用抽样后的数据计算marginal_prob
    marginal_prob = infer_latents_batch(ModelEs_renum, params, data_obs)

    marginal_prob = pd.DataFrame(marginal_prob)   # list of dict → DataFrame
    marginal_prob.insert(0, 'id', range(len(marginal_prob)))
    # marginal_prob.to_csv(f"model3_marginal_prob_{sample_size}.csv", index=False, header=True)
    marginal_prob.to_csv("model3_marginal_prob.csv", index=False, header=True)

# 第一步：读取观测数据
# file_name = f"model3_data_obs_python_{sample_size}.csv"
file_name = "model3_data_obs.csv"
if not os.path.exists(file_name):
    raise FileNotFoundError(f"找不到文件: {file_name}")

sample_df = pd.read_csv(file_name, nrows=5)
if 'id' in sample_df.columns:
    df_data_obs = pd.read_csv(file_name, index_col='id')
else:
    df_data_obs = pd.read_csv(file_name)
print(f"成功读取观测数据，形状: {df_data_obs.shape}")

# 第二步：读取模型结构和边际概率
# marginal_prob = pd.read_csv(f'model3_marginal_prob_{sample_size}.csv')
# model_es      = pd.read_csv(f"model3_ModelEs_renum_{sample_size}.csv')
marginal_prob = pd.read_csv('model3_marginal_prob.csv')
model_es      = pd.read_csv("model3_ModelEs_renum.csv")

# 删除 id 列（若存在）
if 'id' in marginal_prob.columns:
    marginal_prob.drop(columns=['id'], inplace=True)

# 第三步：解析边际概率字符串 → 数值

# 只对潜变量列做转换
latent_vars = sorted(model_es[model_es['vertice_type']=='latent']['vertice'].tolist())
# print(f"潜变量列表: {latent_vars}")

def extract_p1(cell):
    """
    从 "[p0 p1]" 或 "p0 p1" 格式中提取第二个数 p1。
    支持：numpy-array、列表、字符串。
    """
    # 如果已经是数值，直接返回
    if isinstance(cell, (int, float)):
        return float(cell)
    # 如果是 ndarray
    if isinstance(cell, np.ndarray) or isinstance(cell, list):
        return float(cell[1])
    # 如果是字符串
    s = str(cell).strip()
    # 去掉左右中括号
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    parts = s.replace(',', ' ').split()
    if len(parts) >= 2:
        return float(parts[1])
    # 否则返回 NaN
    return np.nan

# 对每个潜变量列应用转换
for var in latent_vars:
    if var not in marginal_prob.columns:
        raise KeyError(f"在边际概率表中找不到列 '{var}'")
    marginal_prob[var] = marginal_prob[var].apply(extract_p1)

# 检查转换结果
# print("P(var=1) 描述统计：")
# print(marginal_prob[latent_vars].describe())

# 第四步：生成真实标签

def generate_labels(marginal_prob_df, latent_vars, threshold=0.5):
    n = len(marginal_prob_df)
    labels = np.zeros(n, dtype=int)
    # 创建一个DataFrame来存储每个潜变量的二进制值
    binary_df = pd.DataFrame(index=marginal_prob_df.index)
    
    for i in range(n):
        bits = (marginal_prob_df.loc[i, latent_vars].values > threshold).astype(int)
        # 如果全 0，那就把最大概率那一位置 1
        if bits.sum() == 0:
            j = marginal_prob_df.loc[i, latent_vars].values.argmax()
            bits[j] = 1
        
        # 保存每个潜变量的二进制值
        for k, var in enumerate(latent_vars):
            binary_df.loc[marginal_prob_df.index[i], var] = bits[k]
        
        # 二进制转整数
        labels[i] = sum((bits[k] << k) for k in range(len(bits)))
    
    # 打印分布
    unique, counts = np.unique(labels, return_counts=True)
    # print("\n生成的真实标签分布：")
    # for u, c in zip(unique, counts):
    #     print(f"LABEL={u}  #samples={c}")
    
    # 创建结果DataFrame
    result_df = pd.DataFrame({'true_label': labels}, index=marginal_prob_df.index)
    # 添加二进制表示
    result_df = pd.concat([result_df, binary_df], axis=1)
    
    return result_df

# 生成真实标签和二进制表示
true_labels_df = generate_labels(marginal_prob, latent_vars, threshold=0.5)
print(f"最终真实标签：样本数={len(true_labels_df)}, 类别数={len(true_labels_df['true_label'].unique())}")

# 保存真实标签到CSV文件
true_labels_csv = 'model3_true_labels.csv'
true_labels_df.to_csv(true_labels_csv)
print(f"真实标签已保存到 {true_labels_csv}")

# 第五步：计算聚类数（上限 64）

def calculate_num_clusters(model_df):
    h = int((model_df['vertice_type']=='latent').sum())
    nc = min(2**h,64)
    print(f"潜变量数={h}，使用聚类数={nc}")
    return nc

num_clusters = calculate_num_clusters(model_es)

# 第六步：执行聚类
cluster_result = cluster_main1(
    df_data_obs,
    model_es,
    marginal_prob,
    true_labels_df[['true_label']],  # 只传入true_label列
    num_clusters,
    num_runs=10,
    random_seed=42
)

# 第七步：保存结果

# out_file = 'model3_SL_clustering_stats_10.csv'
out_file = 'model3_cluster.csv'
cluster_result.to_csv(out_file, index=False, encoding='utf-8-sig')
print(f"聚类结果保存到 {out_file}")
