import os
import numpy as np
import pandas as pd
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
from tree.SL_functions_for_realdata import (information_distance, SLLT, tree_from_edge, RFdist)
from tree.PE_functions_for_realdata import (tree_from_edge, PEMLT, parameter_generation,adjust_para_hat,renumber_latents_df,
                            data_generation2, data_choose, loss_fun)
from tree.BF_functions_for_tree import infer_latents_batch
# from cluster.cluster_methods_real import cluster_main1

# 设置工作目录
# os.chdir("C:/Users/86182/Desktop/0516")

#=========================================读取数据================================================
# data_obs=pd.read_csv("feature_matrix_last_layer.csv", index_col=False)

data_obs=pd.read_csv("./data/frog/修改后frog.csv", index_col=False)
print("前几行数据：")
print(data_obs[:5])

# 假设 D 是一个 NumPy 数组
df_data_obs = pd.DataFrame(data_obs)
# 保存为 CSV 文件
df_data_obs.to_csv("frog_data_obs.csv", index=False, header=True)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# fit_transform 会计算均值和标准差并同时做变换                   
data_obs = scaler.fit_transform(data_obs)

#=====================================结构恢复=====================================
D = information_distance(data_obs)  # 计算信息距离矩阵

print("D: ",D)
tau1 = 1.9
tau2 = 2.3
epson_init=1
epson2=0.5
result = SLLT(D, tau1=tau1, tau2=tau2, epson_init=epson_init, step=0.1, epson2=epson2)


edges_hat = result["edges"]  # 获取边列表
TreeEs = result["tree"]      # 获取估计的树结构
epson = result["epson"]      # 获取阈值参数

# print(result)

#利用结构学习得到的edges_hat构建ModelEs
ModelEs = pd.DataFrame({
    "vertice": edges_hat["nodes"],
    "vertice_type": [None] * len(edges_hat),
    "father_vertice": edges_hat["parent"]
})
# 正常设置索引为 edges_hat["nodes"]
ModelEs.index = edges_hat["nodes"]
ModelEs.index.name = None

# 如果 data_obs 原本是 DataFrame，可以这样转换
if isinstance(data_obs, pd.DataFrame) or isinstance(data_obs, pd.Series):
    data_obs = data_obs.to_numpy()

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

print("ModelEs:")
print(ModelEs)

# 日志文件及字段
log_file = "frog_model_run_log.csv"
log_columns = [
    "timestamp",
    "tau1",
    "tau2",
    "epson_init",
    "epson2",
    "n_observed",
    "n_latent"
]

# 如果日志文件不存在，先写入表头
if not os.path.exists(log_file):
    pd.DataFrame(columns=log_columns).to_csv(log_file, index=False)

# 计算 observed 和 latent 变量的数量
n_latent = (ModelEs["vertice_type"] == "latent").sum()
n_observed = ModelEs.shape[0] - n_latent

# 构造一行日志
log_entry = {
    "timestamp": datetime.now().isoformat(sep=" ", timespec="seconds"),
    "tau1": tau1,
    "tau2": tau2,
    "epson_init": epson_init,
    "epson2": epson2,
    "n_observed": n_observed,
    "n_latent": n_latent
}

# 追加写入 CSV
pd.DataFrame([log_entry]).to_csv(log_file, mode="a", header=False, index=False)

print(f"Logged run: {log_entry}")

ModelEs_renum=renumber_latents_df(ModelEs) #对隐变量重新排号，主要是把根节点定为H1，然后生成相应子节点

ModelEs_renum.to_csv("frog_ModelEs_renum.csv", index=False, header=True)

# # 调用 PEMLT 进行参数估计
para_hat = PEMLT(data_obs, ModelEs_renum, tol=0.0001)

print("para_hat: ",para_hat)

##############传播计算#############################################
#重新参数化，让意义更加明确一些
params = {}
for node, row in ModelEs_renum.iterrows():
    vtype = row["vertice_type"]
    mat   = para_hat[node]

    if vtype == "latent":
        #—— 处理潜变量
        if node == "H1":
            # 根节点：para_hat["H1"] 是 P(H1=1)
            p1 = float(mat)
            params[node] = {
                "states": [0,1],
                "prior":  np.array([1-p1, p1])
            }
        else:
            # 其它潜变量：CPT P(node | parent)
            params[node] = {
                "states": [0,1],
                "cpt":    np.array(mat)   # shape (2,2)
            }

    elif vtype == "observable_discrete":
        #—— 离散观测节点：CPT P(obs | parent)
        params[node] = {
            "states": [0,1],
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
    
print("np.mean(data_obs): ",np.mean(data_obs, axis=0))
print("np.std(data_obs): ",np.std(data_obs, axis=0, ddof=1))

marginal_prob=infer_latents_batch(ModelEs_renum, params, data_obs)

df = pd.DataFrame(marginal_prob)   # list of dict → DataFrame
# 把 DataFrame 保存到当前目录下的文件 marginal_prob.csv
df.to_csv('frog_marginal_prob.csv', index=False)

print(df.head(5))


# #### 以下为聚类，请单独调用
#
# num_runs = 1
# # num_cluster_new为整数值或者'2**h'
# num_cluster_new = 10
# for seed in range(42,52):
#     random_seed = seed
#     data_obs = pd.read_csv(f"../result/frog/frog_data_obs_{seed}_1%.csv", index_col='id')
#     model_es = pd.read_csv(f"../result/frog/frog_ModelEs_renum.csv")
#     marginal_prob = pd.read_csv(f"../result/frog/frog_marginal_prob_{seed}_1%.csv", index_col='id')
#     true_labels_df = pd.read_csv(f"../result/frog/frog_true_labels_{seed}_1%.csv", index_col='id')
#     true_labels = true_labels_df['true_label'].values
#     print('true_labels', true_labels)
#     cluster_main1(data_obs, model_es, marginal_prob, true_labels, num_cluster_new, num_runs, random_seed)