import os
import csv
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from tree.SL_functions_for_realdata import (parameter_generation, data_generation, data_choose,
                                       information_distance, SLLT, tree_from_edge, RFdist)
from tree.PE_functions_for_realdata import (tree_from_edge, PEMLT, parameter_generation, adjust_para_hat, renumber_latents_df,
                                       data_generation, data_choose, loss_fun)
from tree.BF_functions_for_tree import infer_latents_batch
from cluster.cluster_methods import  cluster_main1
import time
import json


# def generate_mixture_gaussian_data(n_samples, n_features):
#     # 定义n_components个高斯分量的混合模型
#     n_components = 32
#
#     # 为n个分量分配权重（总和为1）
#     weights = np.random.dirichlet(np.ones(n_components))
#
#     # 生成初始随机点
#     initial_points = np.random.randn(n_components * 10, n_features)
#
#     # 使用KMeans聚类确保均值之间有足够距离
#     from sklearn.cluster import KMeans
#     kmeans = KMeans(n_clusters=n_components, n_init=10)
#     kmeans.fit(initial_points)
#     means = kmeans.cluster_centers_
#
#     # 对均值进行缩放，增加分布之间的距离
#     scale_factor = 4  # 可调整此参数控制均值之间的距离
#     means = [mean * scale_factor for mean in means]
#
#     # 每个分量的协方差矩阵（保持为对角阵以简化）
#     covs = [np.eye(n_features) * (0.5 + np.random.rand()) for _ in range(n_components)]
#
#     # 生成数据及对应的真实标签
#     data = np.zeros((n_samples, n_features))
#     true_labels = np.zeros(n_samples, dtype=int)  # 存储每个样本的真实高斯分量索引
#
#     for i in range(n_samples):
#         # 选择当前样本所属的高斯分量
#         component = np.random.choice(n_components, p=weights)
#         true_labels[i] = component  # 保存真实标签
#
#         # 从选中的高斯分量生成样本
#         data[i] = multivariate_normal.rvs(mean=means[component], cov=covs[component])
#
#     # 存储生成参数
#     params = {
#         "n_samples": n_samples,
#         "n_features": n_features,
#         "n_components": n_components,
#         "weights": weights.tolist(),
#         "means": [mean.tolist() for mean in means],
#         "covs": [cov.tolist() for cov in covs]
#     }
#
#     # 保存参数到JSON文件
#     with open("mixture_gaussian_params.json", "w") as f:
#         json.dump(params, f)
#
#     # 保存真实标签到文件
#     np.save("true_labels_0509test.npy", true_labels)
#
#     return data, true_labels  # 返回数据和真实标签
def generate_mixture_gaussian_data(n_samples, n_features):
    SEED = 42
    rng = np.random.default_rng(SEED)

    n_components = 32
    weights = rng.dirichlet(np.ones(n_components))

    # 初始随机点
    initial_points = rng.standard_normal((n_components * 10, n_features))

    # KMeans：固定随机态
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_components, n_init=10, random_state=SEED)
    kmeans.fit(initial_points)
    means = kmeans.cluster_centers_

    scale_factor = 4
    means = [mean * scale_factor for mean in means]

    covs = [np.eye(n_features) * (0.5 + rng.random()) for _ in range(n_components)]

    data = np.zeros((n_samples, n_features))
    true_labels = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        component = rng.choice(n_components, p=weights)
        true_labels[i] = component
        data[i] = multivariate_normal.rvs(mean=means[component], cov=covs[component], random_state=rng)

    params = {
        "seed": SEED,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_components": n_components,
        "weights": weights.tolist(),
        "means": [mean.tolist() for mean in means],
        "covs": [cov.tolist() for cov in covs]
    }
    with open("gmm_mixture_gaussian_params.json", "w") as f:
        json.dump(params, f)

    np.save("gmm_true_labels.npy", true_labels)

    # 读取 .npy 文件
    true_labels1 = np.load("gmm_true_labels.npy")
    # 转成 DataFrame 并加上列名
    true_labels2 = pd.DataFrame(true_labels1, columns=["true_label"])
    # 添加 id 列（从 1 开始）
    true_labels2.insert(0, "id", range(0, len(true_labels2)))
    # 保存为 CSV 文件
    true_labels2.to_csv("gmm_true_labels.csv", index=False, encoding="utf-8-sig")

    return data, true_labels



# 生成数据
data_obs, true_labels = generate_mixture_gaussian_data(n_samples=10000, n_features=20)
# 假设 D 是一个 NumPy 数组
df_data_obs = pd.DataFrame(data_obs)
# 保存为 CSV 文件，设置索引名为 'id'
df_data_obs.to_csv("gmm_data_obs.csv", index=True, index_label='id', header=True)
# 打印 data_obs 的前5行
# print(pd.DataFrame(data_obs).head(5))
t=0
####===================================计算信息距离===================================
D = information_distance(data_obs)  # 计算信息距离矩阵
# 假设 D 是一个 NumPy 数组
df_D = pd.DataFrame(D)
print(D)
# =====================================结构恢复=====================================
tau1 = 4
tau2 = 3.3
result = SLLT(D, tau1=tau1, tau2=tau2)
# 假设 result 返回一个列表：
# result[0]: edges_hat —— 边列表，父节点为 "" 表示根节点
# result[1]: TreeEs   —— 估计的树结构
# result[2]: epson    —— 阈值参数

edges_hat = result["edges"]  # 获取边列表
TreeEs = result["tree"]  # 获取估计的树结构
epson = result["epson"]  # 获取阈值参数
print(edges_hat)

# # # ========================= 参数估计函数 PEMLT =========================
# 利用结构学习得到的edges_hat构建ModelEs
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

print(ModelEs)
ModelEs_renum = renumber_latents_df(ModelEs)  # 对隐变量重新排号

ModelEs_renum.to_csv("gmm_ModelEs_renum.csv", index=False, header=True)

# # 调用 PEMLT 进行参数估计
para_hat = PEMLT(data_obs, ModelEs_renum, tol=0.0001)

print("para_hat: ")
print(para_hat)
#################################传播计算#############################################

# 重新参数化，让意义更加明确一些
params = {}
for node, row in ModelEs_renum.iterrows():
    vtype = row["vertice_type"]
    mat = para_hat[node]

    if vtype == "latent":
        # —— 处理潜变量
        if node == "H1":
            # 根节点：para_hat["H1"] 是 P(H1=1)
            p1 = float(mat)
            params[node] = {
                "states": [0, 1],
                "prior": np.array([1 - p1, p1])
            }
        else:
            # 其它潜变量：CPT P(node | parent)
            params[node] = {
                "states": [0, 1],
                "cpt": np.array(mat)  # shape (2,2)
            }

    elif vtype == "observable_discrete":
        # —— 离散观测节点：CPT P(obs | parent)
        params[node] = {
            "states": [0, 1],
            "cpt": np.array(mat)  # shape (2,2)
        }

    elif vtype == "observable_continuous":
        # —— 连续观测节点：第一行是均值，第二行是方差
        means = mat[0, :]  # μ|parent=0, μ|parent=1
        vars_ = mat[1, :]  # σ²|parent=0, σ²|parent=1
        params[node] = {
            "mean": means,  # 1D array length=2
            "var": vars_  # 1D array length=2
        }

    else:
        raise ValueError(f"未知 vertice_type: {vtype}")

marginal_prob = infer_latents_batch(ModelEs_renum, params, data_obs)

marginal_prob = pd.DataFrame(marginal_prob)  # list of dict → DataFrame
# 添加 id 列
marginal_prob.insert(0, 'id', range(len(marginal_prob)))
marginal_prob.to_csv("gmm_marginal_prob.csv", index=False, header=True)


# 删除 id 列（若存在）
if 'id' in marginal_prob.columns:
    marginal_prob.drop(columns=['id'], inplace=True)

sl_i = 0
# 创建真实标签DataFrame
true_labels_df = pd.DataFrame({'true_label': true_labels})

# 调用修正后的函数
cluster_result = cluster_main1(
    df_data_obs,
    ModelEs_renum,
    marginal_prob,
    true_labels_df,  # 传递DataFrame格式的true_labels
    "2**h",          # num_cluster_new参数
    10,  # num_runs参数
    random_seed=42

)

#构建文件名
# filename = 'gmm_SL_clustering_stats_tau1_4_tau2_3.3.csv'
filename = 'gmm_cluster.csv'
#保存结果到 CSV 文件
cluster_result.to_csv(filename, index=False, encoding='utf-8-sig')

