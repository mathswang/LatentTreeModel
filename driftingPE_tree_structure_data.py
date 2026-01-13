import os
import numpy as np
import pandas as pd
import time
from cluster.cluster_methods import cluster_main1
from tree.online_SL_functions_for_realdata import (parameter_generation, data_generation, data_choose, information_distance,
                                              information_distance_online, SLLT, tree_from_edge, RFdist)
from tree.online_PE_functions import (tree_from_edge, PEMLT_online, parameter_generation_online, adjust_para_hat,
                                 renumber_latents_df,
                                 data_generation, data_choose, loss_fun)
from tree.BF_functions_for_tree import infer_latents_batch

import itertools
from tree.online_SL_functions_for_realdata import compute_true_moments_from_para



# 设置工作目录
# os.chdir("your working directory")
# =====================================开始计时=====================================
start_time = time.time()  # Record the start time

#######################################Model#############################################
# 对应正文中Figure 1(a)

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

#  打印结果
print("model:")
print(model)

# =====================================生成参数=====================================
np.random.seed(123)
N_sim = 1  # 重复次数

p_dim = ((model['vertice_type'] == 'observable_continuous') | (model['vertice_type'] == 'observable_discrete')).sum()

# 循环不同的样本大小
sample_sizes = [500]  # 示例样本大小
save_dir = f'./results/PE尝试/{sample_sizes[0]}/dyn200-lam0.5-2-----------------'  # 你的存储路径，指定文件名，若没有，代码可以自动生成
os.makedirs(save_dir, exist_ok=True)

# 新增：用于存储前一个sample_size各t值对应的旧数据
prev_t_data = {}  # 结构：{t: 旧数据数组}


all_true_para_rows = []   # 全部 t 的真实参数累积在这里
for idx, sample_size in enumerate(sample_sizes):
    # true_moment_rows = []  # 用来存当前 sample_size 下所有 t 的真实矩
    moment_rows = []  # 存当前 sample_size 下所有 t 的“真实矩 + online 矩”
    # 信息距离中使用
    EX_inform_online = np.zeros(p_dim, dtype=np.float32)
    EX2_inform_online = np.zeros((p_dim, p_dim), dtype=np.float32)

    # PEMLT函数中使用，用于增量计算
    EXYZ11_online = np.zeros((p_dim, p_dim, p_dim), dtype=np.float32)
    EXYZ12_online = np.zeros((p_dim, p_dim, p_dim), dtype=np.float32)
    EXYZ21_online = np.zeros((p_dim, p_dim, p_dim), dtype=np.float32)
    EXYZ22_online = np.zeros((p_dim, p_dim, p_dim), dtype=np.float32)
    EXY11_online = np.zeros((p_dim, p_dim), dtype=np.float32)
    EXY12_online = np.zeros((p_dim, p_dim), dtype=np.float32)
    EXY21_online = np.zeros((p_dim, p_dim), dtype=np.float32)
    EXY22_online = np.zeros((p_dim, p_dim), dtype=np.float32)
    EX_online = np.zeros(p_dim, dtype=np.float32)
    EX2_online = np.zeros(p_dim, dtype=np.float32)

    RF_result = []  # 创建一个空列表，存储RF距离
    # out_filename = os.path.join(save_dir, f'RF_distance_result_model3-n{sample_size}.csv')
    out_filename = os.path.join(save_dir, 'RF_dis.csv')

    # 计算当前与前一个sample_size的差值（第一个sample_size时delta为sample_size本身）
    if idx == 0:
        delta = sample_size
        prev_sample_size = 0
    else:
        prev_sample_size = sample_sizes[idx - 1]
        delta = sample_size - prev_sample_size
        assert delta > 0, "sample_sizes应按升序排列且严格递增"

    for t in np.arange(0, 1, 1):
        print(f"处理sample_size: {sample_size}, t: {t}")
        # out_filename1 = os.path.join(save_dir, f'SL_clustering_stats_{sample_size}_{t}.csv')
        out_filename1 = os.path.join(save_dir, f'clustering_result_{t}.csv')

        para, mu0, mu1 = parameter_generation_online(model, t=t, N_sim=N_sim)
        # print(f"时间{t}时刻的参数:{para}")

        # ======== 新增：根据真实参数计算总体均值和二阶原点矩 ========
        means_true, seconds_true = compute_true_moments_from_para(model, para)

        # # 按观测节点顺序（1,2,3,...）保存
        # obs_mask = model["vertice_type"].str.startswith("observable")
        # obs_nodes = model.loc[obs_mask, "vertice"].tolist()
        # # 保证按数字排序
        # obs_nodes_sorted = sorted(obs_nodes, key=lambda x: int(x))

        # for v in obs_nodes_sorted:
        #     moment_rows.append({
        #         "t": float(t),
        #         "node": v,
        #         "E_X": means_true[v],
        #         "E_X2": seconds_true[v]
        #     })

        # ===========================================================



        # # ---------------- 保存真实参数 para 到 CSV ----------------
        # true_rows = []
        # for node, val in para.items():
        #     arr = np.array(val).reshape(-1)
        #     row = [node] + arr.tolist()
        #     true_rows.append(row)
        #
        # df_true = pd.DataFrame(true_rows)
        # true_path = os.path.join(save_dir, f"true_para_t{t}.csv")
        # df_true.to_csv(true_path, index=False, header=False, encoding="utf-8-sig")
        #
        #


        # # ----------------------------------------------------------

        # ================== 追加真实参数到总列表 ==================
        for node, val in para.items():
            arr = np.array(val).reshape(-1)
            all_true_para_rows.append(
                {"t": float(t), "node": node, **{f"p{i}": arr[i] for i in range(len(arr))}}
            )
        # =========================================================

        # 生成数据逻辑（关键修改部分）
        if idx == 0:  # 第一个sample_size：生成全新数据
            np.random.seed(123 + int(t))  # 初始种子保证可复现
            data = data_generation(i=0, model=model, para_all=para, n=sample_size)
            prev_t_data[t] = data  # 保存当前t的原始数据用于后续复用

        else:  # 后续sample_size：复用旧数据+生成补充数据
            # 获取前一个sample_size对应t的旧数据
            old_data = prev_t_data[t]
            # 生成补充数据（使用不同种子避免重复）
            # print("idx:", idx)
            np.random.seed(123 + int(t))  # 基于索引和t生成唯一种子
            new_data = data_generation(i=0, model=model, para_all=para, n=delta)
            # 合并旧数据和新数据（注意：需确保数据结构一致）
            data = np.concatenate([old_data, new_data], axis=0)
            # 更新缓存数据用于下一个sample_size
            prev_t_data[t] = data

        data_obs = data_choose(data, model)  # 提取观测数据

        # 以下为原有的信息距离计算、结构恢复、RF距离计算逻辑
        # 结合t-1 时刻的样本信息，计算t时刻的信息距离矩阵
        # D, EX_inform_online, EX2_inform_online = information_distance_online(
        #     data_obs, EX_inform_online, EX2_inform_online, lam=0.5 * (1 - np.exp(-t)))

        # D, EX_inform_online, EX2_inform_online = information_distance_online(
        #     data_obs, EX_inform_online, EX2_inform_online, lam= np.exp(-t))

        # D, EX_inform_online, EX2_inform_online = information_distance_online(
        #     data_obs, EX_inform_online, EX2_inform_online, lam=np.exp(-3))

        D, EX_inform_online, EX2_inform_online = information_distance_online(
            data_obs, EX_inform_online, EX2_inform_online, lam=0.5)

        # 仅结合t时刻的样本信息，计算t时刻的信息距离矩阵
        # D = information_distance(data_obs)  # 计算信息距离矩阵


        # ======== 新增：把 true 矩 + online 矩 写入 moment_rows ========

        # 1) 观测节点名称（与 data_obs 列顺序一致）
        obs_mask = model["vertice_type"].str.startswith("observable")
        obs_nodes = model.loc[obs_mask, "vertice"].tolist()
        # 根据你当前的 model，1~13 都是观测，且 data_choose 保持原顺序，这样排序后与列顺序一致
        obs_nodes_sorted = sorted(obs_nodes, key=lambda x: int(x))

        # 2) 在线估计的一阶、二阶原点矩
        online_E_X = EX_inform_online              # shape = (p_dim,)
        online_E_X2 = np.diag(EX2_inform_online)   # shape = (p_dim,)

        # （可选）当前时间点的样本一阶、二阶原点矩，如果你也想顺便存进去：
        sample_E_X = data_obs.mean(axis=0)
        sample_E_X2 = (data_obs ** 2).mean(axis=0)

        # 简单一致性检查（可选）
        assert len(obs_nodes_sorted) == data_obs.shape[1] == online_E_X.shape[0]

        for j, v in enumerate(obs_nodes_sorted):
            moment_rows.append({
                "t": float(t),
                "node": v,
                # 真实矩（根据 para 精算出来的总体矩）
                "true_E_X":  means_true[v],
                "true_E_X2": seconds_true[v],
                # 在线估计矩（EX_inform_online / EX2_inform_online）
                "online_E_X":  float(online_E_X[j]),
                "online_E_X2": float(online_E_X2[j]),
                # 如果你暂时不想要样本矩，把下面两行删掉即可
                "sample_E_X":  float(sample_E_X[j]),
                "sample_E_X2": float(sample_E_X2[j]),
            })
        # ==============================================================

        tau1 = 3.0
        tau2 = 5.0
        result = SLLT(D, tau1=tau1, tau2=tau2)
        edges_hat = result["edges"]
        TreeEs = result["tree"]
        epson = result["epson"]

        TreeTr = tree_from_edge(model.drop(columns=["vertice_type"]))
        RF_dist = RFdist(TreeTr, TreeEs)
        RF_result.append([t, RF_dist])

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

        ModelEs_renum = renumber_latents_df(ModelEs)  # 对隐变量重新排号
        ModelEs_renum_path = os.path.join(save_dir, f"ModelEs_renum_{t}.csv")
        ModelEs_renum.to_csv(ModelEs_renum_path, index=False, header=True)

        # # 调用 PEMLT 进行参数估计
        # EXYZ_online，EXY_online，EX_online 是否已经完成更新，初始值为0，一但在PEMLT_online取值为1,表示已经更新完毕
        EXYZ_update = np.zeros((p_dim, p_dim, p_dim), dtype=int)
        EXY_update = np.zeros((p_dim, p_dim), dtype=int)
        EX_update = np.zeros(p_dim, dtype=int)

        # 结合t-1 时刻的样本信息，计算t时刻的模型参数
        para_hat = PEMLT_online(data_obs, D, ModelEs_renum, EXYZ11_online, EXYZ12_online, EXYZ21_online, EXYZ22_online,
                                EXY11_online, EXY12_online, EXY21_online, EXY22_online, EX2_online, EX_online,
                                EXYZ_update, EXY_update, EX_update, tol)
        # print(f"时刻{t}的参数估计{para_hat}")
        # 仅结合t时刻的样本信息，计算t时刻的模型参数
        # para_hat = PEMLT(data_obs, ModelEs_renum, tol=0.0001)


        # # -------- 调整 para_hat 的维度 --------
        # para_hat_adj = adjust_para_hat(para_hat, ModelEs_renum)
        #
        # # -------- 重新排序：先观测节点 (1~13)，再 H1,H2,... --------
        #
        # # 观测节点：结点名为数字的
        # obs_nodes = sorted(
        #     [node for node in para_hat_adj.keys() if node.isdigit()],
        #     key=lambda x: int(x)
        # )
        #
        # # latent 节点：结点名以 H 开头的
        # latent_nodes = sorted(
        #     [node for node in para_hat_adj.keys() if node.startswith("H")],
        #     key=lambda x: int(x[1:])
        # )
        #
        # # 合并顺序：先观测节点，再 latent 节点
        # ordered_nodes = obs_nodes + latent_nodes
        #
        # # -------- 保存为 CSV --------
        # est_rows = []
        # for node in ordered_nodes:
        #     val = para_hat_adj[node]
        #     arr = np.array(val).reshape(-1)
        #     row = [node] + arr.tolist()
        #     est_rows.append(row)
        #
        # df_est = pd.DataFrame(est_rows)
        # est_path = os.path.join(save_dir, f"est_para_t{t}.csv")
        # df_est.to_csv(est_path, index=False, header=False, encoding="utf-8-sig")

        #################################聚类#############################################
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

        # 假设 D 是一个 NumPy 数组
        df_data_obs = pd.DataFrame(data_obs)
        # 保存为 CSV 文件
        df_data_obs_path = os.path.join(save_dir, f"data_obs_{t}.csv")
        df_data_obs.to_csv(df_data_obs_path, index=True,
                                 index_label='id', header=True)

        # 使用抽样后的数据计算边际概率marginal_prob
        marginal_prob = infer_latents_batch(ModelEs_renum, params, data_obs)
        marginal_prob = pd.DataFrame(marginal_prob)  # list of dict → DataFrame
        marginal_prob.insert(0, 'id', range(len(marginal_prob)))
        marginal_prob_path = os.path.join(save_dir, f"marginal_prob_{t}.csv")
        marginal_prob.to_csv(marginal_prob_path, index=False, header=True)

        #  利用抽样数据、边际概率、模型结构信息（ModelEs_renum）聚类
        true_labels = []
        cluster_result = cluster_main1(df_data_obs, ModelEs_renum, marginal_prob, true_labels, 256, 10, random_seed=42)
        cluster_result.to_csv(out_filename1, encoding='utf-8-sig', index=False)

    # #===============================保存真实一阶和二阶原点矩=======================================
    # # 在每个 sample_size 结束后，把真实矩写到一个 csv
    # df_true_moments = pd.DataFrame(moment_rows)
    # true_moment_path = os.path.join(save_dir, f"true_moments_sample{sample_size}.csv")
    # df_true_moments.to_csv(true_moment_path, index=False, encoding="utf-8-sig")
    # print(f"真实总体均值和二阶原点矩已保存到: {true_moment_path}")
    # # ==================================================================================


    #===============================保存真实/online/样本一阶和二阶原点矩=======================================
    df_moments = pd.DataFrame(moment_rows)
    moments_path = os.path.join(save_dir, f"moments_sample{sample_size}.csv")
    df_moments.to_csv(moments_path, index=False, encoding="utf-8-sig")
    print(f"真实/online/样本的总体均值和二阶原点矩已保存到: {moments_path}")
    # ==================================================================================

    # ================================保存所有时刻的真实参数===================================
    df_all_true = pd.DataFrame(all_true_para_rows)
    true_all_path = os.path.join(save_dir, f"true_para_all_t_{sample_size}.csv")
    df_all_true.to_csv(true_all_path, index=False, encoding="utf-8-sig")
    print(f"所有时刻的真实参数已保存到: {true_all_path}")
    # ==================================================================================

    df_RF_result = pd.DataFrame(RF_result, columns=['时间演化t', 'RF距离'])
    df_RF_result.to_csv(out_filename, encoding='utf-8-sig', index=False)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"总运行时间: {elapsed_time:.2f} seconds")
