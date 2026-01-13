import math
import networkx as nx
from itertools import combinations
import numpy as np
import pandas as pd
import gc
import re

def make_model(lat_num, obs_num=None, cont_rate=0.5):
    """
    根据隐节点数量、可观测节点数量生成隐树模型
    输入:
      lat_num: 隐节点数量 (>=10)
      obs_num: 可观测节点数量 (默认为 lat_num * 3, 且应 >= lat_num*2)
      cont_rate: 连续节点比例 (0~1)
    输出:
      model: pandas DataFrame，包含三列：
             'vertice'：节点名称（可观测节点为"1", "2", ...；隐节点为 "H1", "H2", ...）
             'vertice_type': 节点类型 ("observable_continuous", "observable_discrete", "latent")
             'father_vertice': 父节点名称，初始均为空字符串，但后续根据层次进行赋值
    """
    if obs_num is None:
        obs_num = lat_num * 3
    # 构造所有节点：前 obs_num 个为可观测节点，后 lat_num 个为隐节点
    vertices = [str(i) for i in range(1, obs_num + 1)] + [f"H{i}" for i in range(1, lat_num + 1)]
    
    # 计算连续节点和离散节点数
    cont_nodes_num = int(obs_num * cont_rate)  # 向下取整
    disc_nodes_num = obs_num - cont_nodes_num
    
    # 构造节点类型向量
    vertice_type = (["observable_continuous"] * cont_nodes_num +
                    ["observable_discrete"] * disc_nodes_num +
                    ["latent"] * lat_num)
    # 初始化父节点，全为空字符串
    father_vertice = ["" for _ in range(obs_num + lat_num)]
    
    # 构造 DataFrame，行索引设为 vertices
    model = pd.DataFrame({
        "vertice": vertices,
        "vertice_type": vertice_type,
        "father_vertice": father_vertice
    })
    model.index = vertices

    # -------------------- 隐节点父节点分配 --------------------
    # 计算第2层隐节点数量 m1 = floor(sqrt(lat_num-1))
    m1 = int(math.sqrt(lat_num - 1))
    # 第3层隐节点数量 m2：初始每个为 m1 - 1，共 m1 个
    m2 = [m1 - 1] * m1
    temp1 = (lat_num - m1 * m1 - 1) // m1
    temp2 = (lat_num - m1 * m1 - 1) % m1
    # 对 m2 每个元素增加 temp1，并对前 temp2 个再加 1
    m2 = [val + temp1 + (1 if i < temp2 else 0) for i, val in enumerate(m2)]
    
    # 隐节点在模型中位于行 obs_num 到 obs_num+lat_num-1（0-indexed）
    # 第二层隐节点：对应于模型行 obs_num+1 到 obs_num+m1-1 （注意：第一个隐节点保持空父节点）
    for j in range(obs_num + 1, obs_num +1+ m1):
        model.at[vertices[j], "father_vertice"] = "H1"
    
    # 第三层隐节点：对应于剩下的隐节点
    a1 = 0
    a2 = 0
    for i in range(m1):
        a2 += m2[i]
        # 第三层隐节点所在行：从 obs_num + m1 + a1 到 obs_num + m1 + a2 - 1
        for j in range(obs_num +1+ m1 + a1, obs_num+1 + m1 + a2):
            model.at[vertices[j], "father_vertice"] = f"H{i+2}"  # i=0对应 H2, i=1对应 H3, etc.
        a1 += m2[i]
    
    
    # -------------------- 可观测节点父节点分配 --------------------
    if lat_num < 10:
        raise ValueError("隐节点太少，需手动设定!")

    n1 = obs_num // lat_num
    n2 = obs_num % lat_num

    print("n1: ",n1,"n2: ",n2)

    if n1 < 2:
        total = sum(m2)
        n1 = obs_num // total
        n2 = obs_num % total

        if n1 < 2:
            raise ValueError("可观测节点太少，需手动设定!")
        else:
            # 每个隐节点的可观测子节点数
            obs_ch_num = [0] * lat_num
            temp_list = [0] + [0] * m1 + [1] * n2 + [0] * (total - n2)
            obs_ch_num = [obs_ch_num[i] + ([0] * (m1 + 1) + [n1] * total)[i] + temp_list[i] for i in range(lat_num)]

            b1 = 0
            b2 = 0
            for i in range(m1 + 1, m1 + 1 + total):
                b2 += obs_ch_num[i]
                if obs_ch_num[i] != 0:
                    for j in range(b1, b2):
                        model.at[str(j + 1), "father_vertice"] = f"H{i+1}"
                b1 += obs_ch_num[i]
    else:
        # 每个隐节点的可观测子节点数
        obs_ch_num = [0] * lat_num
        temp_list = (
            [0]
            + [1] * max(n2 - sum(m2), 0)
            + [0] * (m1 - max(n2 - sum(m2), 0))
            + [1] * min(n2, sum(m2))
            + [0] * max(0, sum(m2) - n2)
        )
        obs_ch_num = [n1 + temp_list[i] for i in range(lat_num)]

        b1 = 0
        b2 = 0
        for i in range(lat_num):
            b2 += obs_ch_num[i]
            for j in range(b1, b2):
                model.at[str(j + 1), "father_vertice"] = f"H{i+1}"
            b1 += obs_ch_num[i]
    # if lat_num < 10:
    #     raise ValueError("隐节点太少，需手动设定!")
    # else:
    #     n1 = obs_num // lat_num
    #     n2 = obs_num % lat_num
    #     if n1 < 2:
    #         total = sum(m2)
    #         n1 = obs_num // total
    #         n2 = obs_num % total
    #         if n1 < 2:
    #             raise ValueError("可观测节点太少，需手动设定!")
    #         else:
    #             # 构造每个隐节点的可观测子节点数向量 obs_ch_num，长度为 (m1+1+total)
    #             base = [0] * (m1 + 1) + [n1] * total
    #             temp_list = [0] + [0] * m1 + [1] * n2 + [0] * (total - n2)
    #             obs_ch_num_full = [base[i] + temp_list[i] for i in range(len(base))]
    #             # 仅取前 lat_num 个数
    #             obs_ch_num = obs_ch_num_full[:lat_num]
    #             b1 = 0
    #             b2 = 0
    #             # i 从 m1+2 到 m1+1+total，转换为 0-index：i runs over range(m1+1, m1+1+total)
    #             for idx, i in enumerate(range(m1+1, m1+1+total)):
    #                 b2 += obs_ch_num[idx]
    #                 if obs_ch_num[idx] != 0:
    #                     # 分配父节点给可观测节点：前 obs_num 行中的索引 b1 到 b2-1
    #                     for j in range(b1, b2):
    #                         model.at[vertices[j], "father_vertice"] = f"H{i}"
    #                 b1 += obs_ch_num[idx]
    #     else:
    #         obs_ch_num = [0] * lat_num
    #         # 构造 temp_list：长度为 lat_num
    #         part1 = [1] * max(n2 - sum(m2), 0)
    #         part2 = [0] * (m1 - max(n2 - sum(m2), 0))
    #         part3 = [1] * min(n2, sum(m2))
    #         part4 = [0] * max(0, sum(m2) - n2)
    #         temp_list = [0] + part1 + part2 + part3 + part4
    #         # 取前 lat_num 个
    #         obs_ch_num = [n1 + temp_list[i] for i in range(lat_num)]
    #         b1 = 0
    #         b2 = 0
    #         for i in range(lat_num):
    #             b2 += obs_ch_num[i]
    #             # 为可观测节点分配父节点，分组顺序为 i从0到lat_num-1，对应父节点 "H" + str(i+1)
    #             for j in range(b1, b2):
    #                 model.at[vertices[j], "father_vertice"] = f"H{i+1}"
    #             b1 += obs_ch_num[i]
    
    return model

def parameter_generation(model, N_sim=100):
    """
    根据具体的 model 生成概率分布的参数，重复 N_sim 次。
    
    输入:
      model: pandas DataFrame，模型数据（应包含列 'vertice', 'vertice_type', 'father_vertice'）
      N_sim: 模拟重复次数 (默认100)
    
    输出:
      para: 字典，键为 model 中每个节点（'vertice'）的名称，
            值为对应的参数数组：
              - 对于连续节点 ("observable_continuous"): 生成一个形状为 (4, N_sim) 的数组，
                行依次为 [mu0, mu1, sigma0, sigma1]，小数点保留1位（mu）和2位（sigma）。
              - 对于离散节点（包括 "latent" 和 "observable_discrete"）：如果节点没有父节点
                (即 father_vertice == ""), 生成一维数组 (长度 N_sim)（即 p_1），
                否则生成形状为 (2, N_sim) 的数组，行依次为 [p_{v|0}, p_{v|1}],
                所有数值均保留2位小数。
    """
    # 记录总节点数 p_t 和可观测节点数 p_o
    p_t = model.shape[0]
    # 可观测变量：observable_continuous 和 observable_discrete
    logi2 = model['vertice_type'] == "observable_discrete"
    logi3 = model['vertice_type'] == "observable_continuous"
    p_o = logi2.sum() + logi3.sum()
    
    # 获取各类型节点的索引（这里用行标签或位置均可）
    # latent: 隐节点； obs_con: 观测连续； obs_dis: 观测离散
    latent_idx = model.index[model['vertice_type'] == "latent"].tolist()
    obs_con_idx = model.index[model['vertice_type'] == "observable_continuous"].tolist()
    obs_dis_idx = model.index[model['vertice_type'] == "observable_discrete"].tolist()
    
    # 初始化参数字典，键为每个节点名称（来自 model['vertice']）
    para = {}
    for v in model['vertice']:
        para[v] = None

    # 处理离散型节点（包括隐节点和观测离散节点）
    # 取 union(latent, obs_dis)
    discrete_mask = model['vertice_type'].isin(["latent", "observable_discrete"])
    for idx, row in model[discrete_mask].iterrows():
        v = row['vertice']
        # 如果父节点为空（根节点）
        if row['father_vertice'] == "":
            # 生成 N_sim 个随机数，均匀分布在 [0.4, 0.6]，保留2位小数
            a = np.round(np.random.uniform(0.4, 0.6, N_sim), 2)
            para[v] = a
        else:
            # 否则生成两个向量 a 和 b，均匀分布在 [0.1, 0.9]，保留2位小数
            a = np.round(np.random.uniform(0.1, 0.9, N_sim), 2)
            b = np.round(np.random.uniform(0.1, 0.9, N_sim), 2)
            c = np.abs(a - b)
            # 保证 a 与 b 的差异至少 0.3
            while np.sum(c < 0.3) > 0:
                idxs = np.where(c < 0.3)[0]
                a[idxs] = np.round(np.random.uniform(0.1, 0.9, len(idxs)), 2)
                b[idxs] = np.round(np.random.uniform(0.1, 0.9, len(idxs)), 2)
                c = np.abs(a - b)
            # 将 a 和 b 按行堆叠为 (2, N_sim) 数组
            para[v] = np.vstack([a, b])
    
    # 处理连续型节点（observable_continuous）
    for idx, row in model.loc[obs_con_idx].iterrows():
        v = row['vertice']
        # 生成 mu0, mu1 均匀分布在 [-2, 2]，保留1位小数
        mu0 = np.round(np.random.uniform(-2, 2, N_sim), 1)
        mu1 = np.round(np.random.uniform(-2, 2, N_sim), 1)
        diff = np.abs(mu0 - mu1)
        # 保证 mu0 与 mu1 的差至少 0.5
        while np.sum(diff < 0.5) > 0:
            idxs = np.where(diff < 0.5)[0]
            mu0[idxs] = np.round(np.random.uniform(-2, 2, len(idxs)), 1)
            mu1[idxs] = np.round(np.random.uniform(-2, 2, len(idxs)), 1)
            diff = np.abs(mu0 - mu1)
        # 生成 sigma0, sigma1 均匀分布在 [0.1, 1]，保留2位小数
        sigma0 = np.round(np.random.uniform(0.1, 1, N_sim), 2)
        sigma1 = np.round(np.random.uniform(0.1, 1, N_sim), 2)
        para[v] = np.vstack([mu0, mu1, sigma0, sigma1])
    
    return para

def data_generation(i, model, para_all, n):
    """
    根据具体的 model 和参数表 para_all 的第 i 组参数产生数据
    输入:
      i: 整数，参数索引 (0-based)
      model: pandas DataFrame，包含 'vertice', 'vertice_type', 'father_vertice'
      para_all: dict，键为节点名称，对应的参数数组（离散节点为一维或 (2,N_sim) 数组；连续节点为 (4,N_sim) 数组）
      n: 样本量 (整数)
    输出:
      data: numpy 数组，形状为 (n, p_t)
    """
    # 分别构造布尔掩码
    logi_latent = model['vertice_type'] == "latent"
    logi_obs_disc = model['vertice_type'] == "observable_discrete"
    logi_obs_cont = model['vertice_type'] == "observable_continuous"
    
    p_t = model.shape[0]
    # 获取各类节点的行号（0-based）
    latent = model.index[logi_latent].tolist()
    obs_con = model.index[logi_obs_cont].tolist()
    obs_dis = model.index[logi_obs_disc].tolist()
    
    # 初始化数据矩阵，n x p_t，填充 -1
    data = -np.ones((n, p_t))
    
    # 建立一个从节点名称到行索引的映射（假设模型中行顺序与列顺序一致）
    vertice_list = model['vertice'].tolist()
    vertice_to_index = {v: idx for idx, v in enumerate(vertice_list)}
    
    # 处理离散型节点（包括隐节点和观测离散节点）
    #自定义排序函数，按照数字大小排序
    def sort_numeric(x):
        if x.startswith("H"):
            return (1, int(x[1:]))
        else:
            return (0, int(x))

    discrete_nodes = sorted(latent, key=sort_numeric) + sorted(obs_dis, key=sort_numeric)


    # print("discrete_nodes",discrete_nodes)

    for node in discrete_nodes:
        j_idx = vertice_to_index[node]  # 使用映射获得整数索引
        v = model.loc[node, "vertice"]
        # print("node: ",node)
        # 如果父节点为空，则直接生成 p_1 数值
        if model.loc[node, "father_vertice"] == "":
            # print("node_father",model.loc[node, "father_vertice"])
            p_1 = para_all[v][i]  # 取第 i 个参数（scalar）
            # 生成 n 次 Bernoulli 试验，概率 p_1
            a = np.random.binomial(1, p_1, n)
            data[:, j_idx] = a
        else:
            # 否则，参数为 2 x N_sim 数组
            p_j = para_all[v][:, i]  # 得到形状 (2,)
            father_v = model.loc[node, "father_vertice"]
            father_idx = vertice_to_index[father_v]# 转换父节点名称到整数索引
            # 获取父节点在数据矩阵中的列索引
            father_idx = vertice_to_index[father_v]
            # 如果父节点数据未生成（第一行依然为 -1），给出警告
            if data[0, father_idx] == -1:
                print(f"警告：节点 {v} 的父节点 {father_v} 数据尚未生成！")
            # 根据父节点值判断使用 p_j 的哪一个参数
            # np.where：如果对应父节点值为 0，则取 p_j[0]；否则取 p_j[1]
            prob = np.where(data[:, father_idx] == 0, p_j[0], p_j[1])
            result = np.random.binomial(1, prob)
            data[:, j_idx] = result
    
    # 处理连续型节点（观测连续）
    for node in obs_con:
        j_idx = vertice_to_index[node]
        v = model.loc[node, "vertice"]
        p_j = para_all[v][:, i]  # 得到形状 (4,)
        father_v = model.loc[node, "father_vertice"]
        father_idx = vertice_to_index[father_v]
        if data[0, father_idx] == -1:
            print(f"警告：节点 {v} 的父节点 {father_v} 数据尚未生成！")
        # 根据父节点值判断使用 p_j 的哪一组参数：
        # 如果父节点值为 0，则 mu = p_j[0]，sigma = p_j[2]；否则 mu = p_j[1]，sigma = p_j[3]
        mu = np.where(data[:, father_idx] == 0, p_j[0], p_j[1])
        sigma = np.where(data[:, father_idx] == 0, p_j[2], p_j[3])
        # 对每个样本生成一个正态分布随机数（利用向量化生成 n 个样本）
        result = np.random.normal(mu, sigma)
        data[:, j_idx] = result
    
    # 调用垃圾回收（Python会自动处理，但可显式调用）
    import gc
    gc.collect()
    
    return data

def data_choose(data, model):
    """
    从 data 中提出可观测变量数据，去除隐节点所在的列。
    假设 model 的 'vertice_type' 列标记了节点类型，
    并且模型的行索引与 data 的列顺序对应。
    """
    # 取出所有隐节点对应的索引（假设 model.index 为节点名称）
    latent = model.index[model["vertice_type"] == "latent"].tolist()
    # 隐节点列：在 data 的 DataFrame 中删除这些列
    df = pd.DataFrame(data, columns=model.index)
    df = df.drop(columns=latent, errors='ignore')
    return df.values

def data_generation2(i, model, para_all, n, filename="data_generated.csv"):
    """
    根据具体的 model 和参数表 para_all 的第 i 组参数产生数据，
    并将结果写入 CSV 文件（若样本量较大，采用分批写入）。
    
    输入:
      i: 整数，参数索引（0-based），即使用 para_all 中第 i 组参数
      model: pandas DataFrame，包含列 'vertice', 'vertice_type', 'father_vertice'
      para_all: 字典，键为节点名称，与 model['vertice'] 对应，
                对于离散节点（latent 与 observable_discrete）：
                  - 如果父节点为空，则值为一维数组 (N_sim长度)；
                  - 否则为形状 (2, N_sim) 的数组.
                对于连续节点 (observable_continuous)：
                  - 值为形状 (4, N_sim) 的数组.
      n: 样本量 (整数)
      filename: 文件名，默认 "data_generated.csv"
    
    输出:
      data: 生成的最终数据（numpy 数组），可通过 data_choose 过滤隐节点。
    """
    # 计算相关指标
    p_t = model.shape[0]
    # 可观测变量数量 p_o = 观测连续 + 观测离散
    p_o = ((model["vertice_type"] == "observable_discrete").sum() +
           (model["vertice_type"] == "observable_continuous").sum())
    
    # 获取各类型节点的索引（0-based，假设 model 的行顺序与生成数据的列顺序一致）
    latent = model.index[model["vertice_type"] == "latent"].tolist()
    obs_con = model.index[model["vertice_type"] == "observable_continuous"].tolist()
    obs_dis = model.index[model["vertice_type"] == "observable_discrete"].tolist()
    
    # 如果样本量与可观测变量数量乘积较小，直接生成数据，否则分批生成
    if n * p_o <= 100000000:
        data = -np.ones((n, p_t))
        # 设置列顺序按照 model['vertice']（假设 model.index 顺序就是原来的顺序）
        # 处理离散节点（latent 和 observable_discrete）
        for j in sorted(set(latent + obs_dis), key=lambda x: model.index.get_loc(x)):
            v = model.loc[j, "vertice"]
            if model.loc[j, "father_vertice"] == "":
                # 父节点为空：生成一维参数数组
                p_1 = para_all[v][i]  # scalar
                a = np.random.binomial(1, p_1, n)
                data[:, model.index.get_loc(j)] = a
            else:
                p_j = para_all[v][:, i]  # 形状 (2,)
                father_v = model.loc[j, "father_vertice"]
                if father_v not in model.index:
                    print(f"警告：父节点 {father_v} 不在模型中！")
                    continue
                father_idx = model.index.get_loc(father_v)
                if data[0, father_idx] == -1:
                    print(f"警告：子节点 {v} 在父节点 {father_v} 之前产生！")
                # 根据父节点生成数据：若父节点值为0取 p_j[0]，否则 p_j[1]
                prob = np.where(data[:, father_idx] == 0, p_j[0], p_j[1])
                result = np.random.binomial(1, prob)
                data[:, model.index.get_loc(j)] = result

        # 处理连续节点（observable_continuous）
        for j in obs_con:
            v = model.loc[j, "vertice"]
            p_j = para_all[v][:, i]  # 形状 (4,)
            father_v = model.loc[j, "father_vertice"]
            if father_v not in model.index:
                print(f"警告：父节点 {father_v} 不在模型中！")
                continue
            father_idx = model.index.get_loc(father_v)
            if data[0, father_idx] == -1:
                print(f"警告：子节点 {v} 在父节点 {father_v} 之前产生！")
            mu = np.where(data[:, father_idx] == 0, p_j[0], p_j[1])
            sigma = np.where(data[:, father_idx] == 0, p_j[2], p_j[3])
            result = np.random.normal(mu, sigma)
            data[:, model.index.get_loc(j)] = result

        # 将列名设置为字符串 "1", "2", ..., "p_t"
        df = pd.DataFrame(data, columns=[str(x) for x in range(1, p_t+1)])
        gc.collect()
        # 过滤隐节点数据
        data_final = data_choose(df.values, model)
        gc.collect()
        df_final = pd.DataFrame(data_final)
        df_final.to_csv(filename, index=False)
        return data_final
    else:
        # 当 n*p_o 非常大时，分批生成数据
        n1 = 100000
        n2 = n % n1
        cycle_num = (n // n1) + (1 if n2 != 0 else 0)
        
        # 写入文件头（列属性名称，假设为 "1", "2", ..., "p_o"）
        header = ",".join([str(x) for x in range(1, p_t+1)])
        with open(filename, "w") as f:
            f.write(header + "\n")
            
        # 初始化累积数据（这里只返回最后批次生成的数据作为函数返回结果）
        final_data = None
        for iter_num in range(cycle_num):
            current_n = n1 if (n2 == 0 or iter_num < cycle_num - 1) else n2
            data = -np.ones((current_n, p_t))
            # 设置列名后续不影响内部运算
            # 处理离散节点
            for j in sorted(set(latent + obs_dis), key=lambda x: model.index.get_loc(x)):
                v = model.loc[j, "vertice"]
                if model.loc[j, "father_vertice"] == "":
                    p_1 = para_all[v][i]
                    a = np.random.binomial(1, p_1, current_n)
                    data[:, model.index.get_loc(j)] = a
                else:
                    p_j = para_all[v][:, i]
                    father_v = model.loc[j, "father_vertice"]
                    if father_v not in model.index:
                        print(f"警告：父节点 {father_v} 不在模型中！")
                        continue
                    father_idx = model.index.get_loc(father_v)
                    if data[0, father_idx] == -1:
                        print(f"警告：子节点 {v} 在父节点 {father_v} 之前产生！")
                    prob = np.where(data[:, father_idx] == 0, p_j[0], p_j[1])
                    result = np.random.binomial(1, prob)
                    data[:, model.index.get_loc(j)] = result
            # 处理连续节点
            for j in obs_con:
                v = model.loc[j, "vertice"]
                p_j = para_all[v][:, i]
                father_v = model.loc[j, "father_vertice"]
                if father_v not in model.index:
                    print(f"警告：父节点 {father_v} 不在模型中！")
                    continue
                father_idx = model.index.get_loc(father_v)
                if data[0, father_idx] == -1:
                    print(f"警告：子节点 {v} 在父节点 {father_v} 之前产生！")
                mu = np.where(data[:, father_idx] == 0, p_j[0], p_j[1])
                sigma = np.where(data[:, father_idx] == 0, p_j[2], p_j[3])
                result = np.random.normal(mu, sigma)
                data[:, model.index.get_loc(j)] = result
            
            # 设置列名为字符串 "1" 到 "p_t"
            df = pd.DataFrame(data, columns=[str(x) for x in range(1, p_t+1)])
            gc.collect()
            data_final_batch = data_choose(df.values, model)
            gc.collect()
            # 以追加方式写入 CSV 文件
            pd.DataFrame(data_final_batch).to_csv(filename, mode='a', index=False, header=False)
            final_data = data_final_batch  # 保存最后一批数据作为返回值
            del data
        print(f"完整的数据请查阅 {filename} 文件!")
        return final_data
    
def data_choose(data, model):
    """
    提取全部数据中的可观测变量数据，即去掉模型中标记为 'latent' 的列。
    
    参数:
      data: numpy 数组，形状 (n, p)，各列对应模型中的节点
      model: pandas DataFrame，必须包含 'vertice_type' 列，
             顺序与 data 的列顺序一致
             
    返回:
      新的 numpy 数组，删除了所有 'latent' 节点对应的列
    """
    # 获取所有标记为 "latent" 的列的索引 (0-based)
    latent_indices = [i for i, vt in enumerate(model['vertice_type']) if vt == "latent"]
    new_data = np.delete(data, latent_indices, axis=1)
    gc.collect()
    return new_data

def information_distance(data):
    """
    计算信息距离矩阵：-log(|correlation|)。
    
    参数:
      data: numpy 数组，假定每一列是一个变量（与 R 中 cor() 默认行为一致）
      
    返回:
      result: numpy 数组，信息距离矩阵
    """
    # 计算相关系数矩阵，rowvar=False 表示每一列为一个变量
    rho = np.corrcoef(data, rowvar=False)
    result = -np.log(np.abs(rho))
    return result


def information_distance2(filename):
    """
    根据文件计算信息距离矩阵。
    
    输入:
      filename: 字符串，数据文件路径。文件第一行为列名称（用逗号分隔），
                文件内容为数值数据，每行代表一个样本。
                
    输出:
      result: pandas DataFrame，行和列名称均为第一行读取的列名，
              每个元素为 -log(|相关系数|)。
    """
    # 打开文件
    with open(filename, "r") as f:
        # 读取第一行，清除引号和空格
        header_line = f.readline().strip()
        header_line = re.sub(r"[\'\"\s]", "", header_line)
        cname = header_line.split(",")
        obs_num = len(cname)
        
        # 初始化EX和EX2
        EX = np.zeros(obs_num)
        EX2 = np.zeros((obs_num, obs_num))
        sample_size = 0
        
        batch_size = 100000  # 每批读取行数
        
        while True:
            # 读取 batch_size 行
            batch = []
            for _ in range(batch_size):
                line = f.readline()
                if not line:
                    break
                batch.append(line.strip())
            if len(batch) == 0:
                break
            sample_size += len(batch)
            
            # 将每行按照逗号分割，转换为浮点数，构造二维数组
            data_i = np.array([list(map(float, l.split(","))) for l in batch])
            
            # 更新 EX（各列之和）和 EX2（data_i 的转置乘以 data_i）
            EX += np.sum(data_i, axis=0)
            EX2 += data_i.T @ data_i
            
            # 清理临时变量
            del data_i, batch
            gc.collect()
    
    # 计算样本均值和二阶矩阵
    EX = EX / sample_size
    EX2 = EX2 / sample_size
    
    # 计算协方差矩阵 Sigma2 = EX2 - EX*EX^T
    Sigma2 = EX2 - np.outer(EX, EX)
    
    # 构造对角矩阵：1/sqrt(diag(Sigma2))
    diag_vals = np.diag(Sigma2)
    # 防止出现零除错误（若存在零方差，可做调整，这里假设不为零）
    Sigma_diag = np.diag(1.0 / np.sqrt(diag_vals))
    
    # 计算相关系数矩阵: rho = Sigma_diag * Sigma2 * Sigma_diag
    rho = Sigma_diag @ Sigma2 @ Sigma_diag
    np.fill_diagonal(rho, 1)
    
    # 信息距离矩阵: -log(|rho|)
    result = -np.log(np.abs(rho))
    
    # 将结果转换为 DataFrame，并设置行列名称
    result_df = pd.DataFrame(result, index=cname, columns=cname)
    return result_df

def preprocess(i, model, para_all, n):
    """
    根据具体的 model 和参数表 para_all 的第 i 组参数产生数据（中间结果，不保存），
    并计算信息距离矩阵。
    
    输入:
      i: 整数，参数索引（0-based）
      model: pandas DataFrame，模型数据，必须包含列 'vertice', 'vertice_type', 'father_vertice'
             其中 'father_vertice' 为空字符串表示没有父节点。
      para_all: dict，键为 model['vertice'] 中的每个节点名称，对应的参数数组；
      n: 样本量 (整数)
    
    输出:
      result_df: pandas DataFrame，信息距离矩阵，其行列名称为字符串 "1","2",... 对应可观测变量个数。
    """
    # 1. 提取各类型节点的索引（基于行号，假设 model 使用默认整数索引）
    logi1 = model["vertice_type"] == "latent"
    logi2 = model["vertice_type"] == "observable_discrete"
    logi3 = model["vertice_type"] == "observable_continuous"
    
    p_t = model.shape[0]
    p_o = logi2.sum() + logi3.sum()  # 可观测变量个数
    latent = model.index[logi1].tolist()
    obs_con = model.index[logi3].tolist()
    obs_dis = model.index[logi2].tolist()
    
    # 用于设置结果行列名称（字符串 "1", "2", ..., "p_o"）
    cname = [str(x) for x in range(1, int(p_o)+1)]
    
    # 情况1: 如果 n * p_o 不大，则一次生成数据
    if n * p_o <= 100000000:
        data = -np.ones((n, p_t))
        # 设置列名为 model['vertice']
        data_df = pd.DataFrame(data, columns=model["vertice"])
        
        # 对于离散型节点（latent 和 observable_discrete）
        for j in sorted(latent + obs_dis):
            v = model.loc[j, "vertice"]
            # 如果父节点为空
            if model.loc[j, "father_vertice"] == "":
                p_1 = para_all[v][i]  # scalar
                a = np.random.binomial(1, p_1, n)
                data_df.iloc[:, j] = a
            else:
                p_j = para_all[v][:, i]  # 形状 (2,)
                father_v = model.loc[j, "father_vertice"]
                # 检查父节点数据是否已生成（注意：这里假设父节点的列名与 father_v 相同）
                if data_df.iloc[0][father_v] == -1:
                    print(f"警告：子节点 {v} 在父节点 {father_v} 之前产生！")
                # 根据父节点的取值判断：若为 0，则使用 p_j[0]，否则 p_j[1]
                prob = np.where(data_df[father_v] == 0, p_j[0], p_j[1])
                result = np.random.binomial(1, prob)
                data_df.iloc[:, j] = result
        
        # 对于连续型节点（observable_continuous）
        for j in obs_con:
            v = model.loc[j, "vertice"]
            p_j = para_all[v][:, i]  # 形状 (4,)
            father_v = model.loc[j, "father_vertice"]
            if data_df.iloc[0][father_v] == -1:
                print(f"警告：子节点 {v} 在父节点 {father_v} 之前产生！")
            mu = np.where(data_df[father_v] == 0, p_j[0], p_j[1])
            sigma = np.where(data_df[father_v] == 0, p_j[2], p_j[3])
            result = np.random.normal(mu, sigma)
            data_df.iloc[:, j] = result
        
        # 重设列名为 "1", "2", ... , "p_t"（不过后面 data_choose 会剔除隐节点）
        data_df.columns = [str(x) for x in range(1, p_t+1)]
        
        gc.collect()
        # 调用 data_choose 函数，返回去除隐节点后的数据
        data_filtered = data_choose(data_df.values, model)
        rho = np.corrcoef(data_filtered, rowvar=False)
        result_matrix = -np.log(np.abs(rho))
        # 将结果转换为 DataFrame，行列名称为 cname
        result_df = pd.DataFrame(result_matrix, index=cname, columns=cname)
        del data_df
        gc.collect()
        return result_df
    else:
        # 情况2: 数据较大，分批生成数据
        n1 = 100000
        n2 = n % n1
        cycle_num = (n // n1) + (0 if n2 == 0 else 1)
        EX = np.zeros(p_o)
        EX2 = np.zeros((p_o, p_o))
        sample_size = 0
        
        for iter_num in range(1, cycle_num + 1):
            # n_current: 每批样本数
            n_current = n1 if (n2 == 0 or iter_num != cycle_num) else n2
            data = -np.ones((n_current, p_t))
            data_df = pd.DataFrame(data, columns=model["vertice"])
            
            # 离散节点部分
            for j in sorted(latent + obs_dis):
                v = model.loc[j, "vertice"]
                if model.loc[j, "father_vertice"] == "":
                    p_1 = para_all[v][i]
                    a = np.random.binomial(1, p_1, n_current)
                    data_df.iloc[:, j] = a
                else:
                    p_j = para_all[v][:, i]
                    father_v = model.loc[j, "father_vertice"]
                    if data_df.iloc[0][father_v] == -1:
                        print(f"警告：子节点 {v} 在父节点 {father_v} 之前产生！")
                    prob = np.where(data_df[father_v] == 0, p_j[0], p_j[1])
                    result = np.random.binomial(1, prob)
                    data_df.iloc[:, j] = result
            
            # 连续节点部分
            for j in obs_con:
                v = model.loc[j, "vertice"]
                p_j = para_all[v][:, i]
                father_v = model.loc[j, "father_vertice"]
                if data_df.iloc[0][father_v] == -1:
                    print(f"警告：子节点 {v} 在父节点 {father_v} 之前产生！")
                mu = np.where(data_df[father_v] == 0, p_j[0], p_j[1])
                sigma = np.where(data_df[father_v] == 0, p_j[2], p_j[3])
                result = np.random.normal(mu, sigma)
                data_df.iloc[:, j] = result
            
            data_df.columns = [str(x) for x in range(1, p_t+1)]
            gc.collect()
            data_filtered = data_choose(data_df.values, model)
            gc.collect()
            sample_size += n_current
            EX += np.sum(data_filtered, axis=0)
            EX2 += data_filtered.T @ data_filtered
            del data_df
            gc.collect()
        
        EX = EX / sample_size
        EX2 = EX2 / sample_size
        Sigma2 = EX2 - np.outer(EX, EX)
        Sigma_diag = np.diag(1.0 / np.sqrt(np.diag(Sigma2)))
        rho = Sigma_diag @ Sigma2 @ Sigma_diag
        np.fill_diagonal(rho, 1)
        result_matrix = -np.log(np.abs(rho))
        cname = [str(x) for x in range(1, int(p_o)+1)]
        result_df = pd.DataFrame(result_matrix, index=cname, columns=cname)
        return result_df
    
def tree_from_edge(edge: pd.DataFrame) -> dict:
    """
    从边集构造树结构。
    输入:
      edge: pandas DataFrame，要求包含两列：
            第一列 'child' 为节点名，
            第二列 'parent' 为父节点名（根节点的父节点为 ""）。
    输出:
      tree: 字典，包含以下键：
            "obs_nodes", "lat_nodes", "str", "pa", "child",
            "des_nodes", "anc_nodes", "empty", "A", "Done", "D"
    """
    # 初始化 tree 字典，各字段初始为空
    tree = {
        "obs_nodes": None,
        "lat_nodes": None,
        "str": None,
        "pa": None,
        "child": None,
        "des_nodes": None,
        "anc_nodes": None,
        "empty": None,
        "A": None,
        "Done": None,
        "D": None
    }
    
    # r: 根节点，即父节点为空字符串
    r = edge.loc[edge['father_vertice'] == "", 'vertice'].tolist()
    
    # 1. 根据子节点名称首字符判断是否为隐节点
    # 假设隐节点以 "H" 开头
    is_latent = edge['vertice'].str.startswith("H")
    p_t = len(edge)
    p_o = (~is_latent).sum()  # 可观测节点数
    
    # 可观测节点：所有不以 "H" 开头的子节点
    obs_nodes = edge.loc[~is_latent, 'vertice'].tolist()
    tree["obs_nodes"] = obs_nodes
    
    # 2. 隐节点：所有以 "H" 开头的子节点
    lat_nodes = edge.loc[is_latent, 'vertice'].tolist()
    tree["lat_nodes"] = lat_nodes
    
    # 3. 构造结构信息 DataFrame
    # ID 为所有子节点（保持原顺序）
    ID = edge['vertice'].tolist()
    # 按 R 代码：前 p_o 行标记为 "observe"，后面为 "latent"
    obs_lat = ["observe"] * p_o + ["latent"] * (p_t - p_o)
    degree = np.ones(p_t, dtype=int)
    # 使用 value_counts 统计非空父节点出现次数
    non_empty_parents = edge.loc[edge['father_vertice'] != "", 'father_vertice']
    temp = non_empty_parents.value_counts()
    for node, count in temp.items():
        if node in ID:
            idx = ID.index(node)
            degree[idx] = int(count) + 1
    # 对根节点 r，减 1
    for r_node in r:
        if r_node in ID:
            idx = ID.index(r_node)
            degree[idx] -= 1
    # 计算 ch_obs_num: 对于观测节点（child not latent），父节点出现的次数
    ch_obs_num = np.zeros(p_t, dtype=int)
    obs_parent = edge.loc[~is_latent, 'father_vertice']
    ch_obs_num_tab = obs_parent.value_counts()
    for node, count in ch_obs_num_tab.items():
        if node in ID:
            idx = ID.index(node)
            ch_obs_num[idx] = int(count)
    st = pd.DataFrame({
        "ID": ID,
        "obs_lat": obs_lat,
        "degree": degree,
        "ch_obs_num": ch_obs_num
    }, index=ID)
    tree["str"] = st

    # print("st: ",st)

    # 4. 父节点字典：从 edge 构造映射 child -> parent
    pa = dict(zip(edge['vertice'], edge['father_vertice']))
    tree["pa"] = pa

    # print("pa",pa)

    # 5. 子节点字典：对每个 latent 节点，收集所有其作为父节点的子节点
    child = {node: [] for node in ID}
    for h in lat_nodes:
        # 选出 edge 中父节点为 h 的行
        children = edge.loc[edge['father_vertice'] == h, 'vertice'].tolist()
        child[h] = children
    tree["child"] = child

    # print("child: ",child)

    # 6. 后代节点：递归函数实现
    def get_des(h, child, des_nodes):
        child_h = child.get(h, [])
        if not child_h:
            return []
        # 判断哪些 child_h 是 latent（以 "H" 开头）
        latent_flags = [str(ch).startswith("H") for ch in child_h]
        if sum(latent_flags) == 0:
            return child_h
        else:
            temp_list = child_h.copy()
            H1 = [ch for ch, flag in zip(child_h, latent_flags) if flag]
            for h1 in H1:
                if des_nodes.get(h1):
                    temp_list = list(set(temp_list) | {h1} | set(des_nodes[h1]))
                else:
                    temp_list = list(set(temp_list) | {h1} | set(get_des(h1, child, des_nodes)))
            return temp_list

    des_nodes = {node: [] for node in ID}
    # 对 latent 节点倒序处理（R 中 for(h in lat_nodes[length(lat_nodes):1]）
    for h in reversed(lat_nodes):
        temp_list = get_des(h, child, des_nodes)
        # 分离 latent 与 observable后代
        latent_list = [x for x in temp_list if str(x).startswith("H")]
        obs_list = [x for x in temp_list if not str(x).startswith("H")]
        # 对 latent_list: 排序依据去掉 "H" 后转换为数字
        try:
            latent_list_sorted = sorted(latent_list, key=lambda x: int(str(x)[1:]))
        except:
            latent_list_sorted = sorted(latent_list)
        # 对 observable list: 排序转换为数字
        try:
            obs_list_sorted = sorted(obs_list, key=lambda x: int(x))
        except:
            obs_list_sorted = sorted(obs_list)
        des_nodes[h] = latent_list_sorted + obs_list_sorted
    tree["des_nodes"] = des_nodes

    # print("des_nodes: ",des_nodes)

    # 7. 祖先节点：利用 networkx 计算最短路径
    anc_nodes = {node: [] for node in ID}
    # 构造图（只考虑非空父节点）
    edge_nonempty = edge[edge['father_vertice'] != ""]
    g = nx.from_pandas_edgelist(edge_nonempty, source="vertice", target="father_vertice", create_using=nx.Graph())
    for v in edge_nonempty['vertice']:
        # 使用第一个根节点作为目标（若有多个根）
        if r:
            try:
                sp = nx.shortest_path(g, source=v, target=r[0])
                # 去掉第一个元素（即 v 自身），得到祖先节点列表
                temp1 = sp[1:] if len(sp) > 1 else []
                anc_nodes[v] = temp1
            except nx.NetworkXNoPath:
                anc_nodes[v] = []
        else:
            anc_nodes[v] = []
    tree["anc_nodes"] = anc_nodes

    # print("anc_nodes: ",anc_nodes)

    # 8. empty 标志
    tree["empty"] = False

    # 9. A: 设置为根节点 r
    tree["A"] = r

    # 10. Done: 所有节点中不在 r 的节点
    tree["Done"] = [node for node in ID if node not in r]

    # 11. D: 计算 1 - I, 其中 I 为单位矩阵，维度为 p_o x p_o
    D_mat = np.ones((p_o, p_o)) - np.eye(p_o)
    tree["D"] = D_mat

    return tree



# def tree_from_edge(edge: pd.DataFrame) -> dict:
#     """
#     从边集构造树结构。
#     输入:
#       edge: pandas DataFrame，要求包含两列：
#             第一列 'child' 为节点名，
#             第二列 'parent' 为父节点名（根节点的父节点为 ""）。
#     输出:
#       tree: 字典，包含以下键：
#             "obs_nodes", "lat_nodes", "str", "pa", "child",
#             "des_nodes", "anc_nodes", "empty", "A", "Done", "D"
#     """
#     # 初始化 tree 字典，各字段初始为空
#     tree = {
#         "obs_nodes": None,
#         "lat_nodes": None,
#         "str": None,
#         "pa": None,
#         "child": None,
#         "des_nodes": None,
#         "anc_nodes": None,
#         "empty": None,
#         "A": None,
#         "Done": None,
#         "D": None
#     }
    
#     # r: 根节点，即父节点为空字符串
#     r = edge.loc[edge['parent'] == "", 'child'].tolist()
    
#     # 1. 根据子节点名称首字符判断是否为隐节点
#     # 假设隐节点以 "H" 开头
#     is_latent = edge['child'].str.startswith("H")
#     p_t = len(edge)
#     p_o = (~is_latent).sum()  # 可观测节点数
    
#     # 可观测节点：所有不以 "H" 开头的子节点
#     obs_nodes = edge.loc[~is_latent, 'child'].tolist()
#     tree["obs_nodes"] = obs_nodes
    
#     # 2. 隐节点：所有以 "H" 开头的子节点
#     lat_nodes = edge.loc[is_latent, 'child'].tolist()
#     tree["lat_nodes"] = lat_nodes
    
#     # 3. 构造结构信息 DataFrame
#     # ID 为所有子节点（保持原顺序）
#     ID = edge['child'].tolist()
#     # 按 R 代码：前 p_o 行标记为 "observe"，后面为 "latent"
#     obs_lat = ["observe"] * p_o + ["latent"] * (p_t - p_o)
#     degree = np.ones(p_t, dtype=int)
#     # 使用 value_counts 统计非空父节点出现次数
#     non_empty_parents = edge.loc[edge['parent'] != "", 'parent']
#     temp = non_empty_parents.value_counts()
#     for node, count in temp.items():
#         if node in ID:
#             idx = ID.index(node)
#             degree[idx] = int(count) + 1
#     # 对根节点 r，减 1
#     for r_node in r:
#         if r_node in ID:
#             idx = ID.index(r_node)
#             degree[idx] -= 1
#     # 计算 ch_obs_num: 对于观测节点（child not latent），父节点出现的次数
#     ch_obs_num = np.zeros(p_t, dtype=int)
#     obs_parent = edge.loc[~is_latent, 'parent']
#     ch_obs_num_tab = obs_parent.value_counts()
#     for node, count in ch_obs_num_tab.items():
#         if node in ID:
#             idx = ID.index(node)
#             ch_obs_num[idx] = int(count)
#     st = pd.DataFrame({
#         "ID": ID,
#         "obs_lat": obs_lat,
#         "degree": degree,
#         "ch_obs_num": ch_obs_num
#     }, index=ID)
#     tree["str"] = st

#     # 4. 父节点字典：从 edge 构造映射 child -> parent
#     pa = dict(zip(edge['child'], edge['parent']))
#     tree["pa"] = pa

#     # 5. 子节点字典：对每个 latent 节点，收集所有其作为父节点的子节点
#     child = {node: [] for node in ID}
#     for h in lat_nodes:
#         # 选出 edge 中父节点为 h 的行
#         children = edge.loc[edge['parent'] == h, 'child'].tolist()
#         child[h] = children
#     tree["child"] = child

#     # 6. 后代节点：递归函数实现
#     def get_des(h, child, des_nodes):
#         child_h = child.get(h, [])
#         if not child_h:
#             return []
#         # 判断哪些 child_h 是 latent（以 "H" 开头）
#         latent_flags = [str(ch).startswith("H") for ch in child_h]
#         if sum(latent_flags) == 0:
#             return child_h
#         else:
#             temp_list = child_h.copy()
#             H1 = [ch for ch, flag in zip(child_h, latent_flags) if flag]
#             for h1 in H1:
#                 if des_nodes.get(h1):
#                     temp_list = list(set(temp_list) | {h1} | set(des_nodes[h1]))
#                 else:
#                     temp_list = list(set(temp_list) | {h1} | set(get_des(h1, child, des_nodes)))
#             return temp_list

#     des_nodes = {node: [] for node in ID}
#     # 对 latent 节点倒序处理（R 中 for(h in lat_nodes[length(lat_nodes):1]）
#     for h in reversed(lat_nodes):
#         temp_list = get_des(h, child, des_nodes)
#         # 分离 latent 与 observable后代
#         latent_list = [x for x in temp_list if str(x).startswith("H")]
#         obs_list = [x for x in temp_list if not str(x).startswith("H")]
#         # 对 latent_list: 排序依据去掉 "H" 后转换为数字
#         try:
#             latent_list_sorted = sorted(latent_list, key=lambda x: int(str(x)[1:]))
#         except:
#             latent_list_sorted = sorted(latent_list)
#         # 对 observable list: 排序转换为数字
#         try:
#             obs_list_sorted = sorted(obs_list, key=lambda x: int(x))
#         except:
#             obs_list_sorted = sorted(obs_list)
#         des_nodes[h] = latent_list_sorted + obs_list_sorted
#     tree["des_nodes"] = des_nodes

#     # 7. 祖先节点：利用 networkx 计算最短路径
#     anc_nodes = {node: [] for node in ID}
#     # 构造图（只考虑非空父节点）
#     edge_nonempty = edge[edge['parent'] != ""]
#     g = nx.from_pandas_edgelist(edge_nonempty, source="child", target="parent", create_using=nx.Graph())
#     for v in edge_nonempty['child']:
#         # 使用第一个根节点作为目标（若有多个根）
#         if r:
#             try:
#                 sp = nx.shortest_path(g, source=v, target=r[0])
#                 # 去掉第一个元素（即 v 自身），得到祖先节点列表
#                 temp1 = sp[1:] if len(sp) > 1 else []
#                 anc_nodes[v] = temp1
#             except nx.NetworkXNoPath:
#                 anc_nodes[v] = []
#         else:
#             anc_nodes[v] = []
#     tree["anc_nodes"] = anc_nodes

#     # 8. empty 标志
#     tree["empty"] = False

#     # 9. A: 设置为根节点 r
#     tree["A"] = r

#     # 10. Done: 所有节点中不在 r 的节点
#     tree["Done"] = [node for node in ID if node not in r]

#     # 11. D: 计算 1 - I, 其中 I 为单位矩阵，维度为 p_o x p_o
#     D_mat = np.ones((p_o, p_o)) - np.eye(p_o)
#     tree["D"] = D_mat

#     return tree

def Init(D):
    """
    根据输入的距离矩阵 D 初始化树结构。
    
    输入:
      D: numpy array, shape (num, num)
    输出:
      tree: dict, 包含以下键：
            "obs_nodes", "lat_nodes", "str", "pa", "child",
            "des_nodes", "anc_nodes", "empty", "A", "Done", "D"
    """

    # print("D shape:", D.shape)

    # 获取矩阵行数
    num = D.shape[0]
    # 可观测节点：字符串形式 "1", "2", ..., "num"
    obs_nodes = [str(i) for i in range(1, num+1)]
    
    
    # 初始化 tree 字典，先将所有键设置为 None，再进行赋值（或直接构造对应类型）
    tree = {}
    # 1. 可观测节点列表
    tree["obs_nodes"] = obs_nodes
    # 2. 隐节点列表：初始为空列表
    tree["lat_nodes"] = []
    # 3. 构造结构信息 DataFrame st
    st = pd.DataFrame({
        "ID": obs_nodes,
        "obs_lat": ["observe"] * num,
        "degree": [0] * num,
        "ch_obs_num": [0] * num
    }, index=obs_nodes)
    tree["str"] = st
    # 4. 父节点字典 pa：所有节点初始父节点为空字符串
    tree["pa"] = {node: "" for node in obs_nodes}
    # 5. 子节点字典：每个节点对应空列表
    tree["child"] = {node: [] for node in obs_nodes}
    # 6. 后代节点字典，初始为每个节点对应空列表
    tree["des_nodes"] = {node: [] for node in obs_nodes}
    # 7. 祖先节点字典，同样每个节点对应空列表
    tree["anc_nodes"] = {node: [] for node in obs_nodes}
    # 8. empty 标志设为 True
    tree["empty"] = True
    # 9. A：初始设为所有可观测节点列表
    tree["A"] = obs_nodes
    # 10. Done：初始设为空列表
    tree["Done"] = []
    # 11. D：保存距离矩阵（转换为 numpy 数组）
    tree["D"] = D
    
    return tree


def update_tree(sab_groups, tree):
    """
    利用最大同胞组来更新树结构 tree。
    
    输入:
      sab_groups: list of lists, 每个子列表为一个同胞组，组内元素均为节点名称。
                  sab_groups中的元素是0-based,
                  在update_tree中使用时需要注意把0-based映射为1-based，因为树的结构字典采用1-based更加自然一些。
      tree: dict, 树结构字典，包含以下键：
            "obs_nodes", "lat_nodes", "str", "pa", "child", "des_nodes",
            "anc_nodes", "empty", "A", "Done", "D"
    
    返回:
      更新后的 tree（dict）
    """
    
    # 若 sab_groups 为空或展开为空，则直接返回 tree
    # isinstance(item, int): 仅对整数项执行 +1，避免对字符串（如 "H1"）等非整数值执行不必要的转换。
    # str(item + 1): 将 0-based 索引转换为 1-based，并转换为字符串，确保 temp 统一为字符串格式。
    # else str(item): 保持非整数的原始值（如 "H1"）不变，防止错误。

    if sab_groups is None or len(sab_groups) == 0:
        return tree
    temp = [str(item + 1) if isinstance(item, int) else str(item) for group in sab_groups for item in group]
    if not temp:
        return tree

    # 当前已有隐节点列表及数量
    lat_nodes = tree.get("lat_nodes", [])
    num = len(lat_nodes)
    num_new = len(sab_groups)
    # 生成新隐节点标签：从 "H{num+1}" 到 "H{num+num_new}"
    lat_nodes_new = [f"H{num + i + 1}" for i in range(num_new)]

    print("lat_nodes_new: ",lat_nodes_new)
    
    # -----------------------2----------------------
    # 更新隐节点列表：将新隐节点追加到原有隐节点列表
    tree["lat_nodes"] = lat_nodes + lat_nodes_new


    
    # -----------------------3----------------------
    # 构造新隐节点的结构信息 DataFrame st2
    # 每个新隐节点的 ID 为对应标签，obs_lat 固定为 "latent"
    # degree 为同胞组中节点数量，ch_obs_num 为同胞组中属于可观测节点的数量
    obs_nodes = tree.get("obs_nodes", [])
    st2_data = {
        "ID": lat_nodes_new,
        "obs_lat": ["latent"] * num_new,
        "degree": [],
        "ch_obs_num": []
    }
    for group in sab_groups:
        st2_data["degree"].append(len(group))
        ch_obs_num = sum(1 for x in group if x in obs_nodes)
        st2_data["ch_obs_num"].append(ch_obs_num)
    st2 = pd.DataFrame(st2_data, index=lat_nodes_new)
    
    # 原结构信息 st1
    st1 = tree.get("str")
    # temp1: 交集：在 st1 的第一列(ID)中与 temp 有交集
    temp1 = list(set(st1.index) & set(temp))
    # 对于交集中的节点，degree 加 1

    # print("temp1: ",temp1,"temp: ",temp)

    for node in temp1:
        st1.at[node, "degree"] += 1
    # 合并 st1 和 st2（行方向合并）
    st = pd.concat([st1, st2])
    tree["str"] = st

    # -----------------------4----------------------
    # 更新父节点字典
    pa1 = tree.get("pa").copy()  # 复制原父节点字典
    pa2 = {str(node): "" for node in lat_nodes_new}  # 初始化新隐节点的父节点为空

    for i in range(num_new):
        group = sab_groups[i]  # 当前同胞组
        new_parent = str(lat_nodes_new[i])  # 该组的新父节点，确保为字符串

        for node in group:
            # 观测节点是 0-based，需要转换为 1-based
            node_str = str(node + 1) if isinstance(node, int) and node < len(tree["obs_nodes"]) else str(node)

            # 更新子节点的父节点
            pa1[node_str] = new_parent  

    # 由于 pa2 只是新隐节点的初始化，我们只在 `pa` 里保留需要的部分
    pa = {**pa1, **pa2}
    tree["pa"] = pa



    # print("pa: ",pa)
    # -----------------------5----------------------
    # 更新子节点字典
    child = tree.get("child", {}).copy()

    for i, new_node in enumerate(lat_nodes_new):
        # 先将 group 中的索引转换为字符串
        children_list = [str(node + 1) if isinstance(node, int) and node < len(tree["obs_nodes"]) else str(node) for node in sab_groups[i]]

        # 按照整数大小排序
        child[new_node] = sorted(children_list, key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x))


    tree["child"] = child

    # print("child: ", child)

    # -----------------------6----------------------
    # 更新后代节点字典
    des_nodes1 = tree.get("des_nodes", {}).copy()
    des_nodes2 = {}

    for i, new_node in enumerate(lat_nodes_new):
        group = sab_groups[i]

        # 先转换为字符串，并且 0-based 变 1-based
        temp3 = set()
        for node in group:
            node_str = str(node + 1) if isinstance(node, int) and node < len(tree["obs_nodes"]) else str(node)
            if node_str in des_nodes1:
                temp3.update(des_nodes1[node_str])  # 获取该节点的后代
            temp3.add(node_str)

        # 排序后再存入 des_nodes2
        des_nodes2[str(new_node)] = sorted(list(temp3), key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x))


    # 合并 des_nodes
    des_nodes1.update(des_nodes2)
    tree["des_nodes"] = des_nodes1

    # print("des_nodes: ", des_nodes1)


    # -----------------------7----------------------
    # 更新祖先节点字典
    anc_nodes1 = tree.get("anc_nodes").copy()
    nodes = list(anc_nodes1.keys())
    anc_nodes2 = {node: [] for node in lat_nodes_new}
    # 对于每个节点在 anc_nodes1，若其祖先列表为空，则尝试添加其父节点；否则看最后一个祖先的父节点，并追加
    for i, node in enumerate(nodes):
        temp4 = anc_nodes1[node]
        if len(temp4) == 0:
            pa_i = pa.get(node, "")
            if pa_i != "":
                anc_nodes1[node] = temp4 + [pa_i]
        else:
            temp5 = temp4[-1]
            pa_i = pa.get(temp5, "")
            if pa_i != "":
                anc_nodes1[node] = temp4 + [pa_i]
    anc_nodes = anc_nodes1.copy()
    anc_nodes.update(anc_nodes2)
    tree["anc_nodes"] = anc_nodes
    
    
    # ------------------------8-----------------------
    tree["empty"] = False

    # ------------------------9-----------------------
    # A: 所有父节点为空的节点，键在 pa 中对应值为空
    A = [node for node, parent in pa.items() if parent == ""]
    tree["A"] = A
    # print("A: ",A)

    # ------------------------10----------------------
    # Done: 所有父节点非空的节点
    Done = [node for node, parent in pa.items() if parent != ""]
    tree["Done"] = Done
    # print("Done: ",Done)
    
    # print("tree: ",tree)
    return tree

def update_tree_remain(rm_ch, tree):
    """
    利用紧相邻关系更新树结构。
    
    输入:
      rm_ch: 一个二维 numpy 数组或列表，其中每一行包含两个元素，
             分别表示 [child, parent]（均为节点名称，字符串）。
      tree: 一个字典，包含以下键：
            "str": pandas DataFrame，结构信息，至少包含 "degree" 列，
                   行索引为节点名称。
            "pa": dict，节点到父节点的映射（字符串 -> 字符串）。
            "child": dict，节点到子节点列表的映射。
            "des_nodes": dict，节点到后代节点列表的映射。
            "anc_nodes": dict，节点到祖先节点列表的映射。
            "empty": bool，标志是否为空。
            "A": list，父节点为空的节点列表（根节点）。
            "Done": list，父节点非空的节点列表。
    
    返回:
      更新后的 tree 字典。
    """

    # print("update_tree_remain(rm_ch: ",rm_ch)
    # 将 rm_ch 转换为 numpy 数组（如果尚未）
    rm_ch = np.array(rm_ch)
    num = rm_ch.shape[0]
    
    # -----------------------3----------------------
    st = tree["str"].copy()
    # 假设 st 的第三列为 "degree"
    # 对于每个 child (第一列) 其 degree 加1
    for child in rm_ch[:, 0]:
        if child in st.index:
            st.at[child, "degree"] += 1
    # 对于每个 parent (第二列) 其 degree 加1
    for parent in rm_ch[:, 1]:
        if parent in st.index:
            st.at[parent, "degree"] += 1
    tree["str"] = st

    # print(tree["str"])

    # -----------------------4----------------------
    pa = tree["pa"].copy()
    # 检查是否有子节点已经有父节点
    for child in rm_ch[:, 0]:
        if child in pa and pa[child] != "":
            # 发出警告
            print(f"警告：子节点 {child} 已经有父节点 {pa[child]}!")
    # 将紧相邻中的子节点的父节点设为对应的父节点（即 rm_ch 每一行 [child, parent]）
    for child, parent in rm_ch:
        pa[child] = parent
    tree["pa"] = pa

    # print(tree["pa"])

    # -----------------------5----------------------
    child_dict = tree["child"].copy()
    # 对于每一行，将子节点添加到对应父节点的子节点列表中
    for child, parent in rm_ch:
        # 如果父节点不存在，则初始化为列表
        if parent in child_dict:
            child_dict[parent].append(child)
        else:
            child_dict[parent] = [child]
    tree["child"] = child_dict

    # print(tree["child"])

    # -----------------------6----------------------
    des_nodes = tree["des_nodes"].copy()
    for i in range(num):
        child_node = rm_ch[i, 0]
        parent = rm_ch[i, 1]
        # 取出子节点的后代（如果不存在，则返回空列表）
        child_des = des_nodes.get(child_node, [])
        # 更新父节点的后代：将 子节点的后代、子节点本身，加上原父节点的后代
        parent_old = des_nodes.get(parent, [])
        des_nodes[parent] = child_des + [child_node] + parent_old
    tree["des_nodes"] = des_nodes

    # print(tree["des_nodes"])

    # -----------------------7----------------------
    anc_nodes = tree["anc_nodes"].copy()
    nodes = list(anc_nodes.keys())
    for node in nodes:
        temp = anc_nodes.get(node, [])
        if len(temp) != 0:
            temp2 = temp[-1]
            pa_i = pa.get(temp2, "")
            # 如果父节点非空，则将其添加到祖先列表中
            if pa_i != "":
                anc_nodes[node] = temp + [pa_i]
        else:
            pa_i = pa.get(node, "")
            if pa_i != "":
                anc_nodes[node] = temp + [pa_i]
    # 对于新添加的隐节点（若有），假设已经在 tree["lat_nodes"] 中，新节点的祖先暂为空
    # 这里不再额外添加 anc_nodes2，而是保持已有 anc_nodes 更新后的字典
    tree["anc_nodes"] = anc_nodes

    # print(tree["anc_nodes"])

    # -----------------------8----------------------
    tree["empty"] = False

    # -----------------------9----------------------
    # A: 所有父节点为空的节点
    A = [node for node, parent in pa.items() if parent == ""]
    tree["A"] = A

    # -----------------------10----------------------
    # Done: 所有父节点非空的节点
    Done = [node for node, parent in pa.items() if parent != ""]
    tree["Done"] = Done

    return tree

# # 示例调用
# if __name__ == "__main__":
#     # 构造示例 edge DataFrame
#     import pandas as pd
#     data = {
#         "child": ["1", "2", "H1", "3", "4", "H2", "H3"],
#         "parent": ["", "", "", "H1", "H1", "H3", ""]
#     }
#     edge_df = pd.DataFrame(data)
#     # 假设我们构造一个初始 tree，简单版，仅用于演示
#     # 这里构造 tree 的各项初始值
#     obs_nodes = [node for node in edge_df["child"] if not str(node).startswith("H")]
#     tree_example = {
#         "obs_nodes": obs_nodes,
#         "lat_nodes": [node for node in edge_df["child"] if str(node).startswith("H")],
#         "str": pd.DataFrame({
#             "ID": edge_df["child"],
#             "obs_lat": ["observe" if not str(x).startswith("H") else "latent" for x in edge_df["child"]],
#             "degree": [1]*len(edge_df),
#             "ch_obs_num": [0]*len(edge_df)
#         }, index=edge_df["child"]),
#         "pa": dict(zip(edge_df["child"], edge_df["parent"])),
#         "child": {node: [] for node in edge_df["child"]},
#         "des_nodes": {node: [] for node in edge_df["child"]},
#         "anc_nodes": {node: [] for node in edge_df["child"]},
#         "empty": True,
#         "A": obs_nodes,
#         "Done": [],
#         "D": np.eye(len(obs_nodes))  # 仅示例
#     }
#     # 示例：假设紧相邻对 rm_ch 为以下数组，每行 [child, parent]
#     rm_ch = np.array([
#         ["3", "H1"],
#         ["4", "H1"]
#     ])
#     updated_tree = update_tree_remain(rm_ch, tree_example)
#     print("Updated tree:")
#     for key, value in updated_tree.items():
#         print(f"{key}: {value}")

def JS_Cha(PHI, epson):
    """
    判断一列 PHI 是否接近，即其最大值与最小值之差是否小于等于 epson。
    如果 PHI 的长度不超过1，则返回 True。
    """
    if len(PHI) <= 1:
        return True
    M = np.max(PHI)
    m = np.min(PHI)
    return (M - m) <= epson

def JS_Cha1(PHI, phi, epson):
    """
    判断一列 PHI 的所有元素是否都与 phi 接近，即每个元素与 phi 的差的绝对值是否都小于等于 epson。
    如果 PHI 为空，则返回 True。
    """
    if len(PHI) == 0:
        return True
    return np.all(np.abs(PHI - phi) <= epson)

def JS_Cha2(PHI, phi, epson2=0.1):
    """
    判断一列 PHI 的所有元素是否都与 phi 差很多，即每个元素与 phi 的差的绝对值是否都大于 epson2。
    如果 PHI 为空，则返回 True。
    """
    if len(PHI) == 0:
        return True
    return np.all(np.abs(PHI - phi) > epson2)



def bifur(u, D, tree):
    """
    求 u 的二叉变量，根据距离矩阵 D 和树结构 tree 判断。
    
    参数:
      u: str, 节点名称
      D: numpy.ndarray, 信息距离矩阵。假设其行列顺序对应于可观测节点，
         且这些节点的名称为数字字符串（1-based，可以转换为整数）。
      tree: dict, 树结构，包含至少以下键：
            "child": dict, 节点到子节点列表的映射；
            "des_nodes": dict, 节点到后代节点列表的映射；
            "obs_nodes": list, 可观测节点名称的列表。
    
    返回:
      list of 2 int: 选择的二叉变量，按数值大小排序后返回，索引为 0-based。
    """
    # print("bifur(u: ",u)

    # 取 u 的子节点列表
    child = tree["child"]
    ch_u = child.get(u, [])
    
    # 将u的子节点分为可观测子节点（不以 "H" 开头）和隐子节点（以 "H" 开头）
    ch_obs = [str(node) for node in ch_u if not str(node).startswith("H")]
    ch_lat = [str(node) for node in ch_u if str(node).startswith("H")]

    # print("ch_obs: ",ch_obs)
    # print("ch_lat: ",ch_lat)
    
    num = len(ch_obs)
    des_nodes = tree["des_nodes"]

    # print("des_nodes[u]: ",des_nodes[u])

    # 情况1: 至少有两个可观测子节点
    if num >= 2:
        # print("u至少有两个可观测子节点")
        try:
            # 将可观测子节点转换为整数（1-based），例如 "1" -> 1, "2" -> 2
            obs_indices = [int(x) for x in ch_obs]
        except ValueError:
            raise ValueError("可观测节点名称必须为可转换为整数的字符串。")
        # 转换为 0-based 索引用于访问 D
        obs_indices_0 = [x - 1 for x in obs_indices]
        subD = D[np.ix_(obs_indices_0, obs_indices_0)]
        # 为了模拟 R 中的列优先（Fortran order），采用 Fortran 顺序扁平化
        flat = subD.flatten(order='F')

        order_idx = np.argsort(flat)
        # 在 R 中：ij <- order(D[ch_obs, ch_obs])[num+1]
        candidate = order_idx[num]  # 对应 R 中的 num+1 (1-based)，此处 candidate 是 0-based 索引于子矩阵
        temp_val = num + 1
        # 如果候选距离为 0，则继续取下一个非零距离
        while flat[order_idx[temp_val]] == 0:
            candidate = order_idx[temp_val]
            temp_val += 1
        # 将平面索引转换为子矩阵中的行和列索引
        i_sub = candidate % num
        j_sub = candidate // num
        # 将对应的节点转换为整数并转为 0-based：即 int(ch_obs) - 1
        a = [int(ch_obs[i_sub]) - 1, int(ch_obs[j_sub]) - 1]
        a_sorted = sorted(a)
        return a_sorted
    
    # 情况2: 只有 1 个可观测子节点
    elif num == 1:
        # print("u只有 1 个可观测子节点")
        a_set = list(set(tree["obs_nodes"]) - set(ch_obs))
        if u in des_nodes:
            b_set = list(set(des_nodes[u]).intersection(set(a_set)))
        else:
            b_set = []
        if not b_set:
            raise ValueError(f"节点 {u} 没有可用的候选节点！")
        try:
            obs_val = int(ch_obs[0])
        except ValueError:
            raise ValueError("可观测节点名称必须为可转换为整数的字符串。")
        # 1-based
        obs_idx = obs_val
        try:
            b_values = [int(x) for x in b_set]
        except ValueError:
            raise ValueError("候选节点名称必须为可转换为整数的字符串。")
        # 转换为 0-based
        obs_idx_0 = obs_idx - 1
        b_indices_0 = [x - 1 for x in b_values]
        d_vals = D[obs_idx_0, b_indices_0]
        order_idx = np.argsort(d_vals)
        chosen_index = order_idx[0]
        result_pair = [obs_val, b_values[chosen_index]]
        # 转换为 0-based：即各元素减 1
        result_pair = [result_pair[0] - 1, result_pair[1] - 1]
        result_pair_sorted = sorted(result_pair)
        return result_pair_sorted
    
    # 情况3: 没有可观测子节点
    else:
        # print("u没有可观测子节点")
        if len(ch_lat) < 2:
            raise ValueError("A latent node should have two child-nodes at least!")
        # 取 u 第一个隐子节点的后代与全体可观测节点的交集
        des_lat1 = sorted(set(des_nodes.get(ch_lat[0], [])) & set(tree["obs_nodes"]), key=lambda x: int(x))
        # 取 u 第二个隐子节点的后代与全体可观测节点的交集
        des_lat2 = sorted(set(des_nodes.get(ch_lat[1], [])) & set(tree["obs_nodes"]), key=lambda x: int(x))

        # print("des_lat1: ",des_lat1)

        # 计算交集并按数值大小排序
        common = sorted(set(des_lat1) & set(des_lat2), key=lambda x: int(x))
        temp_iter = len(common)

        # 处理索引转换（D 是 0-based）
        idx1 = [int(x) for x in des_lat1]
        idx2 = [int(x) for x in des_lat2]
        idx1_0 = [x - 1 for x in idx1]  # 0-based
        idx2_0 = [x - 1 for x in idx2]  # 0-based

        # 提取子矩阵
        subD = D[np.ix_(idx1_0, idx2_0)]
        # 采用 Fortran 顺序展开子矩阵（列优先，即 Fortran order）
        flat = subD.flatten(order='F')
        order_idx = np.argsort(flat)  # 按数值排序后获取索引

        # 选择第一个非零索引
        candidate = order_idx[temp_iter]  


        # print("flat: ")
        # print(flat)
        # print("order_idx: ")
        # print(order_idx)

        # print("candidate0: ",candidate,"order_idx[temp_iter]: ",order_idx[temp_val],"flat[candidate]",flat[candidate])
        while temp_iter < len(order_idx) and order_idx[temp_iter] == 0:
            candidate = order_idx[temp_iter+1]
            print("candidate1: ",candidate)
            temp_iter += 1

        # print("candidate: ",candidate,"common: ",common)

        # 计算最终索引（0-based）
        i_sub = candidate % len(des_lat1)
        j_sub = candidate // len(des_lat1)
        a = [idx1_0[i_sub], idx2_0[j_sub]]  # 确保 0-based

        
        a_sorted = sorted(a)

        # print("a_sorted: ",a_sorted)

        return a_sorted


# def bifur(u, D, tree):
#     """
#     求 u 的二叉变量，根据距离矩阵 D 和树结构 tree 判断。
    
#     参数:
#       u: str, 节点名称
#       D: numpy.ndarray, 信息距离矩阵。假设其行列顺序对应于可观测节点，
#          且这些节点的名称为数字字符串（1-based，可以转换为整数）。
#       tree: dict, 树结构，包含至少以下键：
#             "child": dict, 节点到子节点列表的映射；
#             "des_nodes": dict, 节点到后代节点列表的映射；
#             "obs_nodes": list, 可观测节点名称的列表。
    
#     返回:
#       list of 2 int: 选择的二叉变量，按数值大小排序后返回，索引为 0-based。
#     """
#     # print("bifur(u: ",u)

#     # 取 u 的子节点列表
#     child = tree["child"]
#     ch_u = child.get(u, [])
    
#     # 将u的子节点分为可观测子节点（不以 "H" 开头）和隐子节点（以 "H" 开头）
#     ch_obs = [str(node) for node in ch_u if not str(node).startswith("H")]
#     ch_lat = [str(node) for node in ch_u if str(node).startswith("H")]

#     # print("ch_obs: ",ch_obs)
#     # print("ch_lat: ",ch_lat)
    
#     num = len(ch_obs)
#     des_nodes = tree["des_nodes"]

#     # print("des_nodes[u]: ",des_nodes[u])

#     # 情况1: 至少有两个可观测子节点
#     if num >= 2:
#         # print("u至少有两个可观测子节点")
#         try:
#             # 将可观测子节点转换为整数（1-based），例如 "1" -> 1, "2" -> 2
#             obs_indices = [int(x) for x in ch_obs]
#         except ValueError:
#             raise ValueError("可观测节点名称必须为可转换为整数的字符串。")
#         # 转换为 0-based 索引用于访问 D
#         obs_indices_0 = [x - 1 for x in obs_indices]
#         subD = D[np.ix_(obs_indices_0, obs_indices_0)]
#         flat = subD.flatten()
#         order_idx = np.argsort(flat)
#         # 在 R 中：ij <- order(D[ch_obs, ch_obs])[num+1]
#         candidate = order_idx[num]  # 对应 R 中的 num+1 (1-based)，此处 candidate 是 0-based 索引于子矩阵
#         temp_val = num + 1
#         # 如果候选距离为 0，则继续取下一个非零距离
#         while flat[order_idx[temp_val]] == 0:
#             candidate = order_idx[temp_val]
#             temp_val += 1
#         # 将平面索引转换为子矩阵中的行和列索引
#         i_sub = candidate % num
#         j_sub = candidate // num
#         # 将对应的节点转换为整数并转为 0-based：即 int(ch_obs) - 1
#         a = [int(ch_obs[i_sub]) - 1, int(ch_obs[j_sub]) - 1]
#         a_sorted = sorted(a)
#         return a_sorted
    
#     # 情况2: 只有 1 个可观测子节点
#     elif num == 1:
#         # print("u只有 1 个可观测子节点")
#         a_set = list(set(tree["obs_nodes"]) - set(ch_obs))
#         if u in des_nodes:
#             b_set = list(set(des_nodes[u]).intersection(set(a_set)))
#         else:
#             b_set = []
#         if not b_set:
#             raise ValueError(f"节点 {u} 没有可用的候选节点！")
#         try:
#             obs_val = int(ch_obs[0])
#         except ValueError:
#             raise ValueError("可观测节点名称必须为可转换为整数的字符串。")
#         # 1-based
#         obs_idx = obs_val
#         try:
#             b_values = [int(x) for x in b_set]
#         except ValueError:
#             raise ValueError("候选节点名称必须为可转换为整数的字符串。")
#         # 转换为 0-based
#         obs_idx_0 = obs_idx - 1
#         b_indices_0 = [x - 1 for x in b_values]
#         d_vals = D[obs_idx_0, b_indices_0]
#         order_idx = np.argsort(d_vals)
#         chosen_index = order_idx[0]
#         result_pair = [obs_val, b_values[chosen_index]]
#         # 转换为 0-based：即各元素减 1
#         result_pair = [result_pair[0] - 1, result_pair[1] - 1]
#         result_pair_sorted = sorted(result_pair)
#         return result_pair_sorted
    
#     # 情况3: 没有可观测子节点
#     else:
#         # print("u没有可观测子节点")
#         if len(ch_lat) < 2:
#             raise ValueError("A latent node should have two child-nodes at least!")
#         # 取 u 第一个隐子节点的后代与全体可观测节点的交集
#         des_lat1 = sorted(set(des_nodes.get(ch_lat[0], [])) & set(tree["obs_nodes"]), key=lambda x: int(x))
#         # 取 u 第二个隐子节点的后代与全体可观测节点的交集
#         des_lat2 = sorted(set(des_nodes.get(ch_lat[1], [])) & set(tree["obs_nodes"]), key=lambda x: int(x))

#         # 计算交集并按数值大小排序
#         common = sorted(set(des_lat1) & set(des_lat2), key=lambda x: int(x))
#         temp_val = len(common)

#         # 处理索引转换（D 是 0-based）
#         idx1 = [int(x) for x in des_lat1]
#         idx2 = [int(x) for x in des_lat2]
#         idx1_0 = [x - 1 for x in idx1]  # 0-based
#         idx2_0 = [x - 1 for x in idx2]  # 0-based

#         # 提取子矩阵
#         subD = D[np.ix_(idx1_0, idx2_0)]
#         flat = subD.flatten()
#         order_idx = np.argsort(flat)  # 按数值排序后获取索引

#         # 选择第一个非零索引
#         candidate = order_idx[temp_val]  
#         temp_iter = temp_val + 1
#         while temp_iter < len(order_idx) and flat[order_idx[temp_iter]] == 0:
#             candidate = order_idx[temp_iter]
#             temp_iter += 1

#         # 计算最终索引（0-based）
#         i_sub = candidate % len(des_lat1)
#         j_sub = candidate // len(des_lat1)
#         a = [idx1_0[i_sub], idx2_0[j_sub]]  # 确保 0-based
#         a_sorted = sorted(a)

#         return a_sorted



def JS_V_V_Pair(nodes, D, epson, tau1, tau2):
    """
    判断两个节点是否应构成同胞对，并返回包含节点名称的 DataFrame 或 None。
    
    参数:
      nodes: list 或 tuple，包含两个节点名称，例如 ["1", "2"]。
      D: NumPy，信息距离矩阵，行列名称为节点名称。
      epson: float, 阈值参数，用于判断向量元素是否接近。
      tau1, tau2: float, 距离阈值。
      
    返回:
      如果符合条件，则返回一个 DataFrame，包含两列 'node1' 和 'node2'；
      否则返回 None。
    """
    

    # print("JS_V_V_Pair(nodes: ",nodes[0], nodes[1])
    # 将节点名称转换为整数索引（假设节点名称为数字字符串）
    try:
            i_str, j_str = nodes[0], nodes[1]
            i = int(i_str) - 1  # 转换为 0-based 索引
            j = int(j_str) - 1
    except ValueError:
            raise ValueError("节点名称必须可转换为整数！")
    
    # 如果 i,j 之间的距离大于 tau1，则排除该同胞对，返回 None
    if D[i, j] > tau1:
        return None

    # logi1: 对 D 的列名，标记哪些列名不在 nodes 中
    # 假设 D 的维度为 n x n
    n = D.shape[0]
    idx = np.arange(n)
    logi1 = ~np.isin(idx, [i, j])
    # logi2: 对每一行，判断 D[row, i] 和 D[row, j] 是否都小于等于 tau2
    logi2 = (D[:, i] <= tau2) & (D[:, j] <= tau2)
    # 由于 D 为方阵，行数与列数相同，故两者长度一致
    logi = logi1 & logi2
    num_true = np.sum(logi)
    
    if num_true == 1:
        result = pd.DataFrame({"node1": [i], "node2": [j]})
        return result
    elif num_true == 0:
        result = pd.DataFrame({"node1": [i], "node2": [j]})
        return result
    else:
        # 计算 PHI_ij = D[i, logi] - D[j, logi]
        PHI_ij = D[logi, i] - D[logi, j]
        # 调用 JS_Cha 判断 PHI_ij 是否接近
        flag = JS_Cha(PHI_ij, epson)
        if flag:
            result = pd.DataFrame({"node1": [i], "node2": [j]})
            return result
        else:
            return None
        

        
def JS_V_H_Pair(nodes, tree, epson, tau2, epson2=0.1):
    """
    判断节点对 (v, u) 是否满足条件，返回同胞对结果。
    
    参数:
      nodes: list 或 tuple, 包含两个节点名称 [v, u]（v 为第一个，u 为第二个）
      tree: dict, 树结构，至少包含以下键：
            "D": 信息距离矩阵 (numpy array)
            "obs_nodes": list，所有可观测节点名称（假设顺序与 D 的行顺序一致）
            "des_nodes": dict，映射每个节点到其后代节点（列表，节点名称为字符串）
      epson: float, 阈值参数，用于 JS_Cha 判定
      tau2: float, 距离阈值 tau2
      epson2: float, 次级阈值，默认 0.1
      
    返回:
      如果条件满足，返回一个包含一行、列为 'node1' 和 'node2' 的 DataFrame；
      否则返回 None。
    """
    # 取出节点 v 和 u（原始字符串形式）
    v = nodes[0]
    u = nodes[1]

    # print("JS_V_H_Pair(nodes: ",u,v)
    # 将 v 转换为整数索引 (0-based)，假设 v 为数字字符串
    try:
        v_idx = int(v) - 1
    except Exception as e:
        raise ValueError("节点 v 必须是可以转换为整数的字符串")
    
    # 获取 D（numpy 数组）和可观测节点列表
    D = tree["D"]
    # 构造返回结果 DataFrame（保留原始节点名称）
    result = pd.DataFrame({"node1": [v], "node2": [u]})
    
    obs_nodes = tree["obs_nodes"]
    # temp = union(v, tree["des_nodes"][u])
    temp = set([v]) | set(tree["des_nodes"].get(u, []))
    # M = obs_nodes \ temp
    M = [node for node in obs_nodes if node not in temp]
    
    # 调用 bifur(u, D, tree) 得到一对候选节点（假设返回的是整数索引，0-based）
    ij = bifur(u, D, tree)

    # print("ij:",ij)

    if ij is None:
        return None
    i_node, j_node = ij[0], ij[1]
    
    # 下面访问 D 时，均使用整数索引
    # 构造布尔掩码 logi：对于所有行（对应 obs_nodes），检查 D[row, i_node] <= tau2 且 D[row, v_idx] <= tau2
    n = D.shape[0]
    idx = np.arange(n)
    # logi1 = (D[:, i_node] <= tau2) & (D[:, v_idx] <= tau2)
    # # logi2：对于每个 obs_nodes（按其在 D 中的行顺序），判断是否属于 M
    # logi2 = np.array([node in M for node in obs_nodes])
    # logi = logi1 & logi2
    # num_true = np.sum(logi)

    logi1 = (D[:, i_node] <= tau2) & (D[:, v_idx] <= tau2)
    # `logi2` 计算方式优化，直接用 NumPy 向量化方法提高效率
    logi2 = np.isin(obs_nodes, M)  # 直接检查 obs_nodes 是否在 M 中，保持 NumPy 数组操作
    # 计算最终布尔掩码
    logi = logi1 & logi2
    # 计算 `logi` 中 `True` 的数量
    num_true = np.sum(logi)

    # print("logi: ",logi)

    if num_true == 1:
        # print("num_true == 1:")
        return result
    elif num_true == 0:
        # print("num_true == 0:")
        return None
    else:
        # 计算 PHI_vi = D[logi, v_idx] - D[logi, i_node]
        PHI_vi = D[logi, v_idx] - D[logi, i_node]
        # 计算 phi_vij = D[j_node, v_idx] - D[j_node, i_node]
        phi_vij = D[j_node, v_idx] - D[j_node, i_node]
        flag = JS_Cha(PHI_vi, epson)
        flag1 = JS_Cha2(PHI_vi, phi_vij, epson2)

        # print("flag：",flag,"flag1: ",flag1)

        if flag and flag1:
            return result
        else:
            return None

def JS_H_H_Pair(nodes, nType, tree, epson, tau2, epson2=0.1):
    """
    根据输入节点对和距离矩阵 D 判断同胞对或 remain_child 对或 sabling_pair。
    
    参数:
      nodes: list/tuple, 包含两个节点名称 [u, w] （其中 u 为第一个，w 为第二个）
      nType: 整数，决定返回哪种类型的结果
             当 nType==0 时，返回 'remain_child' 类型结果（可能有两种方向）；
             当 nType!=0 时，返回 'sabling_pair' 类型结果。
      tree: dict, 树结构，必须包含以下键：
            "D": 信息距离矩阵 (numpy array)，索引是 0-based；
            "obs_nodes": list，所有可观测节点名称（1-based，字符串）；
            "des_nodes": dict，映射每个节点到其后代节点列表。
      epson: float，阈值参数；
      tau2: float，距离阈值 tau2；
      epson2: float，次级阈值，默认 0.1。
      
    返回:
      如果满足条件，则返回一个包含一行的 pandas DataFrame，列为 'node1', 'node2', 'type'；
      否则返回 None。
    """
    u, w = nodes
    D = tree["D"]
    
    # print("JS_H_H_Pair(nodes, nType:",u,w,nType)
    # 构造三个候选结果 DataFrame
    result1 = pd.DataFrame({"node1": [u], "node2": [w], "type": ["remain_child"]})
    result2 = pd.DataFrame({"node1": [w], "node2": [u], "type": ["remain_child"]})
    result3 = pd.DataFrame({"node1": [u], "node2": [w], "type": ["sabling_pair"]})
    
    obs_nodes = tree["obs_nodes"]
    # 处理 1-based 到 0-based 的索引转换
    obs_nodes_index = np.array([int(x) - 1 for x in obs_nodes])

    # temp: 联合 u 的后代和 w 的后代（取并集）
    temp = set(tree["des_nodes"].get(u, [])) | set(tree["des_nodes"].get(w, []))
    # M: obs_nodes 中去除 temp 的部分，即非 u、w 子节点的显节点
    M = [node for node in obs_nodes if node not in temp]
    
    # bifur(u, D, tree) 返回一对节点列表，bifure已经转为 0-based
    ij = bifur(u, D, tree)
    if ij is None or len(ij) < 2:
        return None
    i_node, j_node = int(ij[0]), int(ij[1])
    
    # 对 w 调用 bifur
    kl = bifur(w, D, tree)
    if kl is None or len(kl) < 2:
        return None
    k_node, l_node = int(kl[0]), int(kl[1])

    # print("i,j,k,l: ",i_node, j_node,k_node, l_node)
    
    # 构造 logi1: 对 obs_nodes 判断是否在 M 中
    logi1 = pd.Series([node in M for node in obs_nodes], index=obs_nodes)

    # 计算 logi2
    logi2 = (D[i_node, obs_nodes_index] <= tau2) & (D[k_node, obs_nodes_index] <= tau2)
    logi = logi1 & logi2  # 结合两个条件

    # 计算 phi_ikj 和 phi_ikl
    phi_ikj = D[i_node, j_node] - D[k_node, j_node]
    phi_ikl = D[i_node, l_node] - D[k_node, l_node]
    
    # 若有满足条件的列（logi 为 True 的列数不为 0）
    if logi.sum() != 0:
        # 取出满足条件的列标签，并转换为 0-based
        cols_logi = np.array([int(x) - 1 for x in logi.index[logi]])

        # 计算 PHI_ik
        PHI_ik = D[i_node, cols_logi] - D[k_node, cols_logi]

        # print("PHI_ik: ",PHI_ik)

        flag = JS_Cha(PHI_ik, epson)
        flag11 = JS_Cha1(PHI_ik, phi_ikl, epson)
        flag12 = JS_Cha2(PHI_ik, phi_ikj, epson2)
        
        flag21 = JS_Cha2(PHI_ik, phi_ikl, epson2)
        flag22 = JS_Cha1(PHI_ik, phi_ikj, epson)
    else:
        # 若没有满足条件的列，则后续逻辑无法进行
        return None
    
    
    # print("flag: ",flag,"flag11: ",flag11,"flag12: ",flag12,"flag21: ",flag21,"flag22: ",flag22)

    if nType == 0:
        if flag and flag11 and flag12:
            return result1
        elif flag and flag21 and flag22:
            return result2
        else:
            return None
    else:
        if flag and flag12 and flag21:
            return result3
        else:
            return None

    return None  # 此行通常不会执行



def JS_View_View_Pair(tree, epson, tau1, tau2):
    """
    根据信息距离和阈值，从树的独立节点集合 tree["A"] 中构造所有可能的节点对，
    对每个节点对调用 JS_V_V_Pair 得到满足条件的同胞对（sab_pair），
    然后利用网络图计算连通分量（即最大同胞组），并更新树结构（调用 Update_tree）。
    
    参数:
      tree: dict，包含至少以下键： "A" (独立节点列表), "D" (信息距离矩阵，pandas DataFrame),
            "obs_nodes" (可观测节点列表) 以及后续 Update_tree 所需字段。
      epson: float，用于 JS_V_V_Pair 的阈值参数
      tau1: float，距离 tau1 阈值
      tau2: float，距离 tau2 阈值
      
    返回:
      更新后的 tree（dict）
    """
    # 设置 tree.empty 为 True（表示本步骤认为 tree 是空的）
    tree["empty"] = True
    
    A = tree["A"]
    D = tree["D"]  # 假设 D 为 pandas DataFrame，行列标签为节点名称
    # 构造所有组合（两两组合），得到一个二维数组，每行包含两个节点名称
    pair = np.array(list(combinations(A, 2)))
    
    # print("pair: ",pair)
    # 对每一对节点调用 JS_V_V_Pair 函数，传入参数 D, epson, tau1, tau2
    sab_pair_list = [JS_V_V_Pair(row, D, epson, tau1, tau2) for row in pair]
    # 过滤掉返回 None 的结果
    sab_pair_list = [df for df in sab_pair_list if df is not None]
    
    # 如果没有任何满足条件的对，则直接返回 tree
    if len(sab_pair_list) == 0:
        return tree
    
    # 将各结果 DataFrame 合并成一个 DataFrame（假设它们均具有相同的列 'node1','node2'）
    sab_pair_df = pd.concat(sab_pair_list, ignore_index=True)

    # print("sab_pair_df: ",sab_pair_df)

    # 利用 sab_pair_df 构造无向图
    # 注意：R 中 graph.data.frame(sab_pair) 默认创建无向图
    g = nx.from_pandas_edgelist(sab_pair_df, source="node1", target="node2")
    
    # 计算连通分量（每个连通分量为一个集合）
    components = list(nx.connected_components(g))
    # 此处将连通分量作为最大同胞组 sab_groups
    # 让每个连通分量内部的节点按照数值顺序排序
    sab_groups = [sorted(comp) for comp in components]

    print("sab_groups in JS_View_View_Pair: ",sab_groups)
    
    # 调用 Update_tree 更新树结构（假设已定义 update_tree 函数）
    tree = update_tree(sab_groups, tree)

    # print("tree[pa]: ",tree["pa"])
    
    return tree


def JS_View_Hide_Pair(tree, epson, tau2, epson2):
    """
    根据树结构 tree、距离阈值 tau2 及误差参数 epson、epson2，
    从独立节点集合 tree["A"] 中构造所有可能的节点对，
    筛选出第一个节点在可观测节点集合中而第二个在隐节点集合中的对，
    对每个候选对调用 JS_V_H_Pair 判断是否满足条件，
    最后将得到的边集构造成图，根据连通分量获得同胞组，
    并调用 update_tree 更新树结构。
    
    参数:
      tree: dict, 树结构，至少包含 "A", "obs_nodes", "lat_nodes", "D" 等键。
      epson: float, JS_V_H_Pair 中用于判断向量是否接近的阈值
      tau2: float, 距离阈值 tau2
      epson2: float, 次级误差阈值
      
    返回:
      更新后的 tree（dict）。如果没有候选对，则直接返回原 tree。
    """
    # 标记此次步骤 tree 被更新为“空”（empty True）
    tree["empty"] = True
    
    A = tree["A"]
    D = tree["D"]  # D 为 pandas DataFrame
    # 生成 A 中所有两两组合，转换为 numpy 数组，形状 (n, 2)
    pairs = np.array(list(combinations(A, 2)))
    
    # logi1: 第一列节点是否属于可观测节点
    logi1 = np.array([node in tree["obs_nodes"] for node in pairs[:, 0]])
    # logi2: 第二列节点是否属于隐节点
    logi2 = np.array([node in tree["lat_nodes"] for node in pairs[:, 1]])
    logi = logi1 & logi2
    
    if np.sum(logi) == 0:
        return tree
    
    # 筛选符合条件的候选对
    pairs = pairs[logi, :]
    
    # 对每个候选对调用 JS_V_H_Pair，并收集结果（JS_V_H_Pair 返回一个 DataFrame 或 None）
    sab_pair_list = []
    for pair in pairs:
        res = JS_V_H_Pair(pair, tree, epson, tau2, epson2)
        if res is not None:
            sab_pair_list.append(res)
    
    if len(sab_pair_list) == 0:
        return tree
    
    # 将所有非空结果合并成一个 DataFrame
    sab_pair_df = pd.concat(sab_pair_list, ignore_index=True)
    
    # 构造无向图：使用 sab_pair_df 中的 'node1' 和 'node2' 列
    g = nx.from_pandas_edgelist(sab_pair_df, source='node1', target='node2')
    
    # 计算连通分量（每个连通分量为一个集合，即最大同胞组）
    sab_groups = list(nx.connected_components(g))
    
    print("sab_groups in JS_View_Hide_Pair:",sab_groups)
    # 使用更新函数更新 tree 结构（假设 update_tree 已实现）
    tree = update_tree(sab_groups, tree)
    
    return tree


def JS_Hide_Hide_Pair(nType, tree, epson, tau2, epson2=0.1):
    """
    根据树结构 tree，从独立节点集合 tree["A"] 中生成所有两两组合，
    筛选出其中两个节点均为隐节点的组合，然后对每一对调用 JS_H_H_Pair，
    将返回的结果合并后，根据 nType 的不同进行后续处理，
    最后更新并返回树结构。

    参数:
      tree: dict，树结构，包含键 "A", "lat_nodes", "obs_nodes", "D" 等
      epson: float，阈值参数，传递给 JS_H_H_Pair
      tau2: float，距离阈值 tau2
      epson2: float，次级误差阈值（默认 0.1）
      nType: int，类型标志：
             当 nType == 0 时，返回 remain_child 类型，
             否则返回 sabling_pair 类型。
    返回:
      更新后的 tree (dict)；若无符合条件的节点对，则直接返回原 tree。
    """
    print("JS_Hide_Hide_Pair(nType:",nType)
    # 将 tree 标记为“空”，即本步骤开始时视为未更新
    tree["empty"] = True

    A = tree["A"]

    # 生成 A 中所有两两组合（独立节点对），返回二维数组，每行包含两个节点
    pair = np.array(list(combinations(A, 2)))
    
    # 筛选出两列均在隐节点集合中的组合
    lat_nodes = set(tree["lat_nodes"])
    # 对每一行，检查第1个和第2个元素是否均属于 lat_nodes
    logi = np.array([ (row[0] in lat_nodes) and (row[1] in lat_nodes) for row in pair ])
    if np.sum(logi) == 0:
        print("logi is empty, returning tree")
        return tree
    pair = pair[logi, :]  # 保留满足条件的组合

    # print("pair: ",pair)

    # 对每一行调用 JS_H_H_Pair，传入参数 nType, tree, epson, tau2, epson2
    sab_pair_list = []
    for row in pair:
        res = JS_H_H_Pair(row, nType, tree, epson, tau2, epson2)
        if res is not None:
            sab_pair_list.append(res)

    # 合并 DataFrame 确保格式整齐
    if sab_pair_list:  # 确保列表非空
        sab_pair_df = pd.concat(sab_pair_list, ignore_index=True)
        # print("sab_pair_df:\n", sab_pair_df)
    else:
        print("JS_H_H_Pair produced no results, returning tree")
        return tree



    if nType == 0:
        # print("where")
        # 对于 nType==0，去除第三列（假设第三列标识类型），得到 remain_ch
        remain_ch = sab_pair_df.drop(columns=sab_pair_df.columns[2])


        # 若一个隐节点是多个节点的子节点，则只取前一个
        unique_nodes = []
        indices = []
        for idx, val in enumerate(remain_ch.iloc[:, 0]):
            if val not in unique_nodes:
                unique_nodes.append(val)
                indices.append(idx)

        if len(unique_nodes) != len(remain_ch.iloc[:, 0]):
            remain_ch = remain_ch.iloc[indices, :]

        # print("After unique child filtering:\n", remain_ch)

        # 若一个隐节点既作为子节点又作为父节点，则只保留它作为父节点的那条
        logi = remain_ch.iloc[:, 0].isin(remain_ch.iloc[:, 1])

        if np.sum(~logi) == 0:
            # 若隐节点互相连成一个环，则只取第一条
            remain_ch = remain_ch.iloc[[0], :]
        else:
            remain_ch = remain_ch.loc[~logi]

        print("remain_ch in JS_Hide_Hide_Pair:\n ",remain_ch)
        # 调用 update_tree_remain 更新树结构
        tree = update_tree_remain(remain_ch, tree)
    else:
        # 当 nType != 0：去除第三列
        sab_pair_df = sab_pair_df.drop(columns=sab_pair_df.columns[2])
        # 构造无向图（边列表 sab_pair_df 的列名假定为 'node1' 和 'node2'）
        g = nx.from_pandas_edgelist(sab_pair_df, source='node1', target='node2')
        # 计算连通分量
        components = list(nx.connected_components(g))
        sab_groups = components  # 此处每个连通分量视为一个同胞组

        print("sab_groups in JS_Hide_Hide_Pair: ",sab_groups)
        # 调用 update_tree 更新树结构
        tree = update_tree(sab_groups, tree)
    
    return tree

def JS_Pair(D, epson, tau1, tau2, epson2=0.1):
    """
    根据距离矩阵 D 及阈值参数构造树结构，并依次调用不同函数更新树。
    
    参数:
      D: 信息距离矩阵，通常为NumPy，行列标签为节点名称
      epson: float, 用于判断距离差是否接近的阈值参数
      tau1: float, 距离 tau1 阈值
      tau2: float, 距离 tau2 阈值
      epson2: float, 次级阈值，默认 0.1
      
    返回:
      tree: 更新后的树结构（字典）
    """
    # 1. 初始化树
    tree = Init(D)

    # print("tree: ",tree)
    
    # 2. 调用 JS_View_View_Pair 更新树
    tree = JS_View_View_Pair(tree, epson=epson, tau1=tau1, tau2=tau2)


    # print("tree[lat_nodes]",tree["lat_nodes"])
    
    # 如果树中 latent 节点不为空，则进入循环更新
    if tree.get("lat_nodes"):
        while True:

            # print("where in JS_Pair()")
            A = tree["A"]
            obs_nodes = tree["obs_nodes"]
            # 取每个节点名称的首字符
            char = [x[0] for x in A]
            # logi: 标记哪些节点名称以 "H" 开头
            logi = [c == "H" for c in char]
            # flag 为：可独立节点中以 "H" 开头的数量小于等于2，且非"H"数量为0
            flag = (sum(logi) <= 2) and (sum([not x for x in logi]) == 0)
            if flag:
                break
            # 如果存在非 "H" 的节点
            if sum([not x for x in logi]) != 0:
                tree = JS_View_Hide_Pair(tree, epson=epson, tau2=tau2, epson2=epson2)
                if not tree.get("empty", False):
                    continue
            tree = JS_Hide_Hide_Pair(0, tree, epson=epson, tau2=tau2, epson2=epson2)
            if not tree.get("empty", False):
                continue
            tree = JS_Hide_Hide_Pair(1, tree, epson=epson, tau2=tau2, epson2=epson2)
            if not tree.get("empty", False):
                continue
            break

    
    # 3. 最后检查独立节点集合 A 中以 "H" 开头的数量
    A = tree["A"]
    obs_nodes = tree["obs_nodes"]
    char = [x[0] for x in A]
    logi = [c == "H" for c in char]
    if sum(logi) == 2:
        # 若正好有两个以 "H" 开头的节点，则构造一个 DataFrame表示紧相邻关系
        rem_ch = pd.DataFrame({"child": [A[1]], "parent": [A[0]]})
        tree = update_tree_remain(rem_ch, tree)
    
    return tree


# def Last_Update(tree):
#     """
#     对于所有仍然没有父节点的节点（包括观测节点和已有隐节点），
#     统一分配一个新的隐节点作为它们的共同父节点，并更新树的结构。
#     同时确保新引入的隐节点也在父节点字典中出现，其父节点值为空。

#     Args:
#         tree (dict): 当前的树结构

#     Returns:
#         dict: 更新后的树结构
#     """
#     # 1. 找出所有父节点仍为空的节点（不限定必须属于 obs_nodes）
#     remaining = [node for node, parent in tree["pa"].items() if parent == ""]
#     if len(remaining) == 0:
#         # 如果没有剩余节点，则直接返回
#         return tree

#     # 2. 创建一个新的隐节点，新隐节点的标签依据已有隐节点数量来生成
#     num_existing = len(tree.get("lat_nodes", []))
#     new_lat = f"H{num_existing + 1}"  # 新隐节点标签

#     # 3. 更新隐节点和内部节点列表
#     if "lat_nodes" not in tree or not tree["lat_nodes"]:
#         tree["lat_nodes"] = []
#     if "inter_nodes" not in tree or not tree["inter_nodes"]:
#         tree["inter_nodes"] = []
#     tree["lat_nodes"].append(new_lat)
#     tree["inter_nodes"].append(new_lat)

#     # 4. 更新结构信息 DataFrame (tree["str"])
#     st_new = pd.DataFrame({
#         "ID": [new_lat],
#         "obs_lat": ["latent"],
#         "degree": [len(remaining)],     # 新隐节点连接的子节点数
#         "ch_obs_num": [len(remaining)]   # 记录新隐节点连接的子节点数
#     }, index=[new_lat])
#     st_old = tree["str"] if isinstance(tree["str"], pd.DataFrame) else pd.DataFrame.from_dict(tree["str"], orient="index")
#     tree["str"] = pd.concat([st_old, st_new])

#     # 5. 更新父节点字典：将所有 remaining 节点的父节点设为新隐节点
#     for node in remaining:
#         tree["pa"][node] = new_lat

#     # 【新增】确保新隐节点也在父节点字典中，以便后续构造边列表时它也能作为键出现
#     tree["pa"][new_lat] = ""

#     # 6. 更新子节点字典：新隐节点的子节点为 remaining
#     tree["child"][new_lat] = remaining

#     # 7. 更新后代节点字典：新隐节点的后代为 remaining 的副本
#     tree["des_nodes"][new_lat] = remaining.copy()

#     # 8. 更新 A 和 D
#     tree["A"] = [node for node in tree["pa"] if tree["pa"][node] == ""]
#     tree["D"] = [node for node in tree["pa"] if tree["pa"][node] != ""]

#     # 9. 更新 nodes_type：新增隐节点的类型设为 "latent"
#     if "nodes_type" in tree and isinstance(tree["nodes_type"], dict):
#         tree["nodes_type"][new_lat] = "latent"
#     else:
#         tree["nodes_type"] = {new_lat: "latent"}

#     # 10. 更新 nodes_name：新增隐节点的名称与标签一致
#     if "nodes_name" in tree and isinstance(tree["nodes_name"], dict):
#         tree["nodes_name"][new_lat] = new_lat
#     else:
#         tree["nodes_name"] = {new_lat: new_lat}

#     return tree



def Last_Update(tree):
    """
    对于所有仍然没有父节点的节点（包括观测节点和已有隐节点）：
      - 如果只有 1 个孤儿，什么也不做，直接返回；
      - 如果有 2 个孤儿，令编号（数值）大的那个做编号小的父节点；
      - 如果 ≥3 个孤儿，按原来逻辑，引入一个新隐节点，把它们全都挂到新隐节点下。

    Args:
        tree (dict): 当前的树结构，至少包含以下字段：
            - tree["pa"]: dict, node->parent ("" 表示无父)
            - tree["str"]: DataFrame or dict，节点信息表
            - tree["lat_nodes"], tree["inter_nodes"]: list, 隐节点列表
            - tree["child"], tree["des_nodes"]: dict, node->children / descendants
            - tree["nodes_type"], tree["nodes_name"]: dict
    Returns:
        dict: 更新后的 tree
    """

    # print(tree)

    def _numeric_key(s):
        # 尝试把节点名解析成数字：先整数字符串，再 H+数字，再回退到原串
        try:
            return int(s)
        except:
            if s.startswith("H"):
                try:
                    return int(s[1:])
                except:
                    pass
        return s

    # 1) 找出所有还没父节点的“孤儿”
    remaining = [n for n,p in tree["pa"].items() if p == ""]

    # print("remaining: ",remaining)

    # 2) 三种情况
    if len(remaining) <= 1:
        # 0 或 1 个孤儿，不变
        return tree

    elif len(remaining) == 2:
        # 只有两个孤儿，编号大的做父
        a, b = remaining
        # 用 membership 来判断 latent vs observable
        a_lat = (a in tree.get("lat_nodes", []))
        b_lat = (b in tree.get("lat_nodes", []))

        # 情况1：一个 latent 一个 observable ⇒ latent 做父
        if a_lat and not b_lat:
            parent, child = a, b
        if b_lat and not a_lat:
            parent, child = b, a
        # 情况2：两个都是latent
        if a_lat and b_lat:
            # 这时用 numeric_key 决定谁大谁小
            if _numeric_key(a) > _numeric_key(b):
                parent, child = a, b
            else:
                parent, child = b, a


        # print("child: ",child,"parent: ",parent)
        tree["pa"][child] = parent

        # 更新 child 列表
        tree.setdefault("child", {})\
            .setdefault(parent, [])\
            .append(child)

        # 更新 descendants 列表
        # parent 的后代包括 child 及 child 原有的后代
        desc = tree.get("des_nodes", {}).get(child, []).copy()
        tree.setdefault("des_nodes", {})[parent] = [child] + desc

    else:
        # 原逻辑：≥3 个孤儿，引入一个新隐节点
        # 生成新隐节点标签
        num_existing = len(tree.get("lat_nodes", []))
        new_lat = f"H{num_existing + 1}"

        # 更新隐节点列表
        tree.setdefault("lat_nodes", []).append(new_lat)
        tree.setdefault("inter_nodes", []).append(new_lat)

        # 更新 str 表
        st_new = pd.DataFrame({
            "ID":        [new_lat],
            "obs_lat":   ["latent"],
            "degree":    [len(remaining)],
            "ch_obs_num":[len(remaining)]
        }, index=[new_lat])
        st_old = (tree["str"]
                  if isinstance(tree["str"], pd.DataFrame)
                  else pd.DataFrame.from_dict(tree["str"], orient="index"))
        tree["str"] = pd.concat([st_old, st_new])

        # 挂载所有孤儿到新隐节点
        for node in remaining:
            tree["pa"][node] = new_lat
        tree["pa"][new_lat] = ""

        # 更新 child 和 des_nodes
        tree.setdefault("child", {})[new_lat]    = remaining.copy()
        tree.setdefault("des_nodes", {})[new_lat]= remaining.copy()

    # 3) 刷新 A（根集）和 D（非根集）
    tree["A"] = [n for n,p in tree["pa"].items() if p == ""]
    tree["D"] = [n for n,p in tree["pa"].items() if p != ""]

    # 4) 确保新加节点在 nodes_type/nodes_name 中
    #    （如果是两节点情况，不会新增隐节点，这里无需额外处理）
    if "nodes_type" in tree and isinstance(tree["nodes_type"], dict):
        for n in remaining:
            if n not in tree["nodes_type"]:
                tree["nodes_type"][n] = "latent" if n.startswith("H") else tree["nodes_type"].get(n, "")
    if "nodes_name" in tree and isinstance(tree["nodes_name"], dict):
        for n in remaining:
            tree["nodes_name"].setdefault(n, n)

    # print(tree)

    return tree




def SLLT(D, tau1=2, tau2=4, epson_init=0.3, step=0.1, epson2=0.03):
    """
    根据距离矩阵 D 及各阈值参数构造树结构，并逐步调整参数 epson 直到满足条件，
    否则当 epson 超过阈值（10）时，根据 obs_nodes 构造同胞组更新树。
    
    参数:
      D: pandas DataFrame，信息距离矩阵（行列标签为节点名称）
      tau1: float，距离 tau1 阈值
      tau2: float，距离 tau2 阈值
      epson_init: float，初始 epson 值
      step: float，每次调整 epson 的步长
      epson2: float，次级阈值（默认 0.1）
      
    返回:
      result: dict，包含三个键：
              "edges": 一个 DataFrame，列为 "nodes" 与 "parent"（父节点为空表示根节点）
              "tree": 更新后的树结构（字典）
              "epson": 最终使用的 epson 值
    """
    epson = epson_init
    

    iter=0

    while True:
        # 调用 JS_Pair 得到树结构
        iter=iter+1

        tree = JS_Pair(D, epson, tau1, tau2, epson2=epson2)

        pa = tree.get("pa", {})
        st = tree.get("str")
        lat_nodes = tree.get("lat_nodes", [])
        obs_nodes = tree.get("obs_nodes", [])
        inter_nodes = tree.get("inter_nodes", [])

        # leaf_nodes: 可观测节点中不在 inter_nodes 内
        leaf_nodes = [node for node in obs_nodes if node not in inter_nodes]

        print("代码测试设置iter",iter,"epson: ",epson,"epson2: ",epson2)
        # #设置迭代次数方便调试
        # if iter==5: 
        #     break
        
        # flag1: 父节点为空的数量是否恰好为 1
        flag1 = sum(1 for v in pa.values() if v == "") == 1

        # print(st)
        
        # flag2: 对于所有隐节点，在结构信息 DataFrame st 中第三列（degree）是否都>=3
        # 这里 st.loc[lat_nodes].iloc[:,2] 获取对应行第三列
        flag2 = (st.loc[lat_nodes].iloc[:, 2] >= 3).sum() == len(lat_nodes) if len(lat_nodes) > 0 else True
        
        # flag3: 对于所有 leaf_nodes，在 st 中第三列是否都等于 1,obs_nodes中的内部节点的degree不为1
        flag3 = (st.loc[leaf_nodes].iloc[:, 2] == 1).sum() == len(leaf_nodes) if len(leaf_nodes) > 0 else True
        
        flag = flag1 and flag2 and flag3


        # print("flag: ",flag,"flag1: ",flag1,"flag2: ",flag2,"flag3: ",flag3)
        if flag:
            break
        
        # 如果 epson 超过2，则对于所有仍然没有父节点的节点（包括观测节点和已有隐节点），
        #统一分配一个新的隐节点作为它们的共同父节点，并更新树的结构。
        if epson > tau1:
            print("epson超过1.4,引入一个隐变量连接所有无父节点的顶点，构建一棵树")
            tree=Last_Update(tree)
            break
        
        epson += step
        tree = None
        gc.collect()
    



    # 最终构造边列表
    # ID 为树中所有节点（从 pa 键获得）
    ID = list(pa.keys())
    # 构造 DataFrame edges，包含 "nodes" 和 "parent"
    edges = pd.DataFrame({"nodes": ID, "parent": [pa[node] for node in ID]})
    
    result = {"edges": edges, "tree": tree, "epson": epson}
    return result


def RFdist(TreeTr, TreeEs):
    """
    计算两个无向树之间的 Robinson-Foulds (RF) 距离。

    参数:
        TreeTr: dict, 真实树结构
        TreeEs: dict, 估计的树结构

    返回:
        RF 距离 (int)

    说明:
        - 只受树结构影响，隐节点的名称不同 和 根节点不同 不会影响 RF 距离。
        - 输入的两个树必须具有相同的可观测节点。
    """
    obs_nodes1 = TreeTr["obs_nodes"]
    obs_nodes2 = TreeEs["obs_nodes"]

    # 确保两个树的可观测节点相同
    if len(obs_nodes1) != len(obs_nodes2) or sorted(obs_nodes1) != sorted(obs_nodes2):
        raise ValueError("输入的两个树必须具有相同的可观测节点!")

    obs_nodes = obs_nodes1  # 统一使用一个

    # 处理真实树 (TreeTr)
    lat_nodes1 = TreeTr["lat_nodes"]
    des_nodes1 = TreeTr["des_nodes"]

    D1 = {}
    D1_com = {}
    for lat_node in lat_nodes1:
        temp = des_nodes1.get(lat_node, [])
        temp1 = sorted(set(obs_nodes) & set(temp))
        D1[lat_node] = set(temp1)  # 使用 set() 进行存储
        temp2 = sorted(set(obs_nodes) - set(temp1))
        D1_com[lat_node] = set(temp2)

    # 处理估计树 (TreeEs)
    lat_nodes2 = TreeEs["lat_nodes"]
    des_nodes2 = TreeEs["des_nodes"]

    D2 = {}
    D2_com = {}
    for lat_node in lat_nodes2:
        temp = des_nodes2.get(lat_node, [])
        temp1 = sorted(set(obs_nodes) & set(temp))
        D2[lat_node] = set(temp1)  # 使用 set() 进行存储
        temp2 = sorted(set(obs_nodes) - set(temp1))
        D2_com[lat_node] = set(temp2)

    # 对 D1 和 D2 中每个隐节点的集合按数值大小排序后输出
    print("D1 (Sorted):")
    for lat_node, vertices in D1.items():
        sorted_vertices = sorted(vertices, key=lambda x: int(x))  # 按数值排序
        print(f"{lat_node}: {sorted_vertices}")

    print("\nD2 (Sorted):")
    for lat_node, vertices in D2.items():
        sorted_vertices = sorted(vertices, key=lambda x: int(x))  # 按数值排序
        print(f"{lat_node}: {sorted_vertices}")

    # 计算 RF 距离
    flag1 = [D1[lat] in D2.values() or D1_com[lat] in D2.values() for lat in D1]
    flag2 = [D2[lat] in D1.values() or D2_com[lat] in D1.values() for lat in D2]

    print("flag1: ",flag1)
    print("flag2: ",flag2)

    RF_distance = (sum(1 - np.array(flag1)) + sum(1 - np.array(flag2))) / 2

    return RF_distance

