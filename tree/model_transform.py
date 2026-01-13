# 功能描述：结构演化
import pandas as pd
from collections import deque
import os
import numpy as np
import math
import random
import datetime as dt
from tree.online_PE_functions import (tree_from_edge)
from tree.online_SL_functions_for_realdata import (RFdist)

# 设置全局随机种子，确保结果可重复
RANDOM_SEED = 123
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def calculate_path_lengths(model):
    """
    计算模型中所有节点对之间的最短路径长度（支持双向路径）
    不连通的节点对返回0
    """
    # 构建双向邻接表
    adjacency = {}

    # 添加子节点->父节点的边
    for _, row in model.iterrows():
        node = row['vertice']
        father = row['father_vertice']
        if father:
            if node not in adjacency:
                adjacency[node] = []
            adjacency[node].append(father)

    # 添加父节点->子节点的边（双向图）
    child_map = {}
    for _, row in model.iterrows():
        node = row['vertice']
        father = row['father_vertice']
        if father:
            if father not in child_map:
                child_map[father] = []
            child_map[father].append(node)

    for father, children in child_map.items():
        if father not in adjacency:
            adjacency[father] = []
        adjacency[father].extend(children)

    # 获取所有节点列表
    nodes = list(model['vertice'])
    n = len(nodes)

    # 初始化距离矩阵，默认值为0（表示不连通）
    distance_matrix = pd.DataFrame(0, index=nodes, columns=nodes)

    # 对每个节点对计算最短路径
    for i in nodes:
        for j in nodes:
            if i == j:
                distance_matrix.loc[i, j] = 0  # 自身到自身的距离为0
                continue

            # BFS初始化
            visited = set()
            queue = deque([(i, 1)])  # 初始距离设为1（避免与默认值0冲突）
            visited.add(i)
            path_length = 0  # 默认不连通

            while queue:
                current_node, dist = queue.popleft()

                # 找到目标节点
                if current_node == j:
                    path_length = dist
                    break

                # 处理当前节点的所有邻居（双向）
                if current_node in adjacency:
                    for neighbor in adjacency[current_node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, dist + 1))

            distance_matrix.loc[i, j] = path_length

    return distance_matrix


def save_matrix(matrix, directory='.', csv_filename='path_lengths.csv'):
    """
    将路径长度矩阵保存为CSV文件

    参数:
    matrix (pd.DataFrame): 路径长度矩阵
    directory (str): 保存文件的目录
    csv_filename (str): CSV文件名
    """
    # 确保目录存在
    os.makedirs(directory, exist_ok=True)

    # 构建完整路径
    csv_path = os.path.join(directory, csv_filename)

    try:
        # 保存为CSV
        matrix.to_csv(csv_path)
        # print(f"路径长度矩阵已保存到 {csv_path}")
    except Exception as e:
        print(f"保存CSV文件时出错: {e}")


def generate_random_matrix(stable_matrix):
    """
    生成一个随机矩阵，大小与stable_matrix相同，对角线元素为0，非对角线元素为[-1,1]之间的随机数
    （移除了内部固定种子，依赖外部种子设置）
    """
    n = stable_matrix.shape[0]
    # 直接生成随机数，无需重复设置种子
    random_matrix = np.random.uniform(-1, 1, size=(n, n))
    np.fill_diagonal(random_matrix, 0)
    return random_matrix


def evolve_matrix(original_matrix, stable_matrix, t):
    """使用ODE思想演化距离矩阵，添加随机扰动增强变化"""
    # 提取DataFrame的数值部分（转换为NumPy数组）
    new_matrix = stable_matrix + (original_matrix - stable_matrix) * np.exp(-0.05*t)
    return new_matrix


def find_h_parents(matrix_df):
    """
    从路径长度矩阵中找出每个观测节点的最佳隐变量父节点

    参数:
    matrix_df (pd.DataFrame): 路径长度矩阵

    返回:
    pd.DataFrame: 包含节点及其对应父节点的DataFrame
    """
    # 提取所有H列的列名
    h_columns = [col for col in matrix_df.columns if col.startswith('H')]

    result = []
    # 遍历所有非H节点（1-13）
    for node in matrix_df.index:
        if not node.startswith('H'):
            # 获取当前节点在H列的值，并找到最小值对应的H节点
            row_values = matrix_df.loc[node, h_columns]
            min_h_node = row_values.idxmin()

            result.append({
                'vertice': node,  # 保持为字符串类型
                'vertice_type': 'observable_continuous',
                'father_vertice': min_h_node
            })

    # 按模型示例的格式整理结果
    df_result = pd.DataFrame(result, columns=['vertice', 'vertice_type', 'father_vertice'])
    return df_result


def filter_h_rows_and_columns(matrix):
    """
    输入一个DataFrame，第一行和第一列为标签，
    输出仅保留行列标签包含'H'的子矩阵
    """
    # 处理空DataFrame的情况
    if matrix.empty:
        return pd.DataFrame()

    # 提取列标签和行标签
    col_labels = matrix.columns
    row_labels = matrix.index

    # 筛选包含'H'的列和行
    cols_to_keep = [lbl for lbl in col_labels if 'H' in str(lbl)]
    rows_to_keep = [lbl for lbl in row_labels if 'H' in str(lbl)]

    # 如果没有找到符合条件的行列，返回空DataFrame
    if not cols_to_keep or not rows_to_keep:
        return pd.DataFrame()

    # 提取筛选后的子矩阵
    result = matrix.loc[rows_to_keep, cols_to_keep]

    return result


def find(parent, i):
    """查找节点i的根节点，并进行路径压缩"""
    if parent[i] == i:
        return i
    parent[i] = find(parent, parent[i])
    return parent[i]


def union(parent, rank, x, y):
    """合并两个集合"""
    xroot = find(parent, x)
    yroot = find(parent, y)

    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1


def build_mst(weight_matrix):
    """根据权重矩阵构建最小生成树"""
    n = len(weight_matrix)
    edges = []

    # 收集所有边及其权重
    for i in range(n):
        for j in range(i + 1, n):
            weight = weight_matrix[i][j]
            if weight != 0:
                edges.append((weight, i, j))

    # 按权重升序排序
    edges.sort(key=lambda x: x[0])

    # 初始化并查集
    parent = list(range(n))
    rank = [0] * n
    mst = []

    # Kruskal算法
    for edge in edges:
        weight, u, v = edge
        x = find(parent, u)
        y = find(parent, v)

        if x != y:
            mst.append((weight, u, v))
            union(parent, rank, x, y)

        if len(mst) == n - 1:
            break

    return mst


def generate_mst_table(weight_matrix, root_node=0):
    """生成MST的节点关系表格"""
    # 处理输入为DataFrame的情况
    if isinstance(weight_matrix, pd.DataFrame):
        weight_array = weight_matrix.values
        node_names = list(weight_matrix.index)
    else:
        n = len(weight_matrix)
        node_names = [f"H{i + 1}" for i in range(n)]
        weight_array = weight_matrix

    # 构建MST
    mst = build_mst(weight_array)

    # 调试输出：打印MST的边
    mst_edges = [(node_names[u], node_names[v], weight) for weight, u, v in mst]
    # print(f"MST边: {mst_edges[:3]}...")  # 只打印前3条边，避免过多输出

    # 构建邻接表
    adj = {i: [] for i in range(len(weight_array))}
    for weight, u, v in mst:
        adj[u].append((v, weight))
        adj[v].append((u, weight))

    # 处理根节点
    if isinstance(root_node, str) and root_node.startswith('H'):
        root_node = int(root_node[1:]) - 1

    # BFS遍历构建父子关系
    parent_indices = {root_node: None}
    queue = [root_node]

    while queue:
        current_idx = queue.pop(0)
        for neighbor_idx, _ in adj[current_idx]:
            if neighbor_idx not in parent_indices:
                parent_indices[neighbor_idx] = current_idx
                queue.append(neighbor_idx)

    # 生成数据行
    data = []
    for i in range(len(weight_array)):
        node_name = node_names[i]
        parent_idx = parent_indices.get(i, None)
        father_name = node_names[parent_idx] if parent_idx is not None else ""
        data.append({
            'vertice': node_name,
            'vertice_type': 'latent',
            'father_vertice': father_name
        })

    return pd.DataFrame(data)


def merge_models(model_only_O, model_only_H):
    """
    将model_only_O和model_only_H合并为model_OH

    参数:
    model_only_O (list): 观测顶点数据列表，每个元素是包含'vertice', 'vertice_type', 'father_vertice'的字典
    model_only_H (list): 隐变量顶点数据列表，每个元素是包含'vertice', 'vertice_type', 'father_vertice'的字典

    返回:
    list: 合并后的顶点数据列表，先观测顶点后隐变量顶点
    """
    # 先添加所有观测顶点
    model_OH = model_only_O.copy()

    # 再添加所有隐变量顶点
    model_OH.extend(model_only_H)

    return model_OH


def process_tree_structure(df):
    processed_df = df.copy()
    # 确保节点和父节点为字符串类型
    processed_df['vertice'] = processed_df['vertice'].astype(str)
    processed_df['father_vertice'] = processed_df['father_vertice'].astype(str)

    max_iterations = 1000  # 防止无限循环的安全上限
    iteration_count = 0
    nodes_removed = True

    while nodes_removed and iteration_count < max_iterations:
        iteration_count += 1
        nodes_removed = False
        prev_row_count = len(processed_df)

        # 构建邻接表
        adjacency = {}
        for _, row in processed_df.iterrows():
            node = row['vertice']
            father = row['father_vertice']
            if node not in adjacency:
                adjacency[node] = set()
            if father:
                if father not in adjacency:
                    adjacency[father] = set()
                adjacency[node].add(father)
                adjacency[father].add(node)

        node_to_father = {str(row['vertice']): str(row['father_vertice'])
                          for _, row in processed_df.iterrows()}

        # 处理度为1的H节点
        h_degree_1 = [node for node in adjacency
                      if node.startswith('H') and len(adjacency[node]) == 1]

        if h_degree_1:
            nodes_removed = True
            # 删除度为1的H节点
            for node in h_degree_1:
                neighbor = adjacency[node].pop()
                if node in adjacency[neighbor]:
                    adjacency[neighbor].remove(node)
            processed_df = processed_df[~processed_df['vertice'].isin(h_degree_1)]

        # 重新构建邻接表（节点已变化）
        adjacency = {}
        for _, row in processed_df.iterrows():
            node = row['vertice']
            father = row['father_vertice']
            if node not in adjacency:
                adjacency[node] = set()
            if father:
                if father not in adjacency:
                    adjacency[father] = set()
                adjacency[node].add(father)
                adjacency[father].add(node)

        # 处理度为2的H节点（按层级从低到高处理）
        h_degree_2 = [node for node in adjacency
                      if node.startswith('H') and len(adjacency[node]) == 2]

        # 计算每个节点的层级（到根的距离）
        node_level = {}
        for node in h_degree_2:
            level = 0
            current = node
            while current in node_to_father and node_to_father[current]:
                level += 1
                current = node_to_father[current]
            node_level[node] = level

        # 按层级从高到低排序（叶节点优先处理）
        h_degree_2_sorted = sorted(h_degree_2, key=lambda x: -node_level.get(x, 0))

        nodes_to_remove = set()
        for node in h_degree_2_sorted:
            if node in nodes_to_remove:
                continue  # 已被上游处理删除

            father = node_to_father.get(node, '')
            if not father or father in nodes_to_remove:
                continue  # 没有父节点或父节点已被删除

            # 获取H节点的子节点（除父节点外的邻居）
            neighbors = list(adjacency[node])
            children = [n for n in neighbors if n != father]

            # 检查子节点是否会连接到自身（避免循环）
            valid_children = []
            for child in children:
                # 确保子节点的新父节点不是自己
                if father != child:
                    valid_children.append(child)
                    # 更新子节点的父节点
                    processed_df.loc[processed_df['vertice'] == child, 'father_vertice'] = father

            # 标记当前节点为待删除
            nodes_to_remove.add(node)

        # 删除度为2的H节点
        if nodes_to_remove:
            nodes_removed = True
            processed_df = processed_df[~processed_df['vertice'].isin(nodes_to_remove)]

        # 安全检查：如果行数没有变化，退出循环
        if len(processed_df) == prev_row_count:
            break

    # 检查是否达到最大迭代次数
    if iteration_count >= max_iterations:
        print(f"警告：达到最大迭代次数 {max_iterations}，可能存在循环引用")

    return processed_df


def generate_deterministic_stable_matrix(original_matrix):
    """
    生成确定性的稳态矩阵，H-H之间的距离缩放程度基于节点编号计算，而非随机生成
    """
    stable_matrix = pd.DataFrame(index=original_matrix.index, columns=original_matrix.columns)

    for i in stable_matrix.index:
        for j in stable_matrix.columns:
            if i.startswith('H') and j.startswith('H'):
                # 提取H后的数字部分并转换为整数
                i_num = int(i[1:])  # 例如'H3'转换为3
                j_num = int(j[1:])

                # 使用节点编号计算确定性的缩放因子（范围0.2-0.9）
                # 这里使用正弦函数确保结果在范围内波动
                scale_factor = 0.2 + 0.7 * abs(math.sin(i_num * j_num))

                # 应用缩放因子
                stable_matrix.loc[i, j] = original_matrix.loc[i, j] * scale_factor
            else:
                # O-H或O-O之间的距离统一缩放
                stable_matrix.loc[i, j] = original_matrix.loc[i, j] * 0.5

    # 确保对角线为0
    np.fill_diagonal(stable_matrix.values, 0)
    return stable_matrix


def models(modelname):
    if modelname == 'model3':
        # model3定义
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

    if modelname == 'stable1':
        # stable1
        # 定义节点名称（1-13 为观测节点，H1-H8 为隐节点）
        nodes_name = [str(i) for i in range(1, 14)] + [f"H{i}" for i in range(1, 9)]

        # 定义父节点（按照 R 代码顺序）
        father_vertice = ["H2", "H2", "H4", "H5", "H5", "H5", "H6", "H6", "H7", "H7", "H8", "H8", "H8",
                          "", "H1", "H1", "H1", "H1", "H3", "H3", "H4"]
        # 导数第四个"H2"变成"H1"
        # 定义节点类型
        vertice_type = (["observable_continuous"] * 13 + ["latent"] * 8)

        # 创建 DataFrame
        model = pd.DataFrame({"vertice": nodes_name, "vertice_type": vertice_type, "father_vertice": father_vertice})
        model.index = nodes_name

    if modelname == 'stable2':
        # stable2
        # 定义节点名称（1-13 为观测节点，H1-H8 为隐节点）
        nodes_name = [str(i) for i in range(1, 14)] + [f"H{i}" for i in range(1, 9)]

        # 定义父节点（按照 R 代码顺序）
        father_vertice = ["H2", "H2", "H4", "H5", "H5", "H5", "H6", "H6", "H7", "H7", "H8", "H8", "H8",
                          "", "H1", "H1", "H1", "H3", "H3", "H3", "H4"]
        # 导数第四个"H2"变成"H3"
        # 定义节点类型
        vertice_type = (["observable_continuous"] * 13 + ["latent"] * 8)

        # 创建 DataFrame
        model = pd.DataFrame({"vertice": nodes_name, "vertice_type": vertice_type, "father_vertice": father_vertice})
        model.index = nodes_name

    if modelname == 'stable3':
        # stable3
        # 定义节点名称（1-13 为观测节点，H1-H8 为隐节点）
        nodes_name = [str(i) for i in range(1, 14)] + [f"H{i}" for i in range(1, 9)]

        # 定义父节点（按照 R 代码顺序）
        father_vertice = ["H2", "H2", "H4", "H5", "H5", "H5", "H6", "H6", "H7", "H7", "H8", "H8", "H8",
                          "", "H3", "H1", "H1", "H2", "H3", "H3", "H4"]
        # 导数第七个"H2"变成"H3"

        # 定义节点类型
        vertice_type = (["observable_continuous"] * 13 + ["latent"] * 8)

        # 创建 DataFrame
        model = pd.DataFrame({"vertice": nodes_name, "vertice_type": vertice_type, "father_vertice": father_vertice})
        model.index = nodes_name

    if modelname == 'stable4':
        # stable4
        # 定义节点名称（1-13 为观测节点，H1-H7 为隐节点）
        nodes_name = [str(i) for i in range(1, 14)] + [f"H{i}" for i in range(1, 8)]
        # 减少一个H
        # 定义父节点（按照 R 代码顺序）
        father_vertice = ["H2", "H2", "H4", "H5", "H5", "H5", "H6", "H6", "H7", "H7", "H4", "H4", "H4",
                          "", "H1", "H1", "H1", "H2", "H3", "H3"]

        # 定义节点类型
        vertice_type = (["observable_continuous"] * 13 + ["latent"] * 7)

        # 创建 DataFrame
        model = pd.DataFrame({"vertice": nodes_name, "vertice_type": vertice_type, "father_vertice": father_vertice})
        model.index = nodes_name

    if modelname == 'stable5':
        # stable5
        # 定义节点名称（1-13 为观测节点，H1-H8 为隐节点）
        nodes_name = [str(i) for i in range(1, 14)] + [f"H{i}" for i in range(1, 9)]
        # 变动很大
        # 定义父节点（按照 R 代码顺序）
        father_vertice = ["H3", "H3", "H4", "H4", "H5", "H5", "H6", "H6", "H7", "H7", "H8", "H8", "H8",
                          "", "H1", "H1", "H1", "H1", "H1", "H2", "H2"]

        # 定义节点类型
        vertice_type = (["observable_continuous"] * 13 + ["latent"] * 8)

        # 创建 DataFrame
        model = pd.DataFrame({"vertice": nodes_name, "vertice_type": vertice_type, "father_vertice": father_vertice})
        model.index = nodes_name

    if modelname == 'stable6':
        # 定义节点名称（1-13 为观测节点，H1-H8 为隐节点）
        nodes_name = [str(i) for i in range(1, 14)] + [f"H{i}" for i in range(1, 9)]
        # 变动很大
        # 定义父节点（按照 R 代码顺序）
        father_vertice = ["H1", "H4", "H5", "H5", "H6", "H6", "H7", "H7", "H8", "H8", "H8", "H8", "H8",
                          "", "H1", "H1", "H2", "H2", "H3", "H3", "H4"]

        # 定义节点类型
        vertice_type = (["observable_continuous"] * 13 + ["latent"] * 8)

        # 创建 DataFrame
        model = pd.DataFrame({"vertice": nodes_name, "vertice_type": vertice_type, "father_vertice": father_vertice})
        model.index = nodes_name



    return model


def model_transform(original_matrix, stable_matrix, t, RANDOM_SEED=None):
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # 权重矩阵演化
    new_matrix = evolve_matrix(original_matrix, stable_matrix, t)

    # O和H的连接
    model_only_O_df = find_h_parents(new_matrix)
    model_only_O = model_only_O_df.to_dict('records')

    # 提取H的子矩阵
    new_matrix_H = filter_h_rows_and_columns(new_matrix)

    # 生成H的最小生成树
    model_only_H_df = generate_mst_table(new_matrix_H)
    model_only_H = model_only_H_df.to_dict('records')

    # 合并模型并保留vertice列，同时设置索引为vertice的值
    model_OH = merge_models(model_only_O, model_only_H)
    df_model_OH = pd.DataFrame(model_OH)  # 直接生成DataFrame，保留vertice列
    df_model_OH.index = df_model_OH['vertice']  # 将索引设置为vertice列的值，**不删除原列**

    # 删除节点并保持索引和vertice列
    df_processed_model_OH = process_tree_structure(df_model_OH)
    return df_processed_model_OH