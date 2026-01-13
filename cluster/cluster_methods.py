import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score, f1_score,
    adjusted_mutual_info_score, homogeneity_score, completeness_score, v_measure_score,
    precision_score, recall_score
)
import time
from scipy.optimize import linear_sum_assignment
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
import os

# 评价指标缩写映射
METRIC_ABBREVIATIONS = {
    'silhouette_score': 'Sil',
    'calinski_harabasz_score': 'CH',
    'davies_bouldin_score': 'DB',
    'adjusted_rand_score': 'ARI',
    'normalized_mutual_info': 'NMI',
    'adjusted_mutual_info': 'AMI',
    'homogeneity_score': 'HS',
    'completeness_score': 'CS',
    # 'v_measure_score': 'VS',
    'f1_macro_score': 'F1m',
    'f1_micro_score': 'F1μ',
    # 'purity': 'Pur',
    'accuracy': 'Acc'
}

# 指标类型映射
METRIC_TYPES = {
    'silhouette_score': '内部指标',
    'calinski_harabasz_score': '内部指标',
    'davies_bouldin_score': '内部指标',
    'adjusted_rand_score': '外部指标',
    'normalized_mutual_info': '外部指标',
    'adjusted_mutual_info': '外部指标',
    # 'v_measure_score': '外部指标',
    # 'purity': '外部指标',
    'accuracy': '外部指标'
}

# 指标排序优先级（保留常用指标）
METRIC_ORDER = [
    'Sil', 'CH', 'DB',  # 内部指标在前
    'ARI', 'NMI', 'Acc',  # 外部指标在后，仅保留常用的
    # 'VS', 'Pur',
]

# 保留的指标列表（用于结果筛选）
RETAINED_METRICS = [
    'silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score',
    'adjusted_rand_score', 'normalized_mutual_info', 'accuracy',
    # 'v_measure_score', 'purity'
]


def extract_second_values(marginal_prob, num):
    def extract_second0(lst):
        if isinstance(lst, np.ndarray):
            return float(lst[0])
        elif isinstance(lst, str):
            return float(lst.split()[0].rstrip('['))
        else:
            return lst  # 处理非数组/字符串情况（如直接是数值）

    def extract_second1(lst):
        if isinstance(lst, np.ndarray):
            return float(lst[1])
        elif isinstance(lst, str):
            return float(lst.split()[1].rstrip(']'))
        else:
            return lst  # 处理非数组/字符串情况（如直接是数值）

    # 对 DataFrame 的每个元素应用提取函数，同时保留索引
    if num == 0:
        marginal_prob = marginal_prob.applymap(extract_second0)
    elif num == 1:
        marginal_prob = marginal_prob.applymap(extract_second1)
    else:
        raise ValueError("num 必须为 0 或 1")
    return marginal_prob  # 直接返回处理后的 DataFrame（保留原始索引）


# ======================
# 基础接口定义
# ======================
class ClusterAlgorithm(ABC):
    @abstractmethod
    def fit_predict(self, data: np.ndarray, num_clusters: int) -> np.ndarray:
        pass


# ======================
# 具体聚类算法实现
# ======================
class KMeansCluster(ClusterAlgorithm):
    def __init__(self, num_init: int = 10, random_state: int = 42):
        self.num_init = num_init
        self.random_state = random_state

    def fit_predict(self, data: np.ndarray, num_clusters: int) -> np.ndarray:
        return KMeans(n_clusters=num_clusters, n_init=self.num_init, random_state=self.random_state).fit_predict(data)


class GMMCluster(ClusterAlgorithm):
    def __init__(self, num_init: int = 1, random_state: int = 42):
        self.num_init = num_init
        self.random_state = random_state

    def fit_predict(self, data: np.ndarray, num_clusters: int) -> np.ndarray:
        return GaussianMixture(n_components=num_clusters, n_init=self.num_init, random_state=self.random_state).fit(
            data).predict(data)


class SpectralCluster(ClusterAlgorithm):
    def __init__(self, num_neighbors: int = 10, affinity: str = 'nearest_neighbors', random_state: int = 42):
        self.num_neighbors = num_neighbors
        self.affinity = affinity
        self.random_state = random_state

    def fit_predict(self, data: np.ndarray, num_clusters: int) -> np.ndarray:
        return SpectralClustering(
            n_clusters=num_clusters,
            affinity=self.affinity,
            n_neighbors=self.num_neighbors if self.affinity == 'nearest_neighbors' else None,
            assign_labels='kmeans',
            random_state=self.random_state
        ).fit_predict(data)


class SinkhornKnoppCluster:
    def __init__(self, epsilon: float = 0.01, max_iter: int = 100, threshold: float = 1e-6, random_state: int = 42):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.threshold = threshold
        self.random_state = random_state

    def fit_predict(self, data: np.ndarray, num_clusters: int) -> np.ndarray:
        n_samples, n_features = data.shape

        # 1. 初始化聚类中心（用K-means预热）
        # 修正：添加random_state参数
        kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=self.random_state)
        centers = kmeans.fit(data).cluster_centers_

        # 2. 计算代价矩阵（欧式距离平方）
        C = np.zeros((n_samples, num_clusters))
        for i in range(n_samples):
            for j in range(num_clusters):
                C[i, j] = np.sum((data[i] - centers[j]) ** 2)

        # 3. 归一化代价矩阵（避免数值问题）
        C = (C - C.min()) / (C.max() - C.min() + 1e-8)

        # 4. 构建Gibbs核（指数核）
        K = np.exp(-C / self.epsilon)

        # 5. 初始化边际分布（均匀分布）
        a = np.ones(n_samples) / n_samples  # 行边际（样本分布）
        b = np.ones(num_clusters) / num_clusters  # 列边际（聚类分布）

        # 6. Sinkhorn-Knopp迭代（核心步骤）
        u = np.ones(n_samples)  # 行缩放因子初始化
        for iter in range(self.max_iter):
            u_prev = u.copy()  # 保存上一次的u，用于收敛检查

            # 交替更新v和u（标准Sinkhorn迭代）
            v = b / (K.T @ u + 1e-10)  # 更新列缩放因子v
            u = a / (K @ v + 1e-10)  # 更新行缩放因子u

            # 提前终止：若u的变化小于阈值，停止迭代
            if np.linalg.norm(u - u_prev) < self.threshold:
                break

        # 7. 计算最优传输矩阵P并分配硬标签
        P = np.diag(u) @ K @ np.diag(v)  # 虽然不需要P，但计算标签需要
        labels = np.argmax(P, axis=1)  # 硬聚类：每个样本分配到概率最大的聚类

        return labels


class CustomCoClustering(ClusterAlgorithm):
    def __init__(self, view1_key: str, view2_key: str, max_iter: int = 100,
                 weight1: float = 0.5, weight2: float = 0.5, random_state: int = 42):
        self.view1_key = view1_key
        self.view2_key = view2_key
        self.max_iter = max_iter
        self.weight1 = weight1
        self.weight2 = weight2
        self.all_data = None
        self.random_state = random_state

    def set_all_data(self, all_data: Dict[str, np.ndarray]):
        self.all_data = all_data

    def fit_predict(self, data: np.ndarray, num_clusters: int) -> np.ndarray:
        if self.all_data is None:
            raise ValueError("请先调用set_all_data方法设置所有数据")

        X1 = self.all_data.get(self.view1_key)
        X2 = self.all_data.get(self.view2_key)

        if X1 is None or X2 is None:
            raise ValueError(f"找不到视图数据: {self.view1_key} 或 {self.view2_key}")

        n_samples = X1.shape[0]

        # 初始化簇中心（基于 X1 初始化）
        kmeans_init = KMeans(n_clusters=num_clusters, n_init=10, random_state=self.random_state)
        labels = kmeans_init.fit_predict(X1)

        # 为两个视图分别维护簇中心
        C1 = np.array([X1[labels == k].mean(axis=0) for k in range(num_clusters)])
        C2 = np.array([X2[labels == k].mean(axis=0) for k in range(num_clusters)])

        for iter in range(self.max_iter):
            new_labels = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                # 计算联合距离：两个视图的加权距离之和
                dist1 = np.sum((X1[i] - C1) ** 2, axis=1)
                dist2 = np.sum((X2[i] - C2) ** 2, axis=1)
                dists = self.weight1 * dist1 + self.weight2 * dist2
                new_labels[i] = np.argmin(dists)

            # 更新簇中心
            for k in range(num_clusters):
                mask = (new_labels == k)
                if np.sum(mask) == 0:
                    # 随机选择一个样本分配到该簇
                    # 修正：使用self.random_state而非固定值
                    np.random.seed(self.random_state)
                    random_idx = np.random.choice(n_samples)
                    mask = np.zeros(n_samples, dtype=bool)
                    mask[random_idx] = True
                C1[k] = X1[mask].mean(axis=0) if np.sum(mask) > 0 else C1[k]
                C2[k] = X2[mask].mean(axis=0) if np.sum(mask) > 0 else C2[k]

            # 检查收敛
            if np.all(labels == new_labels):
                break
            labels = new_labels.copy()

        return labels


# ======================
# GCN特征提取器
# ======================
class GCNFeatureExtractor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class GCNProcessor:
    def __init__(self, model_es: pd.DataFrame, data_type: str, hidden_channels: int = 16, out_channels: int = 8):
        self.model_es = model_es
        self.data_type = data_type
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_index = self.build_graph()
        # 注意：这里暂时不初始化模型，而是在首次提取特征时根据数据维度初始化
        self.gcn_model = None

    def build_graph(self) -> Optional[torch.Tensor]:
        # 保持原有代码不变
        if self.model_es.empty:
            return None

        if self.data_type == 'H' or self.data_type == 'H_std':
            latent_nodes = self.model_es[self.model_es['vertice_type'] == 'latent']['vertice'].tolist()
            filtered_df = self.model_es[
                self.model_es['vertice'].isin(latent_nodes) &
                self.model_es['father_vertice'].isin(latent_nodes)
                ]
        else:
            filtered_df = self.model_es

        if filtered_df.empty:
            return None

        # nodes = list(set(filtered_df['vertice'].tolist() + filtered_df['father_vertice'].tolist()))
        # node_to_idx = {node: i for i, node in enumerate(nodes)}
        # 1) 用稳定顺序生成 nodes（按出现顺序或字典序）
        vals = filtered_df[['vertice', 'father_vertice']].values.ravel('K')
        nodes = pd.unique(vals)  # 或者：nodes = sorted(set(vals))

        node_to_idx = {n: i for i, n in enumerate(nodes)}
        # ... 后续按 node_to_idx 生成 edges

        edges = []
        for _, row in filtered_df.iterrows():
            if pd.notna(row['vertice']) and pd.notna(row['father_vertice']):
                edges.append([node_to_idx[row['vertice']], node_to_idx[row['father_vertice']]])
                edges.append([node_to_idx[row['father_vertice']], node_to_idx[row['vertice']]])

        if not edges:
            return None

        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def extract_features(self, data: np.ndarray) -> np.ndarray:
        if self.edge_index is None:
            print("警告: 无法构建图结构，返回原始数据")
            return data

        x = torch.tensor(data, dtype=torch.float)

        # 首次调用时，根据输入数据的维度初始化GCN模型
        if self.gcn_model is None:
            in_channels = x.shape[1]  # 获取输入特征的维度
            self.gcn_model = GCNFeatureExtractor(in_channels, self.hidden_channels, self.out_channels)

        with torch.no_grad():
            embeddings = self.gcn_model(x, self.edge_index).numpy()
        return embeddings


# ======================
# GCN+聚类包装器
# ======================
class GCNClusterWrapper(ClusterAlgorithm):
    def __init__(self, model_es: pd.DataFrame, data_type: str, cluster_algorithm: ClusterAlgorithm):
        self.gcn_processor = GCNProcessor(model_es, data_type)
        self.cluster_algorithm = cluster_algorithm
        self.original_data_type = data_type  # 保存原始数据类型

    def fit_predict(self, data: np.ndarray, num_clusters: int) -> np.ndarray:
        gcn_features = self.gcn_processor.extract_features(data)
        return self.cluster_algorithm.fit_predict(gcn_features, num_clusters)

    def get_display_name(self):
        """返回格式化后的显示名称"""
        cluster_name = type(self.cluster_algorithm).__name__.replace('Cluster', '')
        return f"{self.original_data_type}+gcn", cluster_name.lower()


# ======================
# 评价指标函数
# ======================
def calculate_internal_metrics(data: np.ndarray, labels: np.ndarray) -> Dict:
    num_clusters = len(np.unique(labels))
    if num_clusters < 2:
        return {
            'silhouette_score': 0,
            'calinski_harabasz_score': 0,
            'davies_bouldin_score': float('nan')
        }

    return {
        'silhouette_score': silhouette_score(data, labels),
        'calinski_harabasz_score': calinski_harabasz_score(data, labels),
        'davies_bouldin_score': davies_bouldin_score(data, labels)
    }


def calculate_external_metrics(true_labels: np.ndarray, labels: np.ndarray) -> Dict:
    mapped_labels = map_cluster_labels(true_labels, labels)

    return {
        'adjusted_rand_score': adjusted_rand_score(true_labels, labels),
        'normalized_mutual_info': normalized_mutual_info_score(true_labels, labels),
        'adjusted_mutual_info': adjusted_mutual_info_score(true_labels, labels),
        'homogeneity_score': homogeneity_score(true_labels, labels),
        'completeness_score': completeness_score(true_labels, labels),
        # 'v_measure_score': v_measure_score(true_labels, labels),
        # 'purity': calculate_purity(true_labels, labels),
        'accuracy': calculate_accuracy(true_labels, labels)
    }


# ======================
# 辅助函数
# ======================

# def map_cluster_labels(true_labels: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
#     unique_true = np.unique(true_labels)
#     unique_cluster = np.unique(cluster_labels)
#     confusion_matrix = np.zeros((len(unique_true), len(unique_cluster)))
#
#     for i, t in enumerate(unique_true):
#         for j, c in enumerate(unique_cluster):
#             confusion_matrix[i, j] = np.sum((true_labels == t) & (cluster_labels == c))
#
#     row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
#     label_map = {col_ind[j]: row_ind[j] for j in range(len(col_ind))}
#     return np.array([label_map.get(l, -1) for l in cluster_labels])

# 你在 map_cluster_labels 里把索引当成了标签值来建表，用了 {col_ind[j]: row_ind[j]}；
# 这样 calculate_accuracy 做比较时两边标签空间不一致，Acc 可能变成 0。
# 把索引映射到真实标签值与聚类标签值即可
def map_cluster_labels(true_labels: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
    unique_true = np.unique(true_labels)
    unique_cluster = np.unique(cluster_labels)

    cm = np.zeros((len(unique_true), len(unique_cluster)))
    for i, t in enumerate(unique_true):
        for j, c in enumerate(unique_cluster):
            cm[i, j] = np.sum((true_labels == t) & (cluster_labels == c))

    # Hungarian 在“索引空间”配对
    row_ind, col_ind = linear_sum_assignment(-cm)

    # 关键：把索引转成“实际标签值”
    label_map = {unique_cluster[col]: unique_true[row] for row, col in zip(row_ind, col_ind)}

    return np.array([label_map.get(l, -1) for l in cluster_labels])



def calculate_purity(true_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, cluster_labels)
    return np.sum(np.max(cm, axis=0)) / np.sum(cm)


def calculate_accuracy(true_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
    mapped_labels = map_cluster_labels(true_labels, cluster_labels)
    return np.mean(true_labels == mapped_labels)


# ======================
# 数据处理模块
# ======================
def preprocess_data(data_obs: pd.DataFrame, marginal_prob: pd.DataFrame) -> Dict:

    # 输出特征数量信息
    print(f"==== 数据特征数量信息 ====")
    print(f"data_obs 的特征数量（列数）: {data_obs.shape[1]}")
    print(f"marginal_prob 的特征数量（列数）: {marginal_prob.shape[1]}")
    print(f"==========================")

    common_idx = data_obs.index.intersection(marginal_prob.index)
    data_obs = data_obs.loc[common_idx]
    marginal_prob = marginal_prob.loc[common_idx]
    print(f"共同索引数量: {len(common_idx)}")

    # 标准化O数据
    scaler_obs = StandardScaler()
    data_obs_std = scaler_obs.fit_transform(data_obs)

    # 标准化H数据
    scaler_h = StandardScaler()
    marginal_prob_std = scaler_h.fit_transform(marginal_prob)

    # 非标准化H数据
    marginal_prob_non_std = marginal_prob.values

    # 标准化O+H数据（使用标准化的O和标准化的H）
    combined_data = np.hstack([data_obs_std, marginal_prob_std])
    scaler_combined = StandardScaler()
    combined_std = scaler_combined.fit_transform(combined_data)

    return {
        'O': data_obs_std,
        'H': marginal_prob_non_std,
        'H_std': marginal_prob_std,
        'O+H': combined_std
    }


# ======================
# 解析聚类数参数
# ======================
def parse_num_clusters(num_clusters: Union[int, str], marginal_prob: pd.DataFrame) -> int:
    if isinstance(num_clusters, int):
        return num_clusters
    elif isinstance(num_clusters, str) and num_clusters.strip().lower() == '2**h':
        h_vars = marginal_prob.shape[1]
        return 2 ** h_vars
    else:
        raise ValueError(f"无效的聚类数参数: {num_clusters}。请使用整数或字符串'2**h'")


# ======================
# 主流程逻辑
# ======================
def cluster_main(
        data_obs: pd.DataFrame,
        marginal_prob: pd.DataFrame,
        true_labels: np.ndarray,
        model_es: pd.DataFrame,
        num_clusters: Union[int, str],
        data_method_mapping_func: Callable[[int], Dict[str, List[Tuple[str, ClusterAlgorithm]]]],
        num_runs: int = 5,
        random_seed: int = 42
) -> pd.DataFrame:
    start_time = time.time()

    try:
        parsed_num_clusters = parse_num_clusters(num_clusters, marginal_prob)
    except ValueError as e:
        print(f"错误: {str(e)}")
        return pd.DataFrame()

    print(f"====开始聚类，聚类数: {parsed_num_clusters} (解析自: {num_clusters})，重复实验次数: {num_runs}====")

    # 数据预处理
    processed_data = preprocess_data(data_obs, marginal_prob)

    # 执行选择性聚类和重复实验
    all_run_results = []

    for run in range(num_runs):
        print(f"\n=== 第 {run + 1}/{num_runs} 次实验 ===")

        current_seed = random_seed + run
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        # —— 每次 run 再次声明确定性，防外部代码修改全局状态 ——
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(current_seed)

        # 使用当前种子生成数据方法映射
        data_method_mapping = data_method_mapping_func(current_seed)

        # 修正：为当前运行的联合聚类算法设置所有数据
        for dt, methods in data_method_mapping.items():
            for name, method in methods:
                if isinstance(method, CustomCoClustering):
                    method.set_all_data(processed_data)

        for data_type, methods in data_method_mapping.items():
            # 特殊处理联合聚类情况
            if isinstance(data_type, str) and '&' in data_type:
                parts = data_type.split('&')
                primary_view = parts[0].strip()

                if primary_view not in processed_data:
                    print(f"警告: 联合聚类的主视图 '{primary_view}' 不存在，跳过")
                    continue

                data = processed_data[primary_view]
            else:
                if data_type not in processed_data:
                    print(f"警告: 无效数据类型 '{data_type}'，跳过")
                    continue
                data = processed_data[data_type]

            if data.shape[0] == 0:
                print(f"警告: {data_type} 数据为空，跳过")
                continue

            for method_name, method in methods:
                try:
                    print(f"正在对 {data_type} 应用 {method_name} 聚类...")

                    method_start_time = time.time()

                    # 检查是否为GCN包装器，获取格式化名称
                    display_data_type = data_type
                    display_method_name = method_name
                    if isinstance(method, GCNClusterWrapper):
                        display_data_type, display_method_name = method.get_display_name()

                    labels = method.fit_predict(data, parsed_num_clusters)

                    method_elapsed_time = time.time() - method_start_time
                    # print('data_obs_2:', data_obs)
                    # print('model_es:', model_es)
                    # print('marginal_prob_01_2:', marginal_prob)
                    # print('true_labels_2:', true_labels)
                    print(f"{method_name} 对 {data_type} 聚类完成，耗时: {method_elapsed_time:.4f} 秒")

                    internal = calculate_internal_metrics(data, labels)
                    if len(true_labels) > 0:
                        external = calculate_external_metrics(true_labels, labels)
                    else:
                        external = {}

                    result = {
                        'run': run + 1,
                        '数据类型': display_data_type,  # 使用格式化后的显示名称
                        '聚类方法': display_method_name,  # 使用格式化后的显示名称
                    }

                    for key, value in internal.items():
                        if key in RETAINED_METRICS:
                            abbr = METRIC_ABBREVIATIONS.get(key, key)
                            result[abbr] = value

                    if len(true_labels) > 0:
                        for key, value in external.items():
                            if key in RETAINED_METRICS:
                                abbr = METRIC_ABBREVIATIONS.get(key, key)
                                result[abbr] = value

                    all_run_results.append(result)
                except Exception as e:
                    print(f"错误: {method_name} 对 {data_type} 聚类失败: {str(e)}")

    # 处理结果
    if not all_run_results:
        print("错误: 所有聚类任务均失败，无法生成结果")
        return pd.DataFrame()

    results_df = pd.DataFrame(all_run_results)

    # 计算均值和方差
    metrics = list(calculate_internal_metrics(np.random.rand(10, 2), np.random.randint(0, 2, 10)).keys())
    metrics += list(calculate_external_metrics(np.random.randint(0, 2, 10), np.random.randint(0, 2, 10)).keys())
    metrics = [m for m in metrics if m in RETAINED_METRICS]
    metrics_abbr = [METRIC_ABBREVIATIONS.get(m, m) for m in metrics]

    summary_data = []
    internal_metrics = [m for m in metrics if METRIC_TYPES.get(m, '') == '内部指标']
    external_metrics = [m for m in metrics if METRIC_TYPES.get(m, '') == '外部指标']
    for (data_type, method), group in results_df.groupby(['数据类型', '聚类方法']):
        for metric in internal_metrics:
            abbr = METRIC_ABBREVIATIONS.get(metric, metric)
            if abbr in group.columns:
                mean_val = group[abbr].mean()
                var_val = group[abbr].var()

                summary_data.append({
                    '数据类型': data_type,
                    '聚类方法': method,
                    '指标类型': METRIC_TYPES.get(metric, '未知类型'),
                    '指标': abbr,
                    '均值': mean_val,
                    '方差': var_val
                })

        if len(true_labels) > 0:
            for metric in external_metrics:
                abbr = METRIC_ABBREVIATIONS.get(metric, metric)
                if abbr in group.columns:
                    mean_val = group[abbr].mean()
                    var_val = group[abbr].var()
                    summary_data.append({
                        '数据类型': data_type,
                        '聚类方法': method,
                        '指标类型': METRIC_TYPES.get(metric, '未知类型'),
                        '指标': abbr,
                        '均值': mean_val,
                        '方差': var_val
                    })

    summary_df = pd.DataFrame(summary_data)

    # 按指标和均值排序
    metric_cat = pd.CategoricalDtype(categories=METRIC_ORDER, ordered=True)
    summary_df['指标'] = summary_df['指标'].astype(metric_cat)

    db_mask = summary_df['指标'] == 'DB'
    summary_df_db = summary_df[db_mask].sort_values(by='均值', ascending=False)
    summary_df_other = summary_df[~db_mask].sort_values(by='均值', ascending=False)

    summary_df_sorted = pd.concat([summary_df_other, summary_df_db])
    summary_df_sorted = summary_df_sorted.sort_values(by='指标', kind='mergesort')

    summary_df_sorted = summary_df_sorted.reset_index(drop=True)

    column_order = ['数据类型', '聚类方法', '指标类型', '指标', '均值', '方差']
    summary_df_sorted = summary_df_sorted[column_order]

    end_time = time.time()
    print(f"\n全部聚类完成，总耗时: {end_time - start_time:.2f} 秒")

    return summary_df_sorted


# ======================
# 使用示例
# ======================
def cluster_main1(data_obs, model_es, marginal_prob_01, true_labels, num_cluster_new, num_runs, random_seed):
    # random_seed = 42  # 无需取消注释
    # print('data_obs:', data_obs)
    # print('model_es:', model_es)
    # print('marginal_prob_01:', marginal_prob_01)
    # print('true_labels:', true_labels)

    # 处理真实标签
    if true_labels is not None:
        if isinstance(true_labels, pd.DataFrame) and not true_labels.empty:
            true_labels = true_labels['true_label'].values
        elif isinstance(true_labels, list) and len(true_labels) > 0:
            # 如果已经是列表且非空，保持原样
            true_labels = np.array(true_labels)  # 转为numpy数组
        else:
            true_labels = []  # 其他情况设为空列表
    else:
        true_labels = []

    def extract_second1(x):
        value = x  # 初始化返回值
        if isinstance(x, np.ndarray):
            if len(x) > 1:  # 确保数组有第二个元素
                value = float(x[1])
        elif isinstance(x, str):
            parts = x.split()
            if len(parts) > 1:
                # 处理类似 '0.5]' 的字符串，移除可能的右括号
                value = float(parts[1].rstrip(']'))

        # 检查提取的值是否为0或无效值，若是则使用极小值替代
        if pd.isna(value) or value == 0:
            return 1e-10  # 非常小的默认值，避免后续计算出现除以零的问题
        return value

    marginal_prob = marginal_prob_01.copy()
    for col in marginal_prob.columns:
        marginal_prob[col] = marginal_prob[col].map(extract_second1)



    def data_method_mapping_factory(seed):
        def create_kmeans_cluster():
            return KMeansCluster(num_init=10, random_state=seed)

        def create_sinkhorn_cluster():
            return SinkhornKnoppCluster(epsilon=0.01, random_state=seed)

        def create_gmm_cluster():
            return GMMCluster(num_init=1, random_state=seed)

        def create_spectral_cluster():
            return SpectralCluster(num_neighbors=10, random_state=seed)

        def create_custom_cocluster():
            return CustomCoClustering(view1_key='O', view2_key='H_std', max_iter=100, random_state=seed)

        return {
            'O': [
                ('kmeans', create_kmeans_cluster()),
                ('sinkhorn', create_sinkhorn_cluster()),
                ('spectral', create_spectral_cluster()),
                ('gmm', create_gmm_cluster())
            ],
            'H': [
                ('kmeans', create_kmeans_cluster()),
                ('sinkhorn', create_sinkhorn_cluster()),
                ('spectral', create_spectral_cluster()),
                ('gmm', create_gmm_cluster()),
                ('gcn+kmeans', GCNClusterWrapper(model_es, 'H', create_kmeans_cluster())),
                ('gcn+gmm', GCNClusterWrapper(model_es, 'H', create_gmm_cluster()))
            ],
            'O+H': [
                ('kmeans', create_kmeans_cluster()),
                ('sinkhorn', create_sinkhorn_cluster()),
                ('spectral', create_spectral_cluster()),
                ('gmm', create_gmm_cluster()),
                ('gcn+kmeans', GCNClusterWrapper(model_es, 'O+H', create_kmeans_cluster())),
                ('gcn+gmm', GCNClusterWrapper(model_es, 'O+H', create_gmm_cluster()))
            ],
            'O&H_std': [
                ('co-clustering', create_custom_cocluster())
            ]
        }



    result_df = cluster_main(
        data_obs,
        marginal_prob,
        true_labels,
        model_es,
        num_clusters=num_cluster_new,
        data_method_mapping_func=data_method_mapping_factory,
        num_runs=num_runs,
        random_seed=random_seed
    )

    # # 无需取消注释
    # filename = f'../result/imagenet64/cluster/2_4_1%_c10_n1_{seed}.csv'
    # # 保存结果到 CSV 文件
    # result_df.to_csv(filename, index=False, encoding='utf-8-sig')


    # folder = f'../result/imagenet64/cluster'
    # filename = f'{folder}/2_2.9_1%_c1000_n1_{seed}.csv'
    # # 创建文件夹（如果不存在）
    # os.makedirs(folder, exist_ok=True)
    # result_df.to_csv(filename, index=False, encoding='utf-8-sig')

    return result_df



# num_runs = 1
# # num_cluster_new为整数值或者'2**h'
# num_cluster_new = 1000
# for seed in range(42,52):
#     random_seed = seed
#     data_obs = pd.read_csv(f"../result/imagenet64/imagenet64_data_obs_{seed}_1%.csv", index_col='id')
#     model_es = pd.read_csv(f"../result/imagenet64/imagenet64_ModelEs_renum.csv")
#     marginal_prob = pd.read_csv(f"../result/imagenet64/imagenet64_marginal_prob_{seed}_1%.csv", index_col='id')
#     true_labels_df = pd.read_csv(f"../result/imagenet64/imagenet64_true_labels_{seed}_1%.csv", index_col='id')
#     true_labels = true_labels_df['true_label'].values
#     print('true_labels', true_labels)
#     cluster_main1(data_obs, model_es, marginal_prob, true_labels, num_cluster_new, num_runs, random_seed)


