# 功能描述：保存RF距离，mu，以及绘图
import pandas as pd
import random
import numpy as np
import csv
import matplotlib.pyplot as plt

def save_rf_distances_with_headers(rf_dist_real, rf_dist_real0,
                                   rf_dist_estimated=None, rf_dist_estimated0=None,
                                   rf_dist_estimated_sta=None, rf_dist_estimated0_sta=None,
                                   filename=None):
    """
    将多个RF距离列表保存到带列标题的CSV文件中
    支持保存2到6个RF距离列表，根据提供的参数自动调整

    参数：
    rf_dist_real (list)：演化前后结构的RF距离列表
    rf_dist_real0 (list)：每次演化与原始结构的RF距离列表
    rf_dist_estimated (list, optional)：估计结构与真实结构的RF距离列表
    rf_dist_estimated0 (list, optional)：估计结构与原始结构的RF距离列表
    rf_dist_estimated_sta (list, optional)：静态估计结构与真实结构的RF距离列表
    rf_dist_estimated0_sta (list, optional)：静态估计结构与原始结构的RF距离列表
    filename (str)：保存的文件名
    """
    # 确保必须的参数存在且非空
    assert rf_dist_real is not None and len(rf_dist_real) > 0, "必须提供非空的rf_dist_real参数"
    assert rf_dist_real0 is not None and len(rf_dist_real0) > 0, "必须提供非空的rf_dist_real0参数"

    # 获取迭代次数（所有列表的长度必须一致）
    n_iterations = len(rf_dist_real)

    # 初始化数据字典
    data = {'iteration': list(range(1, n_iterations + 1))}

    # 添加必选的RF距离数据
    data['rf_dist_real'] = rf_dist_real
    data['rf_dist_real0'] = rf_dist_real0

    # 动态构建列标题列表
    headers = ['iteration', 'rf_dist_real', 'rf_dist_real0']

    # 添加可选的RF距离数据
    if rf_dist_estimated is not None:
        assert len(rf_dist_estimated) == n_iterations, "rf_dist_estimated长度与其他列表不一致"
        data['rf_dist_estimated'] = rf_dist_estimated
        headers.append('rf_dist_estimated')

        if rf_dist_estimated0 is not None:
            assert len(rf_dist_estimated0) == n_iterations, "rf_dist_estimated0长度与其他列表不一致"
            data['rf_dist_estimated0'] = rf_dist_estimated0
            headers.append('rf_dist_estimated0')

    # 添加静态RF距离数据
    if rf_dist_estimated_sta is not None:
        assert len(rf_dist_estimated_sta) == n_iterations, "rf_dist_estimated_sta长度与其他列表不一致"
        data['rf_dist_estimated_sta'] = rf_dist_estimated_sta
        headers.append('rf_dist_estimated_sta')

        if rf_dist_estimated0_sta is not None:
            assert len(rf_dist_estimated0_sta) == n_iterations, "rf_dist_estimated0_sta长度与其他列表不一致"
            data['rf_dist_estimated0_sta'] = rf_dist_estimated0_sta
            headers.append('rf_dist_estimated0_sta')

    # 确保所有列都存在于DataFrame中
    df = pd.DataFrame(data)

    # 验证所有headers都在DataFrame的列中
    missing_columns = [col for col in headers if col not in df.columns]
    if missing_columns:
        print(f"警告: 以下列不存在于数据中: {missing_columns}")
        headers = [col for col in headers if col in df.columns]

    # 保存数据
    df.to_csv(filename, index=False, columns=headers)

def save_history_mu_with_headers(history, continuous_nodes, filename):
    """
    将history中所有连续节点的mu0和mu1保存到带列标题的CSV文件中
    列标题格式：每个节点对应两列，如'Node1_mu0', 'Node1_mu1', 'Node2_mu0', 'Node2_mu1', ...

    参数：
    history (dict)：包含每个节点mu0和mu1的历史记录
    continuous_nodes (list)：连续节点名称列表（需与history中的节点顺序一致）
    filename (str)：保存的文件名
    """
    # 检查每个节点的mu0和mu1长度是否一致
    n_iterations = len(history[continuous_nodes[0]]['mu0'])
    for node in continuous_nodes:
        assert len(history[node]['mu0']) == n_iterations, f"节点 {node} 的mu0长度不一致"
        assert len(history[node]['mu1']) == n_iterations, f"节点 {node} 的mu1长度不一致"

    # 生成列标题（每个节点对应mu0和mu1两列）
    headers = []
    for node in continuous_nodes:
        headers.append(f"{node}_mu0")
        headers.append(f"{node}_mu1")

    # 按迭代步骤收集数据
    all_data = []
    for i in range(n_iterations):
        row = []
        for node in continuous_nodes:
            row.append(history[node]['mu0'][i])  # mu0
            row.append(history[node]['mu1'][i])  # mu1
        all_data.append(row)

    # 保存为带表头的CSV文件
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)  # 写入表头
        writer.writerows(all_data)  # 写入数据行


# --------------------------- 添加保存和绘图函数 ---------------------------
def save_and_plot_rf_dist(rf_dist1, rf_dist2, filename_prefix="rf_dist"):
    """
    保存RF距离数据并绘制折线图

    参数:
    rf_dist1 (list): 第一个RF距离列表（如真实结构变化的距离）
    rf_dist2 (list): 第二个RF距离列表（如估计结构与真实结构的距离）
    filename_prefix (str): 文件名前缀，默认"rf_dist"
    """
    # 保存数据
    np.savez(f"{filename_prefix}_data.npz", rf_dist1=np.array(rf_dist1), rf_dist2=np.array(rf_dist2))

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(rf_dist1, label="演化前后的RF距离", marker='o', linestyle='-', linewidth=1.5, markersize=2)
    plt.plot(rf_dist2, label="估计前后的RF距离", marker='s', linestyle='-', linewidth=1.5, markersize=2)

    plt.xlabel("迭代次数")
    plt.ylabel("RF距离")
    plt.title("两次RF距离随迭代次数变化趋势")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()  # 调整布局防止标签被截断

    # 保存图片
    plt.savefig(f"{filename_prefix}_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"数据和图片已保存，前缀为：{filename_prefix}")

