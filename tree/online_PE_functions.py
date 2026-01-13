import math
import networkx as nx
from itertools import combinations
from collections import Counter
import numpy as np
import pandas as pd
import gc
import re
import igraph as ig
import warnings
from scipy.optimize import lsq_linear
from numpy.linalg import cond
from scipy.stats import trim_mean


# def stand(x):
#     """
#     对向量 x 进行标准化处理，返回 x / ||x|| 。
#     可根据实际需要进行调整。
#     """
#     norm = np.linalg.norm(x)
#     if norm == 0:
#         return x
#     return x / norm


def make_eigs_real_2x2(A: np.ndarray) -> np.ndarray:
    """
    如果 2x2 矩阵 A 的特征值是复数（判别式 < 0），
    则仅修改对角元使其判别式变为 0，且在 Frobenius 范数下扰动最小；
    如果原本特征值已是实数（判别式 >= 0），则直接返回原矩阵。

    参数
    ----------
    A : np.ndarray
        形状为 (2,2) 的实矩阵

    返回
    ----------
    A_new : np.ndarray
        处理后得到的矩阵 (2x2)
    """
    # 1) 读取矩阵元素
    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]

    # 判别式: (a-d)^2 + 4*b*c
    x = a - d
    Delta = x ** 2 + 4 * b * c
    # print("Delta: ",Delta)
    # 若特征值本来就是实的（含相等实根情形），无需改动
    if Delta > 0:
        return A

    # if Delta <0:
    #     print("特征值为复数")

    # 此时 Δ < 0，说明特征值为复数，需要做对角元的微调

    # 2) 计算两个可能的 x' (令判别式 = 0)
    #    x'^2 = -4*b*c, 故 x' = ±2*sqrt(-b*c)
    #    但要选一个与原 x 更接近的，以最小化扰动
    bc_neg = -b * c  # 因为 b*c < 0, 这里 -b*c > 0
    if bc_neg <= 0:
        # 理论上不会进这里，因为 Δ<0 => b*c<0 => bc_neg>0
        # 但做个容错，还是直接返回 A
        return A

    sqrt_bc = np.sqrt(bc_neg)
    x1 = 2 * sqrt_bc
    x2 = -2 * sqrt_bc

    # 选取与 x 距离更近的 x'
    dist1 = abs(x1 - x)
    dist2 = abs(x2 - x)
    x_star = x1 if dist1 <= dist2 else x2

    # 3) 计算 K = x' - x，并做"对称"调整: δa = K/2, δd = -K/2
    K = x_star - x
    delta_a = 0.5 * K
    delta_d = -0.5 * K

    # 构造新的矩阵
    A_new = A.copy()
    ratio = 1.01  # 进一步放大对角线
    A_new[0, 0] = ratio * (a + delta_a)  # a'
    A_new[1, 1] = ratio * (d + delta_d)  # d'

    return A_new


def stand(x):
    """
    对特征向量 x 进行标准化处理。对应连续变量的特征向量矩阵的第二行代表条件方差，需要是正数

    如果 x 是复数型：
      - 若其实部全部小于 1e-5，则：
          若虚部也全小于 1e-5，则返回全零向量；
          否则返回虚部；
      - 否则返回实部，同时发出警告。
    然后：
      - 若 x 中所有元素均为负，则 x 取反；
      - 若 x 的最后一个元素为负，则 x 取反。

    输入:
      x: numpy 数组（一维向量）

    输出:
      x: 经过标准化后的 numpy 数组（实数）
    """
    # 如果 x 是复数型
    # print("x[0]:", x[0], "np.imag(x[0]):", np.imag(x[0]))
    # print("x[1]:", x[1], "np.imag(x[1]):", np.imag(x[1]))
    if np.imag(x[0]) or np.imag(x[1]):
        # 检查实部是否全小于 1e-5
        if np.all(np.real(x) < 1e-5):
            # 若虚部也全小于 1e-5，则设为零
            if np.all(np.imag(x) < 1e-5):
                x = np.zeros_like(x)
            else:
                x = np.imag(x)
        else:
            x = np.real(x)
        # 强制取实部，并发出警告
        x = np.real(x)
        print("Got complex eigenvector!")
        # warnings.warn("Got complex eigenvector!")
    # 如果所有元素均为负，则取反
    if np.sum(x >= 0) == 0:
        x = -x
    # 如果最后一个元素为负，则取反
    if x[-1] < 0:
        x = -x
    return x


# 参数估计
# 输入: i, Aijk, mu_k, node_type, mu_i, mu_i2, root
# ------i: 可观测节点名
# ------Aijk: 需要分解的矩阵
# ------mu_k: \mu_k, X_k的均值
# ------node_type: 节点i的节点类型, "discrete" 或 "continuous"
# ------mu_i: \mu_i, X_i的均值
# ------mu_i2: \mu^2_i, X^2_i的均值
# ------lambda1: \mu_{k|h=0}
# ------lambda2: \mu_{k|h=1}
# --注：若lambda1,2 未指定，则默认假设 lambda1 > lambda2, 即\mu_{k|h=0} > \mu_{k|h=1}
# ------root: ijk是否为"根节点"的三叉变量
# 输出：Aijk的特征向量矩阵(Γ_{i|h})

def para_esti(i, Aijk, mu_k, node_type="discrete", mu_i=None, mu_i2=None,
              lambda1=None, lambda2=None, root=False):
    """
    参数估计函数，对应于 R 中的 para_esti 函数。

    输入:
      i: 可观测节点名 (字符串)
      Aijk: 待分解矩阵 (numpy 数组)
      mu_k: X_k 的均值 (标量)
      node_type: 节点 i 的类型，取值 "discrete" 或 "continuous"
      mu_i: X_i 的均值 (仅当 node_type=="continuous" 时必须提供)
      mu_i2: X_i^2 的均值 (仅当 node_type=="continuous" 时必须提供)
      lambda1, lambda2: 如果指定，分别表示 μ_{k|h=0} 与 μ_{k|h=1}；否则默认按特征值大小排序
      root: 布尔值，若为 True，表示返回结果中包含“根节点”的概率信息

    输出:
      若 root 为 False，则返回 xi，即 Aijk 的特征向量矩阵 (Γ_{i|h})；
      若 root 为 True，则返回一个字典，包含键 "eigenvector" (xi) 和 "prob_of_root" (P_h[1], 裁剪到 [0,1])
    """
    # 检查 node_type 是否合法
    if node_type not in ["discrete", "continuous"]:
        raise ValueError("'node_type' can only take values 'discrete' or 'continuous'!")
    if node_type == "continuous" and (mu_i is None or mu_i2 is None):
        raise ValueError("'mu_i' and 'mu_i2' cannot be None when 'node_type' is 'continuous'!")

    # 重新处理Aijk避免其特征值为复数，相当于理想状态的延展，理想状态下特征值都是实数会返回原始Aijk
    Aijk = make_eigs_real_2x2(Aijk)

    # 计算 Aijk 的特征值和特征向量
    eigvals, eigvecs = np.linalg.eig(Aijk)

    if i == "8":
        print("变量8 Aijk: ", Aijk)
        print("变量8 eigvals: ", eigvals)
        print("变量8 eigvecs: ", eigvecs)
    # 若特征值包含复数部分，则取实部
    if np.iscomplexobj(eigvals):
        eigvals = np.real(eigvals)
    # 如果两个特征值相等，则进行微调
    tuningnumber = 1e-2
    if np.isclose(eigvals[0], eigvals[1]):
        eigvals[0] += tuningnumber
        eigvals[1] -= tuningnumber
    # 将特征向量存入 xi，取前两列
    xi = eigvecs
    # 对 xi 的第一列和第二列分别调用 stand 函数
    xi1 = stand(xi[:, 0])
    xi2 = stand(xi[:, 1])
    # 组合为 n x 2 矩阵
    xi = np.column_stack((xi1, xi2))

    # 根据是否指定 lambda1, lambda2 处理特征值顺序
    if lambda1 is None or lambda2 is None:
        # 按降序排列特征值及对应特征向量
        index = np.argsort(-eigvals)
        lambda_vals = eigvals[index]
        xi = xi[:, index]
    else:  # 这个else是一个校准，根据之前估计出的条件概率值lambda1, lambda2，对齐新估计出的特征值，以保证隐变量状态的一致性
        candidate1 = np.array([lambda1, lambda2])
        candidate2 = np.array([lambda2, lambda1])
        loss1 = np.sum(np.abs(eigvals - candidate1))
        loss2 = np.sum(np.abs(eigvals - candidate2))
        if loss1 <= loss2:
            lambda_vals = eigvals
            loss = loss1
        else:
            lambda_vals = eigvals[[1, 0]]
            xi = xi[:, [1, 0]]
            loss = loss2

    # 针对离散与连续节点分别处理
    if node_type == "discrete":
        temp_val = np.min(xi)
        if temp_val <= 0:
            warnings.warn("The conditional probability must be positive!")
        # 将 xi 中所有负值置 0
        xi = np.where(xi < 0, 0, xi)
        # 构造线性系统：A x = b, 其中 A = [[1, 1], [lambda_vals[0], lambda_vals[1]]]，b = [1, mu_k]
        p_h = (mu_k - lambda_vals[0]) / (lambda_vals[1] - lambda_vals[0])
        # 截断到 [0.01,0.99]
        p_h = np.clip(p_h, 0.01, 0.99)
        P_h = [1 - p_h, p_h]

        # 分别归一化 xi 的两列
        xi[:, 0] = xi[:, 0] / np.sum(xi[:, 0])
        xi[:, 1] = xi[:, 1] / np.sum(xi[:, 1])
    else:
        # 对于连续节点
        # EX_w=(1-p_h)*mu_w|0+p_h*mu_w|1
        p_h = (mu_k - lambda_vals[0]) / (lambda_vals[1] - lambda_vals[0])
        # 截断到 [0.01,0.99]
        p_h = np.clip(p_h, 0.01, 0.99)
        P_h = [1 - p_h, p_h]
        # 计算 A_temp = xi @ diag(P_h)
        A_temp = xi @ np.diag(P_h)
        # 如果 A_temp 的秩小于2，则加上一个小的对角矩阵修正
        if np.linalg.matrix_rank(A_temp) < 2:
            d = min(np.min(np.abs(A_temp)), 0.1)
            A_temp = A_temp + np.diag([d, d])
        b_vec2 = np.array([mu_i, mu_i2])
        # C = np.linalg.solve(A_temp, b_vec2)

        beta2 = np.linalg.solve(A_temp, b_vec2)
        lb = np.array([1e-3, 1e-3])  # 如果没有下界，可以用 -inf.这里需要下界大于0，因为当前xi的第1行代表条件方差都已经调整为正数了
        # print("beta2: ",beta2)
        # A_temp有可能条件数非常大，采用类GAGA正则化方法。这个正则化的方式极大的会影响条件方差的估计，需要仔细考察
        lam = 1e-3  # lam越大两个均值越趋于相同，lam可以用来控制两个均值的分离程度，另外，关于beta2的平方不如绝对值的效果好
        # 相当于理想状态的延展，只要lam充分小，在理想状态下几乎不影响对应的估计，几乎就应该是理想状态取值
        res = lsq_linear(A_temp.T @ A_temp + np.diag(lam / abs(beta2)), A_temp.T @ b_vec2, bounds=(lb, np.inf))
        C = res.x  # 形状 (2,)

        xi = xi @ np.diag(C)

        # 对条件二阶矩部分进行重新的尺度化估计，相当于理想状态的延展，在理想状态下几乎不影响对应的估计，在非理想状态下，可以对条件方差的估计处理到和样本二阶矩一个尺度
        xi[1, 0] = xi[1, 0] * mu_i2 / ((1 - p_h) * xi[1, 0] + p_h * xi[1, 1])
        xi[1, 1] = xi[1, 1] * mu_i2 / ((1 - p_h) * xi[1, 0] + p_h * xi[1, 1])

        # #避免条件二阶矩的估计距离样本二阶矩太远，理想状态下应该不需要这个
        # if abs(xi[1,0]-mu_i2)/mu_i2>0.7 :
        #     xi[1,0]=mu_i2
        # if abs(xi[1,1]-mu_i2)/mu_i2>0.7 :
        #     xi[1,1]=mu_i2

    # 根据 root 参数决定返回格式
    if root:
        # 对 P_h 截断到 [0.01,0.99]
        P_h = np.clip(P_h, 0.01, 0.99)
        result = {"eigenvector": xi, "prob_of_root": P_h[1]}
        return result
    else:
        return xi


# 得到估计矩阵Aijk以及估计需要的其他参数, 如:mu_k, mu_i, mu^(2)_i
# 输入: ijk; data
# 输出: Aijk; mu_k; mu_i, mu^(2)_i
# 注: 若i为离散变量，则result[[3]] = NULL
def GotA(EXYZ11_online,EXYZ12_online,EXYZ21_online,EXYZ22_online,EXY11_online, EXY12_online,EXY21_online,EXY22_online, EX2_online, EX_online,
         EXYZ_update, EXY_update, EX_update,ijk, data, tol=0.0001):
    """
    得到估计矩阵 Aijk 以及估计需要的其他参数, 如 mu_k, mu_i, mu_i2.

    输入:
      ijk: 列表，包含三个元素，对应变量 i, j, k 的列索引（0-indexed）
      data: 真实数据，二维 NumPy 数组，行表示样本，列表示变量
      tol: 判断变量是否连续的阈值（默认 1e-4）

    输出:
      result: 字典，包含以下键：
          "Aijk": 估计矩阵 Aijk (2x2 numpy 数组)
          "mu_k": mu_k，即 X_k 的均值 (标量)
          "mu_i": 当变量 i 为连续时，字典 {"mu_i": 均值, "mu_i2": 均值平方}；
                  当 i 为离散时，设为 None.
    """
    # 从 ijk 中提取 i, j, k（i,j,k是0-indexed的索引，这里使用的数据data行与列都是0-based的索引）
    i = int(ijk[0]) - 1
    j = int(ijk[1]) - 1
    k = int(ijk[2]) - 1

    # 初始化结果字典，并初始化 mu_i 用于连续变量
    result = {"Aijk": None, "mu_k": None, "mu_i": None}
    mu_i_dict = {"mu_i": None, "mu_i2": None}

    # 注意：R中 data[1, i] 表示第一行第 i 列（1-indexed），
    # 这里我们用 data[0, i] 表示第一行第 i 列（0-indexed）
    # 判断变量 i, j 是否为离散型：如果第一行与其整数部分接近，则认为离散
    i_discrete = (abs(data[0, i] - int(data[0, i])) < tol)
    j_discrete = (abs(data[0, j] - int(data[0, j])) < tol)

    lam = 0.01
    # lam = 0

    # 根据 i, j 的类型分别计算：
    if i_discrete and j_discrete:
        # i, j 均为离散型
        # 对于离散变量，取 1 - X 作为“补集”
        ai = 1 - data[:, i]
        aj = 1 - data[:, j]
        ak = data[:, k]
        # 同时保留原数据列作为 ai2, aj2
        ai2 = data[:, i]
        aj2 = data[:, j]
        # 对离散变量，result["mu_i"] 保持为 None
    elif i_discrete and (not j_discrete):
        # i 离散, j 连续
        ai = 1 - data[:, i]
        aj = data[:, j]
        ak = data[:, k]
        ai2 = data[:, i]
        aj2 = aj ** 2
    elif (not i_discrete) and j_discrete:
        # i 连续, j 离散
        ai = data[:, i]
        aj = 1 - data[:, j]
        ak = data[:, k]
        ai2 = ai ** 2
        aj2 = data[:, j]
        # 保存 mu_i 与 mu_i2
        if EX_update[i]==0:#第一次对i更新
            EX_update[i]=1
            # 缩放旧值（in-place）
            EX_online[i]  *= lam
            EX2_online[i]  *= lam
            # 累加新值（in-place）
            EX_online[i]  += (1-lam) * np.mean(ai)
            EX2_online[i]  += (1-lam) * np.mean(ai2)
        # 无论是否第一次更新，下面这两行都会把当前值写入字典
        mu_i_dict["mu_i"]  = EX_online[i]
        mu_i_dict["mu_i2"] = EX2_online[i]

        # mu_i_dict["mu_i"] = np.mean(ai)             #统计量mean(ai)
        # mu_i_dict["mu_i2"] = np.mean(ai2)           #统计量mean(ai2)
        result["mu_i"] = mu_i_dict
    elif (not i_discrete) and (not j_discrete):
        # i, j 均为连续
        ai = data[:, i]
        aj = data[:, j]
        ak = data[:, k]
        ai2 = ai ** 2
        aj2 = aj ** 2

        # if ijk[0]=="7":
        #     print("np.mean(ai2): ",np.mean(ai2),"EX_update[i]: ",EX_update[i],"EX2_online[i]: ",EX2_online[i])

        if EX_update[i]==0:#第一次对i更新
            EX_update[i]=1
            # 缩放旧值（in-place）
            EX_online[i]  *= lam
            EX2_online[i]  *= lam
            # 累加新值（in-place）
            EX_online[i]  += (1-lam) * np.mean(ai)
            EX2_online[i]  += (1-lam) * np.mean(ai2)
        # 无论是否第一次更新，下面这两行都会把当前值写入字典
        mu_i_dict["mu_i"]  = EX_online[i]
        mu_i_dict["mu_i2"] = EX2_online[i]

        # if ijk[0]=="7":
        #     print("mu_i_dict[mu_i2]: ",mu_i_dict["mu_i2"])

        

        # mu_i_dict["mu_i"] = np.mean(ai)               #统计量mean(ai)
        # mu_i_dict["mu_i2"] = np.mean(ai2)             #统计量mean(ai2)
        result["mu_i"] = mu_i_dict
    else:
        raise ValueError("Unexpected variable types!")

    # 计算 mu_k 为 ak 的均值
    ak = data[:, k]
    ak2= ak ** 2
    if EX_update[k]==0:#第一次对k更新
            EX_update[k]=1
            # 缩放旧值（in-place）
            EX_online[k]  *= lam
            EX2_online[k]  *= lam
            # 累加新值（in-place）
            EX_online[k]  += (1-lam) * np.mean(ak)
            EX2_online[k]  += (1-lam) * np.mean(ak2)        #这里平方也需要更新一下，因为下一次的时候有可能i,k互相置换，这样的话，k已经更新，再后面的使用中也会用到2次矩
    mu_k_val = EX_online[k]

    # if ijk[2]=="7":
    #         print("np.mean(ak2): ",np.mean(ak2))

    # mu_k_val = np.mean(ak)                            #统计量mean(ak)


    if EXYZ_update[i,j,k]==0:
        EXYZ_update[i,j,k]=1
        EXYZ11_online[i,j,k]  *= lam
        EXYZ11_online[i,j,k]  += (1-lam) * np.mean(ai * aj * ak)
        EXYZ12_online[i,j,k]  *= lam
        EXYZ12_online[i,j,k]  += (1-lam) * np.mean(ai * aj2 * ak)
        EXYZ21_online[i,j,k]  *= lam
        EXYZ21_online[i,j,k]  += (1-lam) * np.mean(ai2 * aj * ak)
        EXYZ22_online[i,j,k]  *= lam
        EXYZ22_online[i,j,k]  += (1-lam) * np.mean(ai2 * aj2 * ak)

    Eijk_11 = EXYZ11_online[i,j,k]
    Eijk_12 = EXYZ12_online[i,j,k]
    Eijk_21 = EXYZ21_online[i,j,k]
    Eijk_22 = EXYZ22_online[i,j,k]

    # # 计算 Eijk 各元素,                                     #统计量Eijk
    # Eijk_11 = np.mean(ai * aj * ak)
    # Eijk_12 = np.mean(ai * aj2 * ak)
    # Eijk_21 = np.mean(ai2 * aj * ak)
    # Eijk_22 = np.mean(ai2 * aj2 * ak)
    # 构造 Eijk 矩阵，注意 R 中是按列填充：
    # R: matrix(c(Eijk_11, Eijk_21, Eijk_12, Eijk_22), ncol=2, nrow=2)
    # 得到矩阵 [[Eijk_11, Eijk_12],
    #             [Eijk_21, Eijk_22]]
    Eijk = np.array([[Eijk_11, Eijk_12],
                     [Eijk_21, Eijk_22]])


    if EXY_update[i,j]==0:
        EXY_update[i,j]=1
        EXY11_online[i,j]  *= lam
        EXY11_online[i,j]  += (1-lam) * np.mean(ai * aj)
        EXY12_online[i,j]  *= lam
        EXY12_online[i,j]  += (1-lam) * np.mean(ai * aj2)
        EXY21_online[i,j]  *= lam
        EXY21_online[i,j]  += (1-lam) * np.mean(ai2 * aj)
        EXY22_online[i,j]  *= lam
        EXY22_online[i,j]  += (1-lam) * np.mean(ai2 * aj2)

    Eij_11 = EXY11_online[i,j]
    Eij_12 = EXY12_online[i,j]
    Eij_21 = EXY21_online[i,j]
    Eij_22 = EXY22_online[i,j]
    
    # # 计算 Eij 各元素                                       #统计量Eij
    # Eij_11 = np.mean(ai * aj)
    # Eij_12 = np.mean(ai * aj2)
    # Eij_21 = np.mean(ai2 * aj)
    # Eij_22 = np.mean(ai2 * aj2)

    Eij = np.array([[Eij_11, Eij_12],
                    [Eij_21, Eij_22]])

    # 检查 Eij 的秩，如果小于2则加上一个对角修正
    corr = 1e-3
    if np.linalg.matrix_rank(Eij) < 2:
        correction = min(np.min(np.abs(Eij)), corr)
        Eij = Eij + np.diag([correction, correction])

    # 计算 Aijk = Eijk * Eij_inv
    # Aijk = np.linalg.solve(Eij.T, Eijk.T).T

    # # 假设 Eij, Eijk 已经算好
    eps = 1e-5
    n = Eij.shape[0]  # 这里 n=2
    # 构造正则化后的矩阵,避免Eij求逆数值不稳定
    Eij_reg = Eij + eps * np.eye(n)
    Aijk = np.linalg.solve(Eij_reg.T, Eijk.T).T

    # EijkT=Eijk.T
    # EijT=Eij.T
    # EijkT_0=EijkT[:,0]
    # EijkT_1=EijkT[:,1]
    # beta0 = np.linalg.solve(EijT, EijkT_0)
    # beta1 = np.linalg.solve(EijT, EijkT_1)
    # lam=0.001
    # #Eij有可能条件数非常大，采用类GAGA正则化方法。
    # Aijk0T= np.linalg.solve(Eij@EijT+np.diag(lam/ abs(beta0)), Eij@EijkT_0)
    # Aijk1T= np.linalg.solve(Eij@EijT+np.diag(lam/ abs(beta1)), Eij@EijkT_1)
    # Aijk = np.column_stack((Aijk0T, Aijk1T)).T

    # 组装结果
    result["Aijk"] = Aijk
    result["mu_k"] = mu_k_val

    # if ijk[0]=='7' and ijk[1] =='8' and ijk[2]=='10':
    #     print("ijk[0]==7 and ijk[1] ==8 and ijk[2]==10: ",result)
    #     print("EX_update[i]: ",EX_update[i],"EX_update[k]: ",EX_update[k],"EXYZ_update[i,j,k]",EXYZ_update[i,j,k])

    return result


def get_cond_lat(h1, l1, h2, l2, model, para, prob_marg, g, tree):
    """
    获取条件概率 P(h1 = l1 | h2 = l2)。

    输入:
      h1: str, 隐节点名称（例如 "H1"）
      l1: int, h1 的取值（0 或 1）
      h2: str, 隐节点名称（例如 "H2"）
      l2: int, h2 的取值（0 或 1）
      model: pandas DataFrame，模型信息，要求第一列为节点名称
      para: dict，真实模型参数，键为节点名称，对应的值为 numpy 数组（例如形状为 (2, p)）
      prob_marg: dict，边际概率，键为节点名称，对应值为该节点取 1 的概率（例如数组或数值）
      g: igraph.Graph 对象，图对象，每个顶点属性 "name" 为节点名称
      tree: dict，树结构，至少包含键 "obs_nodes", "lat_nodes", "pa", "child", "des_nodes", "anc_nodes"

    输出:
      p_l1l2: 条件概率 P(h1 = l1 | h2 = l2)（数值或数组）
    """
    # 检查 l1, l2 是否为 0 或 1
    if l1 not in (0, 1) or l2 not in (0, 1):
        raise ValueError("Nodes 'h1' and 'h2' can only take values from {0,1}!")

    obs_nodes = tree["obs_nodes"]
    lat_nodes = tree["lat_nodes"]
    pa = tree["pa"]
    ch = tree["child"]
    de = tree["des_nodes"]
    an = tree["anc_nodes"]
    # 选取根节点：在 lat_nodes 中，父节点为空的那个
    root = [node for node in lat_nodes if pa.get(node, "") == ""][0]

    # 设置 model 的索引为第一列，para 的键也转换为字符串
    model = model.copy()
    model.index = model.iloc[:, 0].astype(str)
    para = {str(k): v for k, v in para.items()}

    # 计算从 h2 到 h1 的最短路径（输出为顶点索引列表），然后转换为名称列表
    paths = g.get_shortest_paths(h2, to=h1, output="vpath")
    if not paths or len(paths[0]) == 0:
        raise ValueError("No path found from h2 to h1!")
    path_names = [g.vs[idx]["name"] for idx in paths[0]]
    # h_m 为路径中倒数第二个节点
    if len(path_names) < 2:
        raise ValueError("Path from h2 to h1 is too short!")
    h_m = path_names[-2]

    # 分支1：若 h_m 等于 pa[h1]
    if h_m == pa.get(h1, None):
        if l1 == 1:
            # 对应 R 中 para[[h1]][1,] 与 para[[h1]][2,]，注意 Python下下标从0开始
            p_l10 = para[h1][0, :]
            p_l11 = para[h1][1, :]
        else:
            p_l10 = 1 - para[h1][0, :]
            p_l11 = 1 - para[h1][1, :]
    else:
        # 分支2：若 h_m 不等于 pa[h1]
        if l1 == 1:
            p_h1l1 = prob_marg[h1]  # P(h1=1)
            p_1l1 = para[h_m][1, :]  # P(hm=1 | h1=1)
            p_0l1 = 1 - p_1l1  # P(hm=0 | h1=1)
        else:
            p_h1l1 = 1 - prob_marg[h1]  # P(h1=0)
            p_1l1 = para[h_m][0, :]  # P(hm=1 | h1=0)
            p_0l1 = 1 - p_1l1  # P(hm=0 | h1=0)
        p_hm1 = prob_marg[h_m]  # P(hm = 1)
        p_hm0 = 1 - p_hm1  # P(hm = 0)
        # 计算条件概率：
        p_l10 = p_0l1 * p_h1l1 / p_hm0
        p_l11 = p_1l1 * p_h1l1 / p_hm1

    # 根据路径长度判断 h1 与 h2 之间是否有其它节点
    if len(path_names) >= 3:
        # 递归调用 get_cond_lat：以 h_m 作为第一个隐节点，取其取值1；h2, l2 不变
        p_1l2 = get_cond_lat(h_m, 1, h2, l2, model, para, prob_marg, g, tree)
        p_0l2 = 1 - p_1l2
    else:
        # 若 h1 与 h2 之间没有其它节点，则：
        p_1l2 = 1 if l2 == 1 else 0
        p_0l2 = 1 if l2 == 0 else 0

    # 最终条件概率
    p_l1l2 = p_l10 * p_0l2 + p_l11 * p_1l2

    # 清理变量（Python 一般自动内存管理，此处可调用 gc）
    gc.collect()
    return p_l1l2


def get_cond_obs(i, a, h, l, model, para, prob_marg, g, tree):
    """
    获取条件期望 E(X_i^a | h=l)（同时用于计算条件概率），
    其中 X_i 的条件取值依赖于 i 的父节点 h_m（即 h 与 i 之间的路径上的倒数第二个节点）。

    输入:
      i: str, 可观测节点名称
      a: int, 阶数；对于连续节点取 1 表示条件均值，取 2 表示条件二阶矩；对于离散节点，取 0 或 1
      h: str, 隐节点名称
      l: int, 隐节点取值（0 或 1）
      model: pandas DataFrame，模型信息，要求第一列为节点名称，第二列为节点类型
      para: dict, 真实模型参数，键为节点名称，对应的值为 numpy 数组（离散: shape (2, p); 连续: shape (4, p)）
      prob_marg: dict, 隐节点取1的概率，键为隐节点名称
      g: igraph.Graph 对象，图，其中每个顶点属性 "name" 为节点名称
      tree: dict, 树结构，至少包含键 "obs_nodes", "lat_nodes", "pa", "child", "des_nodes", "anc_nodes"

    输出:
      mu_il: 条件期望 E(X_i^a | h=l)，计算公式：
             mu_il = p_ia0 * p_{h_m=0|h=l} + p_ia1 * p_{h_m=1|h=l}
             其中 p_ia0、p_ia1 分别为：
               - 离散节点: 如果 a==0，则 p_ia0 = 1 - para[i][0, :], p_ia1 = 1 - para[i][1, :]
                              如果 a==1，则 p_ia0 = para[i][0, :],     p_ia1 = para[i][1, :]
               - 连续节点: 如果 a==1，则 p_ia0 = para[i][0, :], p_ia1 = para[i][1, :]
                              如果 a==2，则 p_ia0 = para[i][2, :]**2 + para[i][0, :]**2,
                                         p_ia1 = para[i][3, :]**2 + para[i][1, :]**2
    """
    # 从 tree 中提取信息
    obs_nodes = tree["obs_nodes"]
    lat_nodes = tree["lat_nodes"]
    pa = tree["pa"]
    ch = tree["child"]
    de = tree["des_nodes"]
    an = tree["anc_nodes"]

    # 选取根节点：在 lat_nodes 中，找到父节点为空字符串的那个
    root = [node for node in lat_nodes if pa.get(node, "") == ""][0]

    # 设置 model 的索引为第一列（节点名称），确保能用节点名称查找
    model = model.copy()
    model.index = model.iloc[:, 0].astype(str)
    # 确保 para 的键均为字符串
    para = {str(key): value for key, value in para.items()}

    # 检查 i 对应的节点类型是否为可观测
    node_type = model.loc[i, model.columns[1]]
    if node_type not in ["observable_discrete", "observable_continuous"]:
        raise ValueError("Node 'i' should be observed!")

    # 计算从 h 到 i 的最短路径（使用 igraph），并取该路径的顶点名称
    # g.get_shortest_paths 返回一个列表，每个元素为顶点索引列表
    paths = g.get_shortest_paths(h, to=i, output="vpath")
    if len(paths) == 0 or len(paths[0]) == 0:
        raise ValueError("No path found from h to i!")
    path_h_to_i = [g.vs[idx]["name"] for idx in paths[0]]

    # h_m 为路径中倒数第二个节点，即 i 的父节点（从 h 到 i的路径应至少有两个节点）
    if len(path_h_to_i) < 2:
        raise ValueError("Path from h to i is too short!")
    h_m = path_h_to_i[-2]

    # 判断 h_m 是否等于 pa[i]
    if h_m == pa.get(i, None):
        # 根据 i 的类型分别计算 p_ia0 和 p_ia1
        if node_type == "observable_discrete":
            if a == 0:
                # R中： 1 - para[[i]][1,]，Python下标从0开始
                p_ia0 = 1 - para[i][0, :]
                p_ia1 = 1 - para[i][1, :]
            elif a == 1:
                p_ia0 = para[i][0, :]
                p_ia1 = para[i][1, :]
            else:
                raise ValueError("'i' is a discrete node, so 'a' should be 0 or 1!")
        else:  # observable_continuous
            if a == 1:
                p_ia0 = para[i][0, :]
                p_ia1 = para[i][1, :]
            elif a == 2:
                # 注意：R中 para[[i]][3,] 表示第三行，对应 Python 下标2；para[[i]][1,] 对应下标0
                p_ia0 = (para[i][2, :]) ** 2 + (para[i][0, :]) ** 2
                p_ia1 = (para[i][3, :]) ** 2 + (para[i][1, :]) ** 2
            else:
                raise ValueError("'i' is a continuous node, so 'a' should be 1 or 2!")

        # 计算从 h 到 i 的路径长度，若长度>=3，表示 h != pa(i)
        if len(path_h_to_i) >= 3:
            # 调用辅助函数 get_cond_lat 获取 p_{h_m = 1 | h = l}
            p_1l = get_cond_lat(h_m, 1, h, l, model, para, prob_marg, g, tree)
            p_0l = 1 - p_1l
        else:
            # 若路径长度小于3，则 h 为 i 的直接父节点
            if l == 0:
                p_0l = 1
                p_1l = 0
            elif l == 1:
                p_0l = 0
                p_1l = 1
            else:
                raise ValueError("l should be 0 or 1!")

        # 计算最终条件期望
        mu_il = p_ia0 * p_0l + p_ia1 * p_1l
    else:
        raise ValueError("Parent node of 'i' should be adjacent to 'i'!")

    return mu_il


def GotE_true(ijk, h=None, a=1, b=1, model=None, para=None, prob_marg=None, g=None, tree=None):
    """
    获取真实模型下 E(X_i^a * X_j^b * X_k)，其中 i, j, k 分别为可观测节点，条件于隐节点 h。

    输入:
      ijk: list，长度至少为 3，表示可观测节点 i, j, k 的标识（例如字符串或数字）
      h: 隐节点（如果为 None，则根据 i 的父节点确定）
      a: 阶数，对 i 的幂次（连续变量时可取 1 或 2；离散变量通常为1）
      b: 阶数，对 j 的幂次（连续变量时可取 1 或 2；离散变量通常为1）
      model: pandas DataFrame，模型信息，第1列为节点名称，第2列为节点类型
      para: dict，真实模型参数，键为节点名称，对应的值为参数矩阵（其中第二维为重复实验数）
      prob_marg: dict，边际概率，键为隐节点名称，值为该隐节点取 1 的概率
      g: 图对象（例如 igraph.Graph），若未提供则由 tree 中的信息构造
      tree: dict，树结构，至少包含键 "obs_nodes", "lat_nodes", "pa", "child", "des_nodes", "anc_nodes"

    输出:
      Eijk: 期望值 E(X_i^a * X_j^b * X_k)（数值）
    """
    # 从 tree 中提取必要信息
    obs_nodes = tree["obs_nodes"]
    lat_nodes = tree["lat_nodes"]
    pa = tree["pa"]
    ch = tree["child"]
    de = tree["des_nodes"]
    an = tree["anc_nodes"]

    # 确定根节点：在潜变量中找出父节点为空的第一个
    root = [node for node in lat_nodes if pa.get(node, "") == ""][0]

    # 将 model 的行索引设为第一列（节点名称），确保查找正确
    model = model.copy()
    model.index = model.iloc[:, 0].astype(str)
    # 确保 para 的键均为字符串
    para = {str(k_): v for k_, v in para.items()}

    num = len(ijk)
    # 如果 h 为空，则取 ijk[0] 对应的父节点（注意：R 中 ijk[1] 对应 Python 中 ijk[0]）
    if h is None:
        h = pa.get(str(ijk[0]), None)
        if h is None:
            raise ValueError("Cannot determine hidden node h from pa!")

    # 取得 P(h=1) 和 P(h=0)
    p_h1 = prob_marg[h]  # 隐节点 h 取1的概率
    p_h0 = 1 - p_h1

    # 处理第一个节点 i
    i = ijk[0]
    # 调用辅助函数 get_cond_obs 得到条件期望 E(X_i^a | h=l)
    mu_ia0 = get_cond_obs(i=i, a=a, h=h, l=0, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
    mu_ia1 = get_cond_obs(i=i, a=a, h=h, l=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
    equ0 = p_h0 * mu_ia0  # P(h=0)*E(X_i^a|h=0)
    equ1 = p_h1 * mu_ia1  # P(h=1)*E(X_i^a|h=1)

    # 如果 ijk 长度>=2，处理第二个节点 j
    if num >= 2:
        j = ijk[1]
        mu_jb0 = get_cond_obs(i=j, a=b, h=h, l=0, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        mu_jb1 = get_cond_obs(i=j, a=b, h=h, l=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        equ0 *= mu_jb0  # 更新：P(h=0)*E(X_i^a X_j^b|h=0)
        equ1 *= mu_jb1  # 更新：P(h=1)*E(X_i^a X_j^b|h=1)

    # 如果 ijk 长度>=3，处理第三个节点 k
    if num >= 3:
        k_node = ijk[2]
        mu_k10 = get_cond_obs(i=k_node, a=1, h=h, l=0, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        mu_k11 = get_cond_obs(i=k_node, a=1, h=h, l=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        equ0 *= mu_k10  # 更新：P(h=0)*E(X_i^a X_j^b X_k|h=0)
        equ1 *= mu_k11  # 更新：P(h=1)*E(X_i^a X_j^b X_k|h=1)

    # 最终期望值为两部分之和
    Eijk = equ0 + equ1
    return Eijk


def GotA_true(ijk, h=None, model=None, para=None, prob_marg=None, g=None, tree=None):
    """
    根据真实模型参数，获取真实的 Aijk, mu_k 与 mu_i。
    其中 Aijk = Eijk @ inv(Eij)；mu_k 与 mu_i, mu_i2 分别为 X_k 的均值与 X_i 的均值和平方均值。

    输入:
      ijk: list，长度应为3，表示可观测节点 i, j, k
      h: 隐节点（如果为 None，则用 tree 中 pa 对 i 取值）
      model: pandas DataFrame，模型信息，第1列为节点名称，第2列为节点类型
      para: 字典，真实模型参数，键为节点名称，对应值为 numpy 数组，形状为 (n, p)，p 为重复实验次数
      prob_marg: 隐节点取值为1的边际概率（例如长度为500的数组）
      g: igraph.Graph 对象
      tree: 字典，树结构，至少包含键 "obs_nodes", "lat_nodes", "str", "pa", "child", "des_nodes", "anc_nodes"

    输出:
      result: 字典，包含以下键：
         "Aijk": 长度为 p 的列表，每个元素为 2x2 numpy 数组
         "mu_k": X_k 的均值 (标量)
         "mu_i": 若节点 i 为连续，则为字典 {"mu_i": ..., "mu_i2": ...}；若 i 为离散，则为 None
    """
    # 从 tree 中提取信息
    obs_nodes = tree["obs_nodes"]
    lat_nodes = tree["lat_nodes"]
    st = tree["str"]
    pa = tree["pa"]
    ch = tree["child"]
    de = tree["des_nodes"]
    an = tree["anc_nodes"]
    # 选取根节点：R 中 root <- lat_nodes[pa[lat_nodes] == ""]
    # 这里假定根节点为在 lat_nodes 中，其对应的 pa 值为空字符串
    root = [node for node in lat_nodes if pa.get(node, "") == ""][0]

    # 设置 model 的索引为第一列，并保证 para 的键与 model 索引一致
    model = model.copy()
    model.index = model.iloc[:, 0].astype(str)
    # 假定 para 的键与 model.index 中的值一致（这里可做必要转换）
    for key in list(para.keys()):
        para[str(key)] = para.pop(key)

    # 取 p 为 para 中第一个元素第二维度的大小
    sample_key = next(iter(para))
    p = para[sample_key].shape[1]

    # 检查 ijk 长度
    if len(ijk) != 3:
        raise ValueError("The 'ijk' should contain 3 observed nodes!")
    # 设定 i, j, k（这里 ijk 的元素保持不变，假定类型可直接作为键使用）
    i = str(ijk[0])
    j = str(ijk[1])
    k = str(ijk[2])

    if h is None:
        # h 取 i 的父节点
        h = pa.get(i, None)
        if h is None:
            raise ValueError("Cannot determine hidden node 'h' from pa!")

    # 初始化结果字典
    result = {"Aijk": None, "mu_k": None, "mu_i": None}
    # 初始化 mu_i（仅对连续变量有效），存储 mu_i 与 mu_i2
    mu_i_dict = {"mu_i": None, "mu_i2": None}

    # 分支处理：依据 model 中节点类型
    # 注意：model.loc[node, model.columns[1]] 表示节点 node 的类型
    type_i = model.loc[i, model.columns[1]]
    type_j = model.loc[j, model.columns[1]]

    # 定义辅助变量 Eijk_ij 与 Eij_ij，均为长度为 p 的数组（后面每个元素对应一个实验组）
    # 分支1: i, j 均为 observable_discrete
    if type_i == "observable_discrete" and type_j == "observable_discrete":
        EXk = GotE_true(k, h=h, a=1, b=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        EXiXk = GotE_true([i, k], h=h, a=1, b=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        EXjXk = GotE_true([j, k], h=h, a=1, b=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        EXiXjXk = GotE_true(ijk, h=h, a=1, b=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        # 计算 Eijk 分量
        Eijk_11 = EXk - EXiXk - EXjXk + EXiXjXk
        Eijk_12 = EXjXk - EXiXjXk
        Eijk_21 = EXiXk - EXiXjXk
        Eijk_22 = EXiXjXk

        EXi = GotE_true(i, h=h, a=1, b=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        EXj = GotE_true(j, h=h, a=1, b=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        EXiXj = GotE_true([i, j], h=h, a=1, b=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        Eij_11 = 1 - EXi - EXj + EXiXj
        Eij_12 = EXj - EXiXj
        Eij_21 = EXi - EXiXj
        Eij_22 = EXiXj
    # 分支2: i 为 observable_discrete, j 为 observable_continuous
    elif type_i == "observable_discrete" and type_j == "observable_continuous":
        EXjXk = GotE_true([j, k], h=h, a=1, b=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        EX2jXk = GotE_true([j, k], h=h, a=2, b=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        EXiXjXk = GotE_true(ijk, h=h, a=1, b=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        EXiX2jXk = GotE_true(ijk, h=h, a=1, b=2, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        Eijk_11 = EXjXk - EXiXjXk
        Eijk_12 = EX2jXk - EXiX2jXk
        Eijk_21 = EXiXjXk
        Eijk_22 = EXiX2jXk

        EXj = GotE_true(j, h=h, a=1, b=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        EX2j = GotE_true(j, h=h, a=2, b=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        EXiXj = GotE_true([i, j], h=h, a=1, b=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        EXiX2j = GotE_true([i, j], h=h, a=1, b=2, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        Eij_11 = EXj - EXiXj
        Eij_12 = EX2j - EXiX2j
        Eij_21 = EXiXj
        Eij_22 = EXiX2j
    # 分支3: i 为 observable_continuous, j 为 observable_discrete
    elif type_i == "observable_continuous" and type_j == "observable_discrete":
        EXiXk = GotE_true([i, k], h=h, a=1, b=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        EX2iXk = GotE_true([i, k], h=h, a=2, b=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        EXiXjXk = GotE_true(ijk, h=h, a=1, b=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        EX2iXjXk = GotE_true(ijk, h=h, a=2, b=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        Eijk_11 = EXiXk - EXiXjXk
        Eijk_12 = EXiXjXk
        Eijk_21 = EX2iXk - EX2iXjXk
        Eijk_22 = EX2iXjXk

        EXi = GotE_true(i, h=h, a=1, b=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        EX2i = GotE_true(i, h=h, a=2, b=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        EXiXj = GotE_true([i, j], h=h, a=1, b=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        EX2iXj = GotE_true([i, j], h=h, a=2, b=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        Eij_11 = EXi - EXiXj
        Eij_12 = EXiXj
        Eij_21 = EX2i - EX2iXj
        Eij_22 = EX2iXj
        mu_i_dict["mu_i"] = GotE_true(i, h=h, a=1, b=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        mu_i_dict["mu_i2"] = GotE_true(i, h=h, a=2, b=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        result["mu_i"] = mu_i_dict
    # 分支4: i, j 均为 observable_continuous
    elif type_i == "observable_continuous" and type_j == "observable_continuous":
        Eijk_11 = GotE_true(ijk, h=h, a=1, b=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        Eijk_12 = GotE_true(ijk, h=h, a=1, b=2, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        Eijk_21 = GotE_true(ijk, h=h, a=2, b=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        Eijk_22 = GotE_true(ijk, h=h, a=2, b=2, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)

        Eij_11 = GotE_true([i, j], h=h, a=1, b=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        Eij_12 = GotE_true([i, j], h=h, a=1, b=2, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        Eij_21 = GotE_true([i, j], h=h, a=2, b=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        Eij_22 = GotE_true([i, j], h=h, a=2, b=2, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        EXi = GotE_true(i, h=h, a=1, b=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        EX2i = GotE_true(i, h=h, a=2, b=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
        mu_i_dict["mu_i"] = EXi
        mu_i_dict["mu_i2"] = EX2i
        result["mu_i"] = mu_i_dict
    else:
        raise ValueError("Node 'i' and 'j' only can be 'observable_continuous' or 'observable_discrete'!")

    # 对 k，均用 a=1, b=1 得到 EXk
    EXk = GotE_true(k, h=h, a=1, b=1, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
    result["mu_k"] = EXk

    # 构造 Eijk 与 Eij 的数组，维度为 (p, 2, 2)
    # R 代码：array(c(Eijk_11,Eijk_21,Eijk_12,Eijk_22), dim=c(p,2,2))
    Eijk = np.empty((p, 2, 2))
    Eijk[:, 0, 0] = Eijk_11
    Eijk[:, 1, 0] = Eijk_21
    Eijk[:, 0, 1] = Eijk_12
    Eijk[:, 1, 1] = Eijk_22

    Eij = np.empty((p, 2, 2))
    Eij[:, 0, 0] = Eij_11
    Eij[:, 1, 0] = Eij_21
    Eij[:, 0, 1] = Eij_12
    Eij[:, 1, 1] = Eij_22

    # 对每个实验组（索引 0~p-1），计算 Aijk = Eijk @ inv(Eij)
    Aijk = [np.linalg.inv(Eij[index, :, :]).dot(Eijk[index, :, :]) for index in range(p)]
    result["Aijk"] = Aijk

    # 清除其它变量（Python 会自动回收内存，这里调用 gc）
    gc.collect()
    return result


# 参数估计(针对某个隐节点的所有可观测子节点)
# 输入: ijk; h; flag; tree; data; tol; model; para; prob_marg; g
# ------ijk: 三叉变量组成的list, 长度 = 可观测子节点个数; 每个元素都是一组三叉变量
# ------h: 隐节点
# ------flag: 0-1列; 每个元素对应一组三叉变量; flag = 0 <=> mu_{k|h=0} > mu_{k|h=1}
# ------tree:
# --注: 真实数据, 实际应用
# ------data: 真实数据, data.frame
# ------tol: 判断变量是否为连续的阈值
# --注: 需要带入的真实参数, 用来检验算法是否出错的
# ------model: 模型, 1-6, data.frame
# ------para: list, 长度为节点数; 真实模型参数, 用来产生数据的500组模型参数
# ------prob_marg: list; 边际概率, 同样是500组
# ------g: igraph包产生的图

def para_esti_obs(EXYZ11_online,EXYZ12_online,EXYZ21_online,EXYZ22_online,EXY11_online, EXY12_online,EXY21_online,EXY22_online, EX2_online, EX_online,
                  EXYZ_update, EXY_update, EX_update,ijk, h=None, flag=None, tree=None, data=None, tol=1e-6,
                  model=None, para=None, prob_marg=None, g=None):
    """
    针对某个隐节点 h 的所有可观测子节点进行参数估计。

    输入:
      ijk: list，每个元素为三叉变量 (i, j, k) 的列表，长度等于可观测子节点个数
      h: 隐节点，字符串（如 "H1"）；若未提供，则根据 ijk[0] 和 tree 的父节点关系推断
      flag: list，与 ijk 长度相同，每个元素为 0 或 1；
            0 表示矩阵分解时让特征根从大到小排列即可： μ_{k|h=0} > μ_{k|h=1}。1 表示分解后特征根的先后顺序需要根据之前的参数估计值进行校准对齐；默认全 0
      # flag=0的位置对应的参数估计，只需要在估计时要求分解后的特征根从大到小排列即可，
      # flag=1的位置对应的参数估计，要求根据之前的参数估计值对最新分解得到的特征根进行校准对齐，
      # 以保证条件概率矩阵中，对应隐父节点状态对于所有的子节点是一致的
      tree: 字典，树结构，必须包含键 "obs_nodes", "lat_nodes", "pa", "str", "child", "des_nodes", "D"
      data: 真实数据（例如 pandas DataFrame或 numpy 数组），当未提供真实模型参数时使用
      tol: 判断变量是否连续的阈值（默认 1e-6）
      model: 模型 DataFrame，当提供真实模型参数时使用
      para: 真实模型参数（字典或列表），用于产生数据的多组参数
      prob_marg: 边际概率参数（与 para 对应），用于真实参数输入时使用
      g: igraph.Graph 对象；若未提供，则根据 tree['pa'] 构造

    输出:
      para_hat: 字典，键为每个可观测子节点 i（从 ijk 中提取），值为该节点的估计参数（可能为列表或单个值）
    """

    if flag is None:
        flag = [0] * len(ijk)

    # 检查 h 是否为隐节点
    if h is not None and not h.startswith("H"):
        raise ValueError("Node 'h' should be latent!")

    # 从 tree 中提取变量
    obs_nodes = tree["obs_nodes"]
    lat_nodes = tree["lat_nodes"]
    pa = tree["pa"]
    st = tree["str"]
    ch = tree["child"]
    de = tree["des_nodes"]
    D = tree.get("D", None)

    # 如果 g 未提供，则从 pa 构造边集并生成无向图
    if g is None:
        # 构造边列表：键为节点名称，值为其父节点；忽略父节点为空的情况
        edges = [(node, parent) for node, parent in pa.items() if parent != ""]
        # 假定使用 python-igraph 构造图（需要提前安装 python-igraph）
        import igraph as ig
        g = ig.Graph.TupleList(edges, directed=False)

    # 如果 D 未提供，则构造一个全 1 的矩阵（对角线 0），大小为 (len(obs_nodes), len(obs_nodes))
    if D is None:
        n_obs = len(obs_nodes)
        D = np.ones((n_obs, n_obs))
        np.fill_diagonal(D, 0)
        # 也可以将 D 转换为 DataFrame（若后续需要按名称索引）
        D = pd.DataFrame(D, index=obs_nodes, columns=obs_nodes)

    # print("ijk:")
    # print(ijk)

    # 检查 ijk 是否为列表
    if not isinstance(ijk, list):
        raise ValueError("The object 'ijk' should be a list!")

    # 若 h 未提供，则根据 ijk[0][0] 从 pa 推断 h
    if h is None:
        i1 = ijk[0][0]
        h = pa.get(i1)  # 不检查 h 是否为 None，与 R 代码保持一致

    # 判断 h 是否为根节点（即 pa[h] == ""）
    # 是否为根节点, 创建列表存储根节点的"边际概率"参数
    is_root = (pa.get(h, "") == "")
    root_para = {}  # 若 h 为根节点，存储每个隐子节点对应的边际概率参数

    # 构造 I：创建一个与 ijk 长度相同的空字符串列表
    I = ["" for _ in range(len(ijk))]
    # flag = 0 的那些 ijk，可以一起处理
    for l in range(len(ijk)):
        I[l] = ijk[l][0]

    # 找出 I 中的连续节点，得到估计需要输入的参数

    # node_type: 字典，键为 I 中的节点，每个值初始化为空字符串
    node_type = {node: "" for node in I}

    # para_input: 字典，每个键为 I 中的节点，对应的值初始化为 None
    # 注释中提到，每个 para_input 对象应包含：Aijk, mu_k, 以及 mu_i (一个长度为2的列表)
    para_input = {node: None for node in I}

    # 根据是否输入真实模型参数决定分支：
    if para is None or model is None or prob_marg is None:  # ------------输入的是数据
        # 输入的是数据，若 data 未提供则报错
        if data is None:
            raise ValueError(
                "Please input complete parameters 'model', 'para' and 'para_marg' or the parameter 'data'!")

        # 对 I 中每个节点进行处理
        for l in range(len(I)):
            i = I[l]
            # print("i: ",i)
            # 判断节点是离散还是连续：若 data.iloc[0][i] 接近其整数部分，则认为离散
            idx = int(i)  # 转换为整数
            if abs(data[0][idx - 1] - int(data[0][idx - 1])) < tol:
                node_type[i] = "discrete"
            else:
                node_type[i] = "continuous"

            # 调用 GotA 函数，处理当前节点对应的 ijk 值，并赋值给 para_input 字典
            para_input[i] = GotA(EXYZ11_online,EXYZ12_online,EXYZ21_online,EXYZ22_online,EXY11_online, EXY12_online,EXY21_online,EXY22_online, EX2_online, EX_online,
                                 EXYZ_update, EXY_update, EX_update,ijk[l], data, tol)
            # print("l: ",l,"ijk[l]: ",ijk[l])
            # print("para_input[i]: ",para_input[i])


    else:  # -------------------------------------------------------------输入的是真实的模型参数
        for l in range(len(I)):
            i = I[l]
            # 这里假设 model 是一个 DataFrame，使用 .loc 根据节点名 i 获取对应行
            if model.loc[i, model.columns[1]] == "observable_continuous":
                node_type[i] = "continuous"
            elif model.loc[i, model.columns[1]] == "observable_discrete":
                node_type[i] = "discrete"

            para_input[i] = GotA_true(ijk[l], h, model, para, prob_marg, g, tree)

    # 释放内存
    gc.collect()

    # 预先初始化 para_hat: 字典，键为 I 中的节点，初始值设为 None
    para_hat = {node: None for node in I}

    # 计算 flag==0 与 flag==1 的布尔列表,
    # flag=0的位置对应的参数估计，只需要在估计时要求特征根从大到小排列即可，
    # flag=1的位置对应的参数估计，要求根据之前的参数估计值对最新分解得到的特征根进行校准对齐，
    # 以保证条件概率矩阵中，对应隐父节点状态对于所有的子节点是一致的
    logi0 = [f == 0 for f in flag]
    logi1 = [f == 1 for f in flag]

    if sum(logi0) == 0:
        # flag 全为 1 时，报错
        raise ValueError("The observed nodes' 'flag' should have the value 0!")
    elif sum(logi1) == 0:
        # flag 全为 0 的情况
        for l in range(len(I)):
            i = I[l]
            ijk_i = ijk[l]
            para_input_i = para_input[i]
            Aijk_i = para_input_i["Aijk"]
            mu_k = para_input_i["mu_k"]
            mu_i = para_input_i["mu_i"]

            # print("mu_i: in sum(logi1) == 0")
            # print("mu_i: ",mu_i)

            if isinstance(Aijk_i, list):
                # 如果 Aijk_i 中有多个元素，要求长度一致
                if max(len(Aijk_i), len(mu_k)) != min(len(Aijk_i), len(mu_k)):
                    raise ValueError("'Aijk_i', 'mu_k' and 'mu_i' have different length!")
                p = len(Aijk_i)
                temp = [None] * p
                for ll in range(p):
                    temp[ll] = para_esti(
                        i,
                        Aijk_i[ll],
                        mu_k[ll],
                        node_type[i],
                        mu_i=mu_i[0][ll],
                        mu_i2=mu_i[1][ll],
                        lambda1=None,
                        lambda2=None,
                        root=is_root
                    )
                para_hat[i] = temp
            else:
                # 如果 Aijk_i 只有一个元素，mu_k 应只有一个值
                if isinstance(mu_k, (list, tuple)):
                    if len(mu_k) != 1:
                        raise ValueError("'Aijk_i' and 'mu_k' should have length 1!")
                para_hat[i] = para_esti(
                    i,
                    Aijk_i,
                    mu_k,
                    node_type[i],
                    mu_i=mu_i["mu_i"] if mu_i is not None else None,  # mu_i为None时，mu_i没法取下标，需要特殊处理一下
                    mu_i2=mu_i["mu_i2"] if mu_i is not None else None,  # mu_i为None时，mu_i没法取下标，需要特殊处理一下
                    lambda1=None,
                    lambda2=None,
                    root=is_root
                )
    else:
        # flag 中既有 0 又有 1 的情况
        # 先处理 flag==0 的节点
        indices_logi0 = [idx for idx, val in enumerate(logi0) if val]

        # print("indices_logi0: ",indices_logi0)
        for l in indices_logi0:
            i = I[l]
            ijk_i = ijk[l]
            para_input_i = para_input[i]
            Aijk_i = para_input_i["Aijk"]
            mu_k = para_input_i["mu_k"]
            mu_i = para_input_i["mu_i"]

            # print("mu_i:")
            # print(mu_i)

            if isinstance(Aijk_i, list):
                if max(len(Aijk_i), len(mu_k)) != min(len(Aijk_i), len(mu_k)):
                    raise ValueError("'Aijk_i', 'mu_k' and 'mu_i' have different length!")
                p = len(Aijk_i)
                temp = [None] * p
                for ll in range(p):
                    temp[ll] = para_esti(
                        i,
                        Aijk_i[ll],
                        mu_k[ll],
                        node_type[i],
                        mu_i=mu_i[0][ll],
                        mu_i2=mu_i[1][ll],
                        lambda1=None,
                        lambda2=None,
                        root=is_root
                    )
                para_hat[i] = temp
            else:
                # print("mu_k: ",mu_k)
                if isinstance(mu_k, (list, tuple)):
                    if len(mu_k) != 1:
                        raise ValueError("'Aijk_i' and 'mu_k' should have length 1!")
                para_hat[i] = para_esti(
                    i,
                    Aijk_i,
                    mu_k,
                    node_type[i],
                    mu_i=mu_i["mu_i"] if mu_i is not None else None,  # mu_i为None时，mu_i没法取下标，需要特殊处理一下
                    mu_i2=mu_i["mu_i2"] if mu_i is not None else None,
                    lambda1=None,
                    lambda2=None,
                    root=is_root
                )

        # 再处理 flag==1 的节点（最后一个 flag=1 的可观测子节点）
        indices_logi1 = [idx for idx, val in enumerate(logi1) if val]
        for l in indices_logi1:
            i = I[l]
            ijk_i = ijk[l]
            # R 代码中：j <- ijk_i[2]; k <- ijk_i[3]
            # Python 中索引从 0 开始：j 对应 ijk_i[1]，k 对应 ijk_i[2]
            j = ijk_i[1]
            k = ijk_i[2]
            para_input_i = para_input[i]
            Aijk_i = para_input_i["Aijk"]
            mu_k = para_input_i["mu_k"]
            mu_i = para_input_i["mu_i"]

            if isinstance(Aijk_i, list):
                if max(len(Aijk_i), len(mu_k)) != min(len(Aijk_i), len(mu_k)):
                    raise ValueError("'Aijk_i', 'mu_k' and 'mu_i' have different length!")
                p = len(Aijk_i)
                temp = [None] * p
                for ll in range(p):
                    # 取得 para_hat[k] 中第 ll 个元素
                    if k not in I:
                        raise ValueError("The node " + str(k) + " should belong to 'I'!")
                    # 假定 para_hat[k][ll] 为二维结构（如列表或 NumPy 数组），第一行为连续型，第二行为离散型
                    mat = para_hat[k][ll]
                    if node_type[k] == "continuous":
                        # 取第一行：Python 中索引 0
                        lambda_val = mat[0]
                    elif node_type[k] == "discrete":
                        lambda_val = mat[1]
                    else:
                        lambda_val = None
                    temp[ll] = para_esti(
                        i,
                        Aijk_i[ll],
                        mu_k[ll],
                        node_type[i],
                        mu_i=mu_i[0][ll],
                        mu_i2=mu_i[1][ll],
                        lambda1=lambda_val[0],
                        lambda2=lambda_val[1],
                        root=is_root
                    )
                para_hat[i] = temp
            else:
                if isinstance(mu_k, (list, tuple)):
                    if len(mu_k) != 1:
                        raise ValueError("'Aijk_i' and 'mu_k' should have length 1!")
                if k not in I:
                    raise ValueError("The node " + str(k) + " should belong to 'I'!")
                else:
                    if node_type[k] == "continuous":
                        # 对于h是否为根节点，para_hat的值的数据结构是不同的。具体参见para_esti()的返回值
                        if is_root:
                            lambda_val = para_hat[k]['eigenvector'][0]
                        else:
                            lambda_val = para_hat[k][0]
                    elif node_type[k] == "discrete":
                        # 对于h是否为根节点，para_hat的值的数据结构是不同的。具体参见para_esti()的返回值
                        if is_root:
                            lambda_val = para_hat[k]['eigenvector'][1]
                        else:
                            lambda_val = para_hat[k][1]
                    else:
                        lambda_val = None
                para_hat[i] = para_esti(
                    i,
                    Aijk_i,
                    mu_k,
                    node_type[i],
                    mu_i=mu_i["mu_i"] if mu_i is not None else None,  # mu_i为None时，mu_i没法取下标，需要特殊处理一下,
                    mu_i2=mu_i["mu_i2"] if mu_i is not None else None,  # mu_i为None时，mu_i没法取下标，需要特殊处理一下,
                    lambda1=lambda_val[0],
                    lambda2=lambda_val[1],
                    root=is_root
                )

    # 清理不需要的变量（Python 中通常不需要手动清理，垃圾回收会自动处理）
    # 最后返回 para_hat
    return para_hat


# 三叉变量的标准化(h的可观测子节点)
# tri_nodes_obs输出结果的标准化
# 输入: Ijk
# 输出: ijk; flag
# ------ijk: 子列表, 每个元素都是长度为3的字符向量, c(i,j,k)
# ------flag: 长度与ijk相等, flag = 0 <=> μ_{k|h=0} > μ_{k|h=1}
# ---------------------------flag = 1 <=> μ_{k|h=0} !> μ_{k|h=1}

def tri_nodes_stand(Ijk):
    """
    对三叉变量进行标准化处理。

    输入:
      Ijk: 一个长度为 3 的列表或元组，其中：
           Ijk[0] 是 h 的所有可观测子节点列表 I，
           Ijk[1] 是 j（可能为 None，当 I 只有 1 个节点时需要提供 j），
           Ijk[2] 是 k。

    输出:
      返回一个字典，包含两个键：
         "tri_nodes": 一个列表，每个元素都是长度为 3 的列表 [i, j, k]，
                      对应于每个可观测子节点 i 的匹配（三叉变量）。
         "key_node_type": 对应的 flag 列表，
                          flag = 0 表示 μ_{k|h=0} > μ_{k|h=1}，
                          flag = 1 表示 μ_{k|h=0} 不大于 μ_{k|h=1}。

    规则：
      1. 当 I 中有 3 个及以上节点：
         - 如果 k 不在 I 中，则报错；
         - 如果 k 不在 I 的最后位置，则将 k 移至 I 的最后；
         - 对于 I 中前 n-1 个元素（n = len(I)），依次构造三叉变量：
              对于 l = 0 到 n-2, 定义 i = I[l]，j = I[l+1]（当 l == n-2 时 j = I[0]），
              三叉变量为 [i, j, k]，对应 flag = 0；
         - 对于最后一个元素 I[n-1]，令 i = I[n-1]，j = I[1]，k_ = I[0]，构造三叉变量 [i, j, k_]，对应 flag = 1.
      2. 当 I 中恰有 2 个节点：
         - 如果 k 在 I 中，则报错；
         - 构造两个三叉变量：
              第一个为 [I[0], I[1], k]，第二个为 [I[1], I[0], k]，flag 均设为 0.
      3. 当 I 中只有 1 个节点：
         - 如果 k 在 I 中，则报错；
         - 如果 j 为 None，则报错；
         - 构造三叉变量为 [I[0], j, k]，flag 设为 0.
    """

    # print("Ijk: ")
    # print(Ijk)

    I = Ijk["I"]
    j = Ijk["j"]
    k = Ijk["k"]

    # 初始化 tri_nodes 列表，长度与 I 相同；以及 flag 列表
    tri_nodes = [None] * len(I)
    flag = [-1] * len(I)

    if len(I) >= 3:  # ---------------------如果I中有多于3个节点
        if k not in I:  # ------如果k不在I中，报错
            raise ValueError("The node 'k' should belong to I!")
        if I[-1] != k:  # -----------------如果k不在I的最后，则把k放到I的最后
            I = [x for x in I if x != k] + [k]
        # 对于 I 中前 n-1 个元素，构造三叉变量
        for l in range(len(I) - 1):
            i_val = I[l]
            # 若 l < n-2，则 j = I[l+1]，否则（l == n-2）令 j = I[0]
            if l < len(I) - 2:
                j_new = I[l + 1]
            else:
                j_new = I[0]
            tri_nodes[l] = [i_val, j_new, k]
            flag[l] = 0
        # 对最后一个元素 I[n-1]
        i_val = I[-1]
        # 定义 k_ = I[0]，j = I[1]
        if len(I) < 2:
            raise ValueError("I should have at least 2 elements here")
        k_ = I[0]
        j_new = I[1]
        tri_nodes[-1] = [i_val, j_new, k_]
        flag[-1] = 1

    elif len(I) == 2:
        if k in I:  # ------如果k在I中，报错
            raise ValueError("The node 'k' shouldn't belong to I!")
        flag = [0, 0]
        tri_nodes[0] = [I[0], I[1], k]
        tri_nodes[1] = [I[1], I[0], k]

    elif len(I) == 1:
        if k in I:  # ------如果k在I中，报错
            raise ValueError("The node 'k' shouldn't belong to I!")
        # 如果 j_val 为 None，则报错
        if j is None:
            raise ValueError("The node 'j' shouldn't be NULL!")
        tri_nodes[0] = [I[0], j, k]
        flag = [0]
    else:
        raise ValueError("I is empty!")

    result = {"tri_nodes": tri_nodes, "key_node_type": flag}
    return result


# 三岔变量(可观测子节点)
# 得到h的三岔变量I,j,k; 其中I为h的所有可观测子节点集，k为与h关联的可观测子节点(\mu_{k|h=0} > \mu_{k|h=1})
# ----若I只有一个节点i，则j为h的某个隐子节点的可观测后代，k为h的非后代可观测节点；
# ----若I中有两个节点，则j = NULL，k为一个非I的可观测节点；
# ----若I中有多于三个节点，则j = NULL，且k = I[length(I)]
# 注：1. 在参数估计过程中，每个h都只能对应一个节点k(key_node)
# ----2. I中的节点都是需要估计参数的节点(Γ_{i|h})
# 输入: h, tree
# ------h: 字符串，隐节点, 如"H1"
# ------tree: h所在的树结构
# 输出: {I; j; k}, {key_node: H1:k1; H2:k2; H3:k3}
def tri_nodes_obs(h, tree):
    """
    根据隐节点 h 和树结构 tree 得到 h 的三岔变量（三叉变量）。

    输入:
      h: 字符串，隐节点，如 "H1"。如果 h 不是以 "H" 开头，则报错。
      tree: 字典，包含以下键：
            "obs_nodes": 可观测节点列表
            "lat_nodes": 潜变量节点列表
            "child": 子节点关系字典 {node: [child nodes]}
            "des_nodes": 后代节点关系字典 {node: [所有后代节点]}
            "D": 信息距离矩阵（numpy 数组或 pandas DataFrame）

    输出:
      dict: 包含键 "I", "j", "k" 其中：
            I: h 的所有可观测子节点集 (intersection(child, obs_nodes))
            j: 根据规则匹配得到的中间节点（若适用，否则为 None）
            k: 与 h 关联的可观测子节点（满足 μ_{k|h=0} > μ_{k|h=1} 的条件，具体由距离最小决定）

    规则说明（与 R 代码一致）：
      - 若 h 的子节点数 ≥ 3:
          * 若 I（可观测子节点）数 ≥ 3，则 k = I 中最后一个节点。
          * 若 I 数 = 2，则令 K = (de[h] ∩ obs_nodes) \ I，
            并计算对每个 k ∈ K, d(k)=∑_{i∈I} D[i,k]，选 d(k) 最小者为 k。
          * 若 I 数 = 1，则：
              - 令 J = de[H[0]] ∩ obs_nodes，计算 d(j)=D[I,j]，取最小者作为 j；
              - 令 K = de[H[1]] ∩ obs_nodes，计算 d(k)=D[I,k]，取最小者作为 k。
          * 若 I 为空，则警告并返回 None。
      - 若 h 的子节点数 = 2:
          * 若 I 数 = 2，则令 K = obs_nodes \ I，
            计算 d(k)=∑_{i∈I} D[i,k]，取最小者作为 k。
          * 若 I 数 = 1，则：
              - 令 J = de[H[0]] ∩ obs_nodes，计算 d(j)=D[I,j]，取最小者作为 j；
              - 令 K = obs_nodes \ de[h]，计算 d(k)=D[I,k]，取最小者作为 k。
          * 若 I 为空，则警告并返回 None.
      - 否则（子节点数不足2）报错。
    """
    # 检查 h 是否为隐节点
    if not h.startswith("H"):
        raise ValueError("Node 'h' should be latent!")

    obs_nodes = tree["obs_nodes"]  # 可观测节点列表
    lat_nodes = tree["lat_nodes"]  # 潜变量节点列表
    child_dict = tree["child"]  # 子节点关系字典
    des_nodes = tree["des_nodes"]  # 后代节点关系字典
    D = tree.get("D", None)  # 信息距离矩阵

    # print(D)

    # 如果 D 为空，则构造一个全为1且对角线为0的矩阵
    if D is None:
        n_obs = len(obs_nodes)
        D = np.ones((n_obs, n_obs))
        np.fill_diagonal(D, 0)
        # 如果需要，也可将 D 包装为 pandas DataFrame：D = pd.DataFrame(D, index=obs_nodes, columns=obs_nodes)

    # 获取 h 的所有子节点
    if h not in child_dict:
        warnings.warn(f"Node {h} has no child nodes in tree!")
        return None
    child_h = child_dict[h]

    # I: h 的所有可观测子节点 = intersection(child_h, obs_nodes)
    I = [node for node in child_h if node in obs_nodes]
    j = None  # 初始化 j
    # H: h 的所有潜变量子节点 = intersection(child_h, lat_nodes)
    H = [node for node in child_h if node in lat_nodes]

    # print("h: ", h, "I: ", I)

    # 辅助函数：从 D 中提取子矩阵并按列求和
    def sum_distance(rows, cols):
        # rows, cols 为节点名称列表，从 obs_nodes 中获取其索引
        row_idx = [obs_nodes.index(x) for x in rows]
        col_idx = [obs_nodes.index(x) for x in cols]
        # 若 D 为 pandas DataFrame，则转换为 numpy 数组
        if isinstance(D, pd.DataFrame):
            D_mat = D.values
        else:
            D_mat = D
        sub = D_mat[np.ix_(row_idx, col_idx)]
        return sub.sum(axis=0)

    # 根据 h 的子节点数目进行分支
    if len(child_h) >= 3:  # -----------------------------当h有多于3个子节点时
        if len(I) >= 3:  # ---------------------------当I有多于3个节点时,I: h 的所有可观测子节点
            # 当 I 有多于3个节点时，k 取 I 的最后一个
            k = I[-1]
        elif len(I) == 2:  # ---------------------------当I有2个节点时,I: h 的所有可观测子节点
            # K = (de[h] ∩ obs_nodes) \ I
            de_h = des_nodes.get(h, [])
            K = [node for node in de_h if (node in obs_nodes) and (node not in I)]
            if len(K) == 0:
                warnings.warn(f"{h} should have observed child nodes!")
                return None
            # 对于每个 k in K，计算 d(k)=∑_{i∈I} D[i,k]
            d_K = sum_distance(I, K)
            # 选取 d(k) 最小的那个作为 k
            k = K[np.argmin(d_K)]
        elif len(I) == 1:  # ---------------------------当I有1个节点时,I: h 的所有可观测子节点
            if len(H) < 2:  # H: h 的所有潜变量子节点
                warnings.warn(f"{h} should have at least 2 latent child nodes when I has 1 node!")
                return None
            # J = de[H[0]] ∩ obs_nodes
            de_H1 = des_nodes.get(H[0], [])
            J = [node for node in de_H1 if node in obs_nodes]
            if len(J) == 0:
                warnings.warn(f"{h} should have observed child nodes!")
                return None
            # d_IJ = D[I, J]，注意 I 只有一个元素
            I_node = I[0]
            I_index = obs_nodes.index(I_node)
            J_indices = [obs_nodes.index(x) for x in J]
            D_mat = D.values if isinstance(D, pd.DataFrame) else D
            d_IJ = D_mat[I_index, J_indices]
            j = J[np.argmin(d_IJ)]
            # K = de[H[1]] ∩ obs_nodes
            de_H2 = des_nodes.get(H[1], [])
            K = [node for node in de_H2 if node in obs_nodes]
            if len(K) == 0:
                warnings.warn(f"{h} should have observed child nodes!")
                return None
            K_indices = [obs_nodes.index(x) for x in K]
            d_IK = D_mat[I_index, K_indices]
            k = K[np.argmin(d_IK)]
        else:  # ---------------------------当I是空集时
            warnings.warn(f"{h} have no observed child nodes!")
            return None
    elif len(child_h) == 2:  # -----------------------------当h只有2个子节点时
        if len(I) == 2:  # ---------------------------当I有2个节点时,I: h 的所有可观测子节点
            # K = obs_nodes \ I
            K = [node for node in obs_nodes if node not in I]
            if len(K) == 0:
                warnings.warn(f"{h} should have observed child nodes!")
                return None
            d_K = sum_distance(I, K)
            k = K[np.argmin(d_K)]
        elif len(I) == 1:  # ---------------------------当I有1个节点时,I: h 的所有可观测子节点
            # J = de[H[0]] ∩ obs_nodes
            if len(H) == 0:  # H: h 的所有潜变量子节点
                warnings.warn(f"{h} should have observed child nodes!")
                return None
            de_H = des_nodes.get(H[0], [])
            J = [node for node in de_H if node in obs_nodes]
            if len(J) == 0:
                warnings.warn(f"{h} should have observed child nodes!")
                return None
            I_index = obs_nodes.index(I[0])
            J_indices = [obs_nodes.index(x) for x in J]
            D_mat = D.values if isinstance(D, pd.DataFrame) else D
            d_IJ = D_mat[I_index, J_indices]
            j = J[np.argmin(d_IJ)]  # j来自于h的某个隐子节点的后代可观测节点
            # K = obs_nodes \ de[h]
            de_h = des_nodes.get(h, [])
            K = [node for node in obs_nodes if node not in de_h]
            if len(K) == 0:
                warnings.warn(f"{h} should have observed child nodes!")
                return None
            K_indices = [obs_nodes.index(x) for x in K]
            d_IK = D_mat[I_index, K_indices]
            k = K[np.argmin(d_IK)]  # k来自于h的非后代可观测节点
        else:  # ---------------------------当I是空集时
            warnings.warn(f"{h} have no observed child nodes!")
            return None
    else:
        raise ValueError(f"{h} should have two child nodes at least!")

    result = {"I": I, "j": j, "k": k}
    return result


# def parameter_generation(model, N_sim=100):
#     """
#     根据具体的 model 生成概率分布的参数，重复 N_sim 次。
#
#     输入:
#       model: pandas DataFrame，模型数据（应包含列 'vertice', 'vertice_type', 'father_vertice'）
#       N_sim: 模拟重复次数 (默认100)
#
#     输出:
#       para: 字典，键为 model 中每个节点（'vertice'）的名称，
#             值为对应的参数数组：
#               - 对于连续节点 ("observable_continuous"): 生成一个形状为 (4, N_sim) 的数组，
#                 行依次为 [mu0, mu1, sigma0, sigma1]，小数点保留1位（mu）和2位（sigma）。
#               - 对于离散节点（包括 "latent" 和 "observable_discrete"）：如果节点没有父节点
#                 (即 father_vertice == ""), 生成一维数组 (长度 N_sim)（即 p_1），
#                 否则生成形状为 (2, N_sim) 的数组，行依次为 [p_{v|0}, p_{v|1}],
#                 所有数值均保留2位小数。
#     """
#     # 记录总节点数 p_t 和可观测节点数 p_o
#     p_t = model.shape[0]
#     # 可观测变量：observable_continuous 和 observable_discrete
#     logi2 = model['vertice_type'] == "observable_discrete"
#     logi3 = model['vertice_type'] == "observable_continuous"
#     p_o = logi2.sum() + logi3.sum()
#
#     # 获取各类型节点的索引（这里用行标签或位置均可）
#     # latent: 隐节点； obs_con: 观测连续； obs_dis: 观测离散
#     latent_idx = model.index[model['vertice_type'] == "latent"].tolist()
#     obs_con_idx = model.index[model['vertice_type'] == "observable_continuous"].tolist()
#     obs_dis_idx = model.index[model['vertice_type'] == "observable_discrete"].tolist()
#
#     # 初始化参数字典，键为每个节点名称（来自 model['vertice']）
#     para = {}
#     for v in model['vertice']:
#         para[v] = None
#
#     # 处理离散型节点（包括隐节点和观测离散节点）
#     # 取 union(latent, obs_dis)
#     discrete_mask = model['vertice_type'].isin(["latent", "observable_discrete"])
#     for idx, row in model[discrete_mask].iterrows():
#         v = row['vertice']
#         # 如果父节点为空（根节点）
#         if row['father_vertice'] == "":
#             # 生成 N_sim 个随机数，均匀分布在 [0.4, 0.6]，保留2位小数
#             a = np.round(np.random.uniform(0.4, 0.6, N_sim), 2)
#             para[v] = a
#         else:
#             # 否则生成两个向量 a 和 b，均匀分布在 [0.1, 0.9]，保留2位小数
#             a = np.round(np.random.uniform(0.1, 0.9, N_sim), 2)
#             b = np.round(np.random.uniform(0.1, 0.9, N_sim), 2)
#             c = np.abs(a - b)
#             # 保证 a 与 b 的差异至少 0.3
#             while np.sum(c < 0.3) > 0:
#                 idxs = np.where(c < 0.3)[0]
#                 a[idxs] = np.round(np.random.uniform(0.1, 0.9, len(idxs)), 2)
#                 b[idxs] = np.round(np.random.uniform(0.1, 0.9, len(idxs)), 2)
#                 c = np.abs(a - b)
#             # 将 a 和 b 按行堆叠为 (2, N_sim) 数组
#             para[v] = np.vstack([a, b])
#
#     # 处理连续型节点（observable_continuous）
#     for idx, row in model.loc[obs_con_idx].iterrows():
#         v = row['vertice']
#         # 生成 mu0, mu1 均匀分布在 [-2, 2]，保留1位小数
#         mu0 = np.round(np.random.uniform(-2, 2, N_sim), 1)
#         mu1 = np.round(np.random.uniform(-2, 2, N_sim), 1)
#         diff = np.abs(mu0 - mu1)
#         # 保证 mu0 与 mu1 的差至少 0.5
#         while np.sum(diff < 0.5) > 0:
#             idxs = np.where(diff < 0.5)[0]
#             mu0[idxs] = np.round(np.random.uniform(-2, 2, len(idxs)), 1)
#             mu1[idxs] = np.round(np.random.uniform(-2, 2, len(idxs)), 1)
#             diff = np.abs(mu0 - mu1)
#         # 生成 sigma0, sigma1 均匀分布在 [0.1, 1]，保留2位小数
#         sigma0 = np.round(np.random.uniform(0.1, 1, N_sim), 2)
#         sigma1 = np.round(np.random.uniform(0.1, 1, N_sim), 2)
#         para[v] = np.vstack([mu0, mu1, sigma0, sigma1])
#
#     return para

def parameter_generation_online(model, t, N_sim=100):
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
    np.random.seed(123)
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
        # 设置初始值 mu0(0) 和 mu1(0) 在 [-2, 2] 区间内生成
        mu0_0 = np.random.uniform(-2, 2, N_sim)  # mu0(0) 初始值
        mu1_0 = np.random.uniform(-2, 2, N_sim)  # mu1(0) 初始值

        # 确保 mu0_0 和 mu1_0 之间的差异至少为 0.5
        diff = np.abs(mu0_0 - mu1_0)
        while np.sum(diff < 0.5) > 0:
            idxs = np.where(diff < 0.5)[0]
            mu0_0[idxs] = np.random.uniform(1, 1.5, len(idxs))
            mu1_0[idxs] = np.random.uniform(4, 4.5, len(idxs))
            diff = np.abs(mu0_0 - mu1_0)

        # 设定稳定值 theta_e0 为 0，表示它最终会趋向于 0
        mu0_0_stable = 0.5  # 对应于 mu0 的最终稳定值
        mu1_0_stable = 3.5  # 对应于 mu1 的最终稳定值


        # 使用 ODE 公式进行参数演化
        mu0 = mu0_0_stable + (mu0_0 - mu0_0_stable) * np.exp(-t)  # 使用 ODE 演化公式
        mu1 = mu1_0_stable + (mu1_0 - mu1_0_stable) * np.exp(-t)  # 使用 ODE 演化公式
        print("mu0内部:", mu0_0)
        print("mu1内部:", mu1_0)

        # 生成 sigma0, sigma1 均匀分布在 [0.1, 1]，保留2位小数
        sigma0 = np.round(np.random.uniform(0.1, 0.5, N_sim), 2)
        sigma1 = np.round(np.random.uniform(0.1, 0.5, N_sim), 2)

        para[v] = np.vstack([mu0, mu1, sigma0, sigma1])

    return para, mu0[0], mu1[0]

def parameter_generation_online_static(model, t, N_sim=100):
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
    np.random.seed(123)
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
        # 设置初始值 mu0(0) 和 mu1(0) 在 [-2, 2] 区间内生成
        mu0_0 = np.random.uniform(-2, 2, N_sim)  # mu0(0) 初始值
        mu1_0 = np.random.uniform(-2, 2, N_sim)  # mu1(0) 初始值

        # 确保 mu0_0 和 mu1_0 之间的差异至少为 0.5
        diff = np.abs(mu0_0 - mu1_0)
        while np.sum(diff < 0.5) > 0:
            idxs = np.where(diff < 0.5)[0]
            mu0_0[idxs] = np.random.uniform(1, 1.5, len(idxs))
            mu1_0[idxs] = np.random.uniform(4, 4.5, len(idxs))
            diff = np.abs(mu0_0 - mu1_0)

        # 设定稳定值 theta_e0 为 0，表示它最终会趋向于 0
        mu0_0_stable = 0.5  # 对应于 mu0 的最终稳定值
        mu1_0_stable = 3.5  # 对应于 mu1 的最终稳定值

        # # 静态
        mu0 =mu0_0
        mu1 =mu1_0
        # 使用 ODE 公式进行参数演化
        # mu0 = mu0_0_stable + (mu0_0 - mu0_0_stable) * np.exp(-t)  # 使用 ODE 演化公式
        # mu1 = mu1_0_stable + (mu1_0 - mu1_0_stable) * np.exp(-t)  # 使用 ODE 演化公式
        # print("mu0内部:", mu0_0)
        # print("mu1内部:", mu1_0)

        # 生成 sigma0, sigma1 均匀分布在 [0.1, 1]，保留2位小数
        sigma0 = np.round(np.random.uniform(0.1, 2.5, N_sim), 2)
        sigma1 = np.round(np.random.uniform(0.1, 2.5, N_sim), 2)

        para[v] = np.vstack([mu0, mu1, sigma0, sigma1])

    return para, mu0[0], mu1[0]


def parameter_generation_online_static_one(model, t, N_sim=100, seed = None):
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

    # 设置随机数种子
    if seed is not None:
        np.random.seed(seed)

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
        # 设置初始值 mu0(0) 和 mu1(0) 在 [-2, 2] 区间内生成
        mu0_0 = np.random.uniform(-2, 2, N_sim)  # mu0(0) 初始值
        mu1_0 = np.random.uniform(-2, 2, N_sim)  # mu1(0) 初始值

        # 确保 mu0_0 和 mu1_0 之间的差异至少为 0.5
        diff = np.abs(mu0_0 - mu1_0)
        while np.sum(diff < 0.5) > 0:
            idxs = np.where(diff < 0.5)[0]
            mu0_0[idxs] = np.random.uniform(1, 1.5, len(idxs))
            mu1_0[idxs] = np.random.uniform(4, 4.5, len(idxs))
            diff = np.abs(mu0_0 - mu1_0)

        # 设定稳定值 theta_e0 为 0，表示它最终会趋向于 0
        mu0_0_stable = 0.5  # 对应于 mu0 的最终稳定值
        mu1_0_stable = 3.5  # 对应于 mu1 的最终稳定值

        # 使用 ODE 公式进行参数演化
        mu0 = mu0_0_stable + (mu0_0 - mu0_0_stable) * np.exp(-t)  # 使用 ODE 演化公式
        mu1 = mu1_0_stable + (mu1_0 - mu1_0_stable) * np.exp(-t)  # 使用 ODE 演化公式
        # print("mu0:", mu0)
        # print("mu1-mu0:", mu1-mu0)

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
    # 自定义排序函数，按照数字大小排序
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
            father_idx = vertice_to_index[father_v]  # 转换父节点名称到整数索引
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

def data_generation_one(i, model, para_all, n, seed = None):
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
    # 设置随机数种子
    if seed is not None:
        np.random.seed(seed)
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
    # 自定义排序函数，按照数字大小排序
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
            father_idx = vertice_to_index[father_v]  # 转换父节点名称到整数索引
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


# 处理参数估计后，数据的生成。此时带入的参数中连续变量的参数是均值和方差。原来的直接生成参数后产生数据的版本使用的似乎是标准差
def data_generation2(i, model, para_all, n):
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

    print("model: ", model)

    p_t = model.shape[0]
    # 获取各类节点的行号（0-based）
    latent = model.index[logi_latent].tolist()
    obs_con = model.index[logi_obs_cont].tolist()
    obs_dis = model.index[logi_obs_disc].tolist()

    print("p_t: ", p_t)

    # 初始化数据矩阵，n x p_t，填充 -1
    data = -np.ones((n, p_t))

    # 建立一个从节点名称到行索引的映射（假设模型中行顺序与列顺序一致）
    vertice_list = model['vertice'].tolist()
    vertice_to_index = {v: idx for idx, v in enumerate(vertice_list)}

    print("vertice_to_index: ", vertice_to_index)

    # 处理离散型节点（包括隐节点和观测离散节点）
    # 自定义排序函数，按照数字大小排序
    def sort_numeric(x):
        if x.startswith("H"):
            return (1, int(x[1:]))
        else:
            return (0, int(x))

    discrete_nodes = sorted(latent, key=sort_numeric) + sorted(obs_dis, key=sort_numeric)

    print("discrete_nodes", discrete_nodes)

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
            father_idx = vertice_to_index[father_v]  # 转换父节点名称到整数索引

            # print("node: ",node,"father_v: ",father_v,"p_j: ",p_j)
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
        # print("node: ",node,"p_j: ",p_j)
        father_v = model.loc[node, "father_vertice"]
        father_idx = vertice_to_index[father_v]
        if data[0, father_idx] == -1:
            print(f"警告：节点 {v} 的父节点 {father_v} 数据尚未生成！")
        # 根据父节点值判断使用 p_j 的哪一组参数：
        # 如果父节点值为 0，则 mu = p_j[0]，sigma = p_j[2]；否则 mu = p_j[1]，sigma = p_j[3]
        mu = np.where(data[:, father_idx] == 0, p_j[0], p_j[1])
        sigma2 = np.where(data[:, father_idx] == 0, p_j[2], p_j[3])
        # 对每个样本生成一个正态分布随机数（利用向量化生成 n 个样本）
        result = np.random.normal(mu, np.sqrt(sigma2))
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


# 三叉变量(隐子节点)
# 针对输入的隐节点h1，需要计算出对于h1的每一个隐子节点h2，为了估计Pr(h2|h1)所需的三分叉变量tri_nodes，以及对应的key_node信息，以及key_node_type信息与其隐子节点h2, h3的参数，
# 对h1的每一个隐子节点h2，记录h1,h2的三分叉变量，保持其中有一个观测变量i一致，从而保证在进一步的计算中通过计算Pr(i|h2),Pr(i|h1),进一步计算Pr(h2|h1)
# 针对不同的情形，记录h1的每一个隐子节点h2的key_node_type是为0，或是为1。
# h2的key_node_type为0表示h1的隐状态h1=0或是1还没有确定，可根据分解估计时固定\mu_{k|h1 = 0} > \mu_{k|h1 = 1}来确定h1的两种状态。
# h2的key_node_type为1表示h1的隐状态h1=0或是1在之前的计算中已经确定过，在进行分解估计时需要进行校准检查，匹配之前已经确定的h1的隐状态
def tri_nodes_lat(h1, key_node, tree):
    """
    针对隐节点 h1，生成用于隐-隐参数估计的三叉变量。

    输入:
      h1: str，隐节点名称（例如 "H1"），必须以 "H" 开头
      key_node: dict，每个隐节点对应的 k 节点（满足：μ_{k|h1=0} > μ_{k|h1=1}）
      tree: dict，包含隐树结构和距离矩阵 D（必须为 numpy.ndarray）

    输出:
      dict:
        - "tri_nodes": 每个隐子节点的三叉变量 DataFrame
        - "key_node": 更新后的 key_node
        - "key_node_type": flag: 0 or 1
    """
    if not h1.startswith("H"):
        raise ValueError("Node 'h1' should be latent!")

    obs_nodes = tree["obs_nodes"]
    lat_nodes = tree["lat_nodes"]
    pa = tree["pa"]
    ch = tree["child"]
    de = tree["des_nodes"]
    D = tree["D"]

    obs_idx = {v: i for i, v in enumerate(obs_nodes)}

    if D is None:
        n = len(obs_nodes)
        D = np.ones((n, n))
        np.fill_diagonal(D, 0)

    child = ch[h1]
    H = [x for x in child if x in lat_nodes]
    V = [x for x in child if x in obs_nodes]

    if len(H) == 0:
        warnings.warn("The latent node 'h1' should have a latent child at least!")
        return None

    I_H = {}  # ---------估计隐-隐参数的核心节点
    ijk_dict = {}
    flag = {}  # -------k2节点的属性:
    # 0: \mu_{k2|h1 = 0} > \mu_{k2|h1 = 1} <-> 1: \mu_{k2|h1 = 0} ? \mu_{k2|h1 = 1}
    # 产生出h1的隐子节点的三叉变量
    for h_u in H:  # H是h1的所有隐子节点
        k_u = key_node.get(h_u, "")  # -----------------------------h_u对应的k
        if k_u == "":
            raise ValueError(f"Missing key_node for {h_u}")
        temp = tri_nodes_obs1(h_u, k=k_u, tree=tree)
        I_H[h_u] = temp[0]  # h1的隐子节点的三分叉变量中的第一个
        ijk_dict[h_u] = temp

    # print("I_H: ",I_H,"type(h_u): ",type(h_u),"type(k_u): ",type(k_u))

    print("ijk_dict: ")
    print(ijk_dict)

    ijkjk = {}

    def dist_sum(I_list, target_list):
        # print("I_list: ",I_list)
        # print("target_list: ",target_list)
        row = [obs_idx[str(x)] for x in I_list]
        col = [obs_idx[str(y)] for y in target_list]
        # print("row: ",row)
        # print("col: ",col)
        subD = D[np.ix_(row, col)]
        return subD.sum(axis=0)

    if len(V) >= 2:  # -------------------------------------------------如果h1有超过2个可观测子节点

        print("len(V) >= 2:")
        if len(H) == 1:  # ----------------------------------如果h1只有1个隐子节点
            i = I_H[H[0]]
            d_V = [D[obs_idx[i], obs_idx[v]] for v in V]
        else:  # ----------------------------------------------如果h1至少有2个隐子节点
            I_vals = [I_H[h_u] for h_u in H]
            d_V = dist_sum(I_vals, V)
        sorted_idx = np.argsort(d_V)
        j2, k2 = V[sorted_idx[1]], V[sorted_idx[0]]

        for h_u in H:
            temp1 = ijk_dict[h_u]
            temp2 = [I_H[h_u], j2, k2]
            colname = "root" if pa[h1] == "" else h1
            df = pd.DataFrame({h_u: temp1, colname: temp2}, index=["1", "2", "3"])
            ijkjk[h_u] = df
            flag[h_u] = 1  # --------------------------------------(\mu_{k2|h1 = 0} ? \mu_{k2|h1 = 1})
            # 这是因为此时\mu_{k2|h1 = 0}和\mu_{k2|h1 = 1}已经估计完成，
            # 新的以k2为key_node的分解估计时，要保持特征值的顺序与\mu_{k2|h1 = 0}和\mu_{k2|h1 = 1}一致，
            # 需要校准检查，从而使得h1对应的隐状态一致。
            # flag[h_u] = 1表示进行分解时需要进行校准检查。flag[h_u] = 0表示分解时直接按特征值由大到小排列设置隐变量状态即可

    elif len(V) == 1:  # ------------------------如果h1有1个可观测子节点，指定其为k2
        print("len(V) == 1:")
        # --若h1是root，则从I_H中选j2；
        # --若h1不是root，从非h1后代的可观测子节点中选出j2
        k2 = V[0]
        if pa[h1] == "":  # -------------------------------如果h1是根节点，则j2从I_H中选
            for l, h_u in enumerate(H):
                ll = 0 if l == len(H) - 1 else l + 1
                j2 = I_H[H[ll]]
                temp1 = ijk_dict[h_u]
                df = pd.DataFrame({h_u: temp1, "root": [I_H[h_u], j2, k2]}, index=["1", "2", "3"])
                ijkjk[h_u] = df
                flag[h_u] = 1  # ------------------------------------(\mu_{k2|h1 = 0} ? \mu_{k2|h1 = 1})
        else:  # ------------------------------------------如果h1有父节点,选定一个公共的j2
            J2 = [x for x in obs_nodes if x not in de[h1]]
            if len(H) == 1:  # ----------------------------------如果h1只有1个隐子节点
                d_J = [D[obs_idx[I_H[H[0]]], obs_idx[j]] for j in J2]
            else:  # ----------------------------------------------如果h1至少有2个隐子节点
                I_vals = [I_H[h_u] for h_u in H]
                d_J = dist_sum(I_vals, J2)
            j2 = J2[np.argmin(d_J)]
            for h_u in H:
                temp1 = ijk_dict[h_u]
                df = pd.DataFrame({h_u: temp1, h1: [I_H[h_u], j2, k2]}, index=["1", "2", "3"])
                ijkjk[h_u] = df
                flag[h_u] = 1  # (\mu_{k2|h1 = 0} ? \mu_{k2|h1 = 1})。
                # 这是因为此时\mu_{k2|h1 = 0}和\mu_{k2|h1 = 1}已经估计完成，
                # 新的以k2为key_node的分解估计时，要保持特征值的顺序与\mu_{k2|h1 = 0}和\mu_{k2|h1 = 1}一致，
                # 需要校准检查，从而使得h1对应的隐状态一致。
                # flag[h_u] = 1表示进行分解时需要进行校准检查。flag[h_u] = 0表示分解时直接按特征值由大到小排列设置隐变量状态即可

    elif len(V) == 0:  # ------------------------如果h1没有可观测子节点
        print("len(V) == 0:")
        if key_node.get(h1, "") != "":
            raise ValueError("The latent node 'h1' should not have key_node!")

        I_vals = [I_H[h_u] for h_u in H]

        if len(H) >= 3:  # -----------------------------如果h1有多于3个隐子节点
            # --若h1是root，则从I_H中选j2、k2；
            # --若h1不是root，则从I_H中选j2，从非h1后代的可观测子节点中选出k2
            if pa[h1] == "":  # -----------------------------如果h1是根节点
                # 前n-1个分解利用\mu_{k2|h1 = 0} > \mu_{k2|h1 = 1}进行对应，保持每个分解估计的关于h1的隐状态一致。flag[h_u] = 0
                # 对于最后一个需要利用之前的估计进行校准测试flag[h_u] = 1
                # ---------------------------------前length(H)-1个隐节点
                for l in range(len(H) - 1):
                    h_u = H[l]
                    ll = 0 if l == len(H) - 2 else l + 1
                    j2 = I_H[H[ll]]
                    k2 = I_H[H[-1]]
                    df = pd.DataFrame({h_u: ijk_dict[h_u], "root": [I_H[h_u], j2, k2]}, index=["1", "2", "3"])
                    ijkjk[h_u] = df
                    flag[h_u] = 0  # --------------------------------(\mu_{k2|h1 = 0} < \mu_{k2|h1 = 1})
                # ---------------------------------第length(H)个隐节点
                h_u = H[-1]
                print("I_H: ", I_H)
                I_H_list = [I_H[x] for x in H]
                df = pd.DataFrame({h_u: ijk_dict[h_u], "root": [I_H[h_u], I_H_list[1], I_H_list[0]]},
                                  index=["1", "2", "3"])

                ijkjk[h_u] = df
                flag[h_u] = 1  # 利用之前的估计进行校准测试(\mu_{k2|h1 = 0} ? \mu_{k2|h1 = 1})
                key_node[h1] = I_H[H[-1]]  # ----------------h1的key_node设为I_H的最后一个隐-隐参数的核心节点
            else:  # ----------------------------------------如果h1有父节点,选定一个公共的k2
                K = [x for x in obs_nodes if x not in de[h1]]
                d_K = dist_sum(I_vals, K)
                k2 = K[np.argmin(d_K)]  # -------h1对应的公共的k2
                # ---------------------------------选出与每个h_u对应的j2
                for l, h_u in enumerate(H):
                    ll = 0 if l == len(H) - 1 else l + 1
                    j2 = I_H[H[ll]]
                    df = pd.DataFrame({h_u: ijk_dict[h_u], h1: [I_H[h_u], j2, k2]}, index=["1", "2", "3"])
                    ijkjk[h_u] = df
                    flag[h_u] = 0  # -------------------------------------(\mu_{k2|h1 = 0} > \mu_{k2|h1 = 1})
                    # 此时\mu_{k2|h1 = 0}，\mu_{k2|h1 = 1}还没有计算过，通过设置k2为key_node，分解估计时只需要求\mu_{k2|h1 = 0} > \mu_{k2|h1 = 1}，可以固定h1的两个状态取值
                key_node[h1] = k2  # --------------------------h1的key_node设为新选出的k2

        elif len(H) == 2:
            K = [x for x in obs_nodes if x not in de[h1]]
            print("K in len(H) == 2: ", K)
            d_K = dist_sum(I_vals, K)
            k2 = K[np.argmin(d_K)]  # -------h1对应的公共的k2
            # ---------------------------------选出与每个h_u对应的j2
            for l, h_u in enumerate(H):
                ll = 0 if l == len(H) - 1 else l + 1
                j2 = I_H[H[ll]]
                df = pd.DataFrame({h_u: ijk_dict[h_u], h1: [I_H[h_u], j2, k2]}, index=["1", "2", "3"])
                ijkjk[h_u] = df
                flag[h_u] = 0  # --------------------------------------(\mu_{k2|h1 = 0} > \mu_{k2|h1 = 1})
                # 此时\mu_{k2|h1 = 0}，\mu_{k2|h1 = 1}还没有计算过，通过设置k2为key_node，分解估计时只需要求\mu_{k2|h1 = 0} > \mu_{k2|h1 = 1}，可以固定h1的两个状态取值
            key_node[h1] = k2  # ----------------------------h1的key_node设为新选出的k2
        else:
            raise ValueError("The latent node 'h1' should have 3 neighbors at least!")

    return {"tri_nodes": ijkjk, "key_node": key_node, "key_node_type": flag}


# 带非负约束的最小二乘问题（Nonnegative Least Squares, NNLS）。
# 一种常见的处理方法是将矩阵变量向量化，将问题转换为标量变量的 NNLS 问题，然后利用标准求解器求解。
def solve_nnls_2x2(temp_ch, temp_pa):
    """
    求解 2x2 的非负最小二乘问题:
        min_X || temp_ch * X - temp_pa ||_F^2
    subject to: X 中每个元素 >= 0

    参数:
      temp_ch: 2x2 numpy 数组
      temp_pa: 2x2 numpy 数组

    返回:
      X: 2x2 numpy 数组，满足非负约束的解
    """
    # 向量化 X：这里采用列堆叠（column-stacking）
    # 构造矩阵 A = I_2 ⊗ temp_ch
    I2 = np.eye(2)
    A = np.kron(I2, temp_ch)  # 4x4 矩阵
    # 向量化 temp_pa
    b = temp_pa.flatten('F')  # 'F' 表示 Fortran 风格，即列优先

    beta = np.linalg.solve(A, b)
    # print("beta: ",beta)
    # A有可能条件数非常大，采用类GAGA正则化方法。lam过大估计出的离散变量之间转移概率值会倾向于0.5
    lam = 1e-3
    res = lsq_linear(A.T @ A + np.diag(lam / abs(beta)), A.T @ b, bounds=(1e-3, np.inf))

    # # # 求解非负最小二乘问题
    # res = lsq_linear(A, b, bounds=(0, np.inf))

    x = res.x

    # 将解 x 转换为 2x2 矩阵 X（列优先）
    X = x.reshape((2, 2), order='F')
    return X


# 参数估计(针对某个隐节点的所有隐子节点)
# 针对每个隐节点h输出的三叉变量组{I,j,k}, 我们得到任意i∈I, 参数矩阵Γ_{i|h}
# 注: k∈I的情况与k!∈I的情况; 还有j = NULL的情况
# 输入: Ijk, AIjk
# ------IJK: list(I, j, k)
# ----------I: h的所有可观测子节点
# ----------j: 当I中只有一个节点时，凑数的
# ----------k: h对应的可观测节点, 满足 \mu_{k|h=0} > \mu_{k|h=1}
# ------AIjk: list(I[1]:A_(I[1])jk, I[2]:A_(I[2])jk, ...)
# 输出: {I[1]:{...}; I[2]:{...}; I[3]:{...}; ...}
def para_esti_lat(EXYZ11_online,EXYZ12_online,EXYZ21_online,EXYZ22_online,EXY11_online, EXY12_online,EXY21_online,EXY22_online, EX2_online, EX_online,
                  EXYZ_update, EXY_update, EX_update,ijk, h=None, flag=None, para_hat=None, tree=None, data=None, tol=1e-6,
                  model=None, para=None, prob_marg=None, g=None):
    """
    针对某个隐节点 h 的所有隐子节点，利用输入的三叉变量组 ijk 与对应的真实参数
    估计隐-隐之间的条件概率矩阵。

    输入:
      ijk: dict，键为隐子节点名称，值为一个二维数组或 DataFrame，
           每个包含三叉变量信息；通常包含两列：
           第一列：来自子节点的三叉变量；
           第二列：来自父节点的三叉变量；
      h: 隐节点 h1 的名称（字符串）；如果为 None，则从 ijk 的键中推断所有隐子节点的共同父节点
      flag: 数组或 dict，记录每个隐子节点对应的标志（0或1）；默认全为0
      para_hat: dict，预先估计好的参数（例如来自观测变量部分的估计），用于隐-隐估计中对照
      tree: dict，隐树结构，至少包含键 "obs_nodes", "lat_nodes", "pa", "str", "child", "des_nodes", "D"
      data: 如果未输入真实参数，则输入数据，用于调用 GotA 获取参数
      tol: 容差
      model, para, prob_marg: 当输入真实模型参数时使用
      g: 图对象（例如 igraph.Graph）；若为 None，则从 tree 中构造

    输出:
      若 h 为根节点，则返回 dict {"para_lat_hat": ..., "root_para": ...}；
      否则返回 para_lat_hat，其中：
         para_lat_hat: dict，键为每个隐子节点 h_u，值为估计的条件概率矩阵（NumPy 数组，2×2）
    """

    if flag is None:
        flag = [0] * len(ijk)
    # 检查 h 是否以 "H" 开头
    if h is not None and not h.startswith("H"):
        raise ValueError("Node 'h1' should be latent!")

    obs_nodes = tree["obs_nodes"]
    lat_nodes = tree["lat_nodes"]
    pa = tree["pa"]
    st = tree["str"]
    ch = tree["child"]
    de = tree["des_nodes"]
    D = tree.get("D", None)

    # 如果 g 未提供，则根据 pa 构造图（此处略实现，依赖 python-igraph）
    if g is None:
        # 构造边列表，忽略父节点为空的
        edges = [(node, parent) for node, parent in pa.items() if parent != ""]
        # 例如使用 igraph： g = ig.Graph.TupleList(edges, directed=False)
        pass  # 具体实现省略

    # 如果 D 未提供，则构造默认 D 矩阵
    if D is None:
        n_obs = len(obs_nodes)
        D_arr = np.ones((n_obs, n_obs))
        np.fill_diagonal(D_arr, 0)
        D = pd.DataFrame(D_arr, index=obs_nodes, columns=obs_nodes)

    # 检查 ijk 是否为 dict
    if not isinstance(ijk, dict):
        raise ValueError("The object 'ijk' should be a dict!")

    # num_h 为隐子节点个数；I 存放对应每个隐子节点的第一列的值
    child_latent = list(ijk.keys())  # ---------------------------h的所有隐子节点
    num_h = len(child_latent)
    I = {h_u: None for h_u in child_latent}

    # print("I: ",I)

    if h is None:
        # 若 h 未提供，则从所有隐子节点的父节点中取公共父节点
        parents = [pa.get(h_u, None) for h_u in child_latent]
        unique_parents = set(parents)
        if len(unique_parents) != 1 or None in unique_parents:
            raise ValueError("These latent nodes should have a common parent node!")
        h = unique_parents.pop()

    # 对每个隐子节点，从对应的 ijk 值中取第一列的第一行作为 I[h_u]
    for h_u in child_latent:

        temp = ijk[h_u]
        print("h_u: ", h_u)
        # print("temp: ",temp)
        # 假定 temp 为 pandas DataFrame或二维 NumPy 数组
        if isinstance(temp, pd.DataFrame):
            # print("H2 type:", type(temp.iloc[0, 0]))
            I[h_u] = temp.iloc[0, 0]
            temp_colname = temp.columns[0]
        else:
            # print("H2 type:", type(temp[0, 0]))
            I[h_u] = temp[0, 0]
            # 这里没有列名信息，则忽略
            temp_colname = h_u
        if h_u != temp_colname:
            warnings.warn("The names of 'ijk' don't match with the names of latent child nodes!")
            # 用 temp 的列名更新
            child_latent[child_latent.index(h_u)] = temp_colname
    # I 的键保持为 child_latent

    # print("ijk: ",ijk)

    print("I: ", I)

    # 根据 I 中的值判断每个节点的类型，并获取估计输入参数
    # 找出I中的连续节点, 得到估计需要输入的参数
    node_type = {}
    # para_input: list, 第一个是Aijk; 第二个是mu_k; 第三个是mu_i(list), 长度为2
    para_input_ch = {}  # 针对子节点
    para_input_pa = {}  # 针对父节点
    if (para is None) or (model is None) or (prob_marg is None):  # ------------输入的是数据
        if data is None:
            raise ValueError(
                "Please input complete parameters 'model', 'para' and 'para_marg' or the parameter 'data'!")
        for h_u in child_latent:
            i_val = I[h_u]
            # 判断 data 第一行该列是否接近整数,注意这里的data的行列需要时0-based
            val = data[0, int(i_val) - 1]
            if abs(val - int(val)) < tol:
                node_type[i_val] = "discrete"
            else:
                node_type[i_val] = "continuous"

            # print("h_u: ",h_u,"i_val: ",i_val,"node_type[i_val]: ",node_type[i_val],"val: ",val)
            # 调用 GotA 获取参数
            # 假定 GotA 返回 [Aijk, mu_k, mu_i],"Aijk": 估计矩阵 Aijk (2x2 numpy 数组), "mu_k": mu_k，即 X_k 的均值 (标量),
            # # "mu_i": 当变量 i 为连续时，字典 {"mu_i": 均值, "mu_i2": 均值平方}； 当 i 为离散时，设为 None.
            
            para_input_ch[i_val] = GotA(EXYZ11_online,EXYZ12_online,EXYZ21_online,EXYZ22_online,EXY11_online, EXY12_online,EXY21_online,EXY22_online, EX2_online, EX_online,
                                        EXYZ_update, EXY_update, EX_update,
                ijk=ijk[h_u].iloc[:, 0] if isinstance(ijk[h_u], pd.DataFrame) else ijk[h_u][:, 0],
                data=data, tol=tol)
            
            para_input_pa[i_val]= GotA(EXYZ11_online,EXYZ12_online,EXYZ21_online,EXYZ22_online,EXY11_online, EXY12_online,EXY21_online,EXY22_online, EX2_online, EX_online,
                                       EXYZ_update, EXY_update, EX_update,
                ijk=ijk[h_u].iloc[:, 1] if isinstance(ijk[h_u], pd.DataFrame) else ijk[h_u][:, 1],
                data=data, tol=tol)
            
            # print("para_input_ch[i_val]: ",para_input_ch[i_val])
            # print("para_input_pa[i_val]: ", para_input_pa[i_val])
            # print("GotA in para_esti_lat")
    else:  # -------------------------------------------------------------输入的是真实的模型参数
        # 使用真实参数
        for h_u in child_latent:
            i_val = I[h_u]
            # 根据 model 获取节点类型
            node_type[i_val] = model.loc[i_val, model.columns[1]]
            # 得到子节点和父节点的三叉变量
            para_input_ch[i_val] = GotA_true(
                ijk=ijk[h_u].iloc[:, 0] if isinstance(ijk[h_u], pd.DataFrame) else ijk[h_u][:, 0],
                h=h_u, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
            para_input_pa[i_val] = GotA_true(
                ijk=ijk[h_u].iloc[:, 1] if isinstance(ijk[h_u], pd.DataFrame) else ijk[h_u][:, 1],
                h=h, model=model, para=para, prob_marg=prob_marg, g=g, tree=tree)
            # print("GotA_true in para_esti_lat")
    gc.collect()

    # 判断 h 是否为根节点（即 pa[h] == ""）
    # 是否为根节点, 创建列表存储根节点的"边际概率"参数
    is_root = (pa.get(h, "") == "")
    root_para = {}  # 若 h 为根节点，存储每个隐子节点对应的边际概率参数

    # 对每个隐子节点 h_u，检查 ijk 的列名是否与 h_u 匹配
    for h_u in child_latent:
        temp = ijk[h_u]
        # 假定 temp 为 DataFrame，检查第二列的列名是否为 "root"（若 h 为根）或 h
        if isinstance(temp, pd.DataFrame):
            colname2 = temp.columns[1]
        else:
            colname2 = h  # 无列名信息时，暂设为 h
        if (colname2 == "root") != is_root:
            raise ValueError("The colnames of 'ijk' don't match the root!")

    # 生成逻辑向量：对于每个 flag 元素，判断是否等于 0 或 1
    logi0 = [v == 0 for v in flag.values()]
    logi1 = [v == 1 for v in flag.values()]

    print("flag: ", flag, "logi0: ", logi0, "logi1: ", logi1)
    # for key, value in flag.items():
    #     print(f"Key: {key}, Value: {value}, Type: {type(value)}")

    # 假定 flag 是 dict，键为隐子节点
    para_lat_hat = {child: None for child in child_latent}
    para_ih = {}
    if sum(logi0) != 0:  # 如果存在 flag==0 的情况，优先处理

        # 创建一个字典 para_ih，其键为 I 中对应 flag==0 的元素，初始值设为 None
        I_keys = list(I.keys())  # 将字典 I 的键转换为列表
        para_ih = {I_keys[i]: None for i, val in enumerate(logi0) if val}
        print("flag==0")
        # 遍历所有 flag==0 的索引
        for h_u, f in flag.items():
            if f == 0:
                i_val = I[h_u]
                print("h_u: in f==0", h_u, "i_val: ", i_val, "is_root: ", is_root)
                # 获取辅助参数：分别来自子节点部分和父节点部分
                # 对应的 GotA 或 GotA_true 返回 [Aijk, mu_k, mu_i]
                para_input_i_ch = para_input_ch[i_val]
                Aijk_i_ch = para_input_i_ch["Aijk"]
                mu_k_ch = para_input_i_ch["mu_k"]
                mu_i_ch = para_input_i_ch["mu_i"]

                para_input_i_pa = para_input_pa[i_val]
                Aijk_i_pa = para_input_i_pa["Aijk"]
                mu_k_pa = para_input_i_pa["mu_k"]
                mu_i_pa = para_input_i_pa["mu_i"]
                # 判断是否有多个候选（即 Aijk_i_ch 为 list）
                if isinstance(Aijk_i_ch, list) and isinstance(Aijk_i_pa,
                                                              list):  # -------------------------------如果Aijk_i中有不止一个
                    print("如果Aijk_i中有不止一个")
                    p_val = max(len(Aijk_i_ch), len(mu_k_ch), len(Aijk_i_pa), len(mu_k_pa))
                    # 如果长度不一致，则报错
                    if len(Aijk_i_ch) != p_val or len(mu_k_ch) != p_val or len(Aijk_i_pa) != p_val or len(
                            mu_k_pa) != p_val:
                        raise ValueError("'Aijk_i', 'mu_k' and 'mu_i' have different length!")
                    temp_ch = [None] * p_val
                    temp_pa = [None] * p_val
                    temp_lat = [None] * p_val
                    temp_root_para = np.zeros(p_val)
                    for ll in range(p_val):
                        # 调用 para_esti 对 i_val 估计参数，来自子节点部分
                        temp_ch[ll] = para_esti(i=i_val, Aijk=Aijk_i_ch[ll], mu_k=mu_k_ch[ll],
                                                node_type=node_type[i_val],
                                                mu_i=mu_i_ch[0][ll], mu_i2=mu_i_ch[1][ll],
                                                lambda1=None, lambda2=None, root=False)
                        # 对父节点部分
                        temp = para_esti(i=i_val, Aijk=Aijk_i_pa[ll], mu_k=mu_k_pa[ll],
                                         node_type=node_type[i_val],
                                         mu_i=mu_i_pa[0][ll], mu_i2=mu_i_pa[1][ll],
                                         lambda1=None, lambda2=None, root=is_root)
                        if is_root:
                            temp_root_para[ll] = temp["prob_of_root"]
                            temp = temp["eigenvector"]
                        temp_pa[ll] = temp
                        # 计算条件概率矩阵：temp_lat = inv(temp_ch) * temp_pa
                        # P_{h_|h} = Γ^{-1}_{i|h_} %*% Γ_{i|h}; 其中h = pa(h_)
                        if np.linalg.matrix_rank(temp_ch[ll]) < 2:
                            temp_eps = 1 if np.min(np.abs(temp_ch[ll])) == 0 else np.min(np.abs(temp_ch[ll]))
                            temp_ch[ll] = temp_ch[ll] + np.diag([min(temp_eps, 0.1)] * 2)

                        # temp_lat[ll] = np.linalg.solve(temp_ch[ll], temp_pa[ll])
                        # 带非负约束的最小二乘问题（Nonnegative Least Squares, NNLS）。
                        temp_lat[ll] = solve_nnls_2x2(temp_ch[ll], temp_pa[ll])

                        # 对每列归一化
                        temp_lat[ll][:, 0] = temp_lat[ll][:, 0] / np.sum(temp_lat[ll][:, 0])
                        temp_lat[ll][:, 1] = temp_lat[ll][:, 1] / np.sum(temp_lat[ll][:, 1])
                    para_lat_hat[h_u] = temp_lat  # -------------------------隐节点与隐节点之间的条件概率矩阵
                    if is_root:
                        # 保存根节点边际概率参数
                        root_para[h_u] = temp_root_para
                    para_ih[i_val] = temp_pa  # 存储 i|h 条件估计，用于后续 k2 推导
                else:  # -------------------------------------------如果Aijk_i中只有一个
                    # Ensure only one estimate
                    if not (hasattr(mu_k_ch, '__len__') and len(mu_k_ch) == 1) and not np.isscalar(mu_k_ch):
                        raise ValueError("'Aijk_i' and 'mu_k' should have length 1!")

                    mu_i_ch_val = mu_i_ch["mu_i"] if mu_i_ch is not None else None
                    mu_i2_ch_val = mu_i_ch["mu_i2"] if mu_i_ch is not None else None
                    mu_i_pa_val = mu_i_pa["mu_i"] if mu_i_pa is not None else None
                    mu_i2_pa_val = mu_i_pa["mu_i2"] if mu_i_pa is not None else None

                    # print("mu_i_ch_val: ",mu_i_ch_val,"mu_i2_ch_val: ",mu_i2_ch_val)

                    temp_ch = para_esti(i=i_val, Aijk=Aijk_i_ch, mu_k=mu_k_ch, node_type=node_type[i_val],
                                        mu_i=mu_i_ch_val, mu_i2=mu_i2_ch_val,
                                        lambda1=None, lambda2=None, root=False)

                    temp_pa = para_esti(i=i_val, Aijk=Aijk_i_pa, mu_k=mu_k_pa, node_type=node_type[i_val],
                                        mu_i=mu_i_pa_val, mu_i2=mu_i2_pa_val,
                                        lambda1=None, lambda2=None, root=is_root)

                    if is_root:
                        root_para[h_u] = temp_pa["prob_of_root"]
                        temp_pa = temp_pa["eigenvector"]  # ----------------------根节点的边际概率

                    if np.linalg.matrix_rank(temp_ch) < 2 or np.linalg.cond(temp_ch) > 1e4:
                        temp_eps = 1 if np.min(np.abs(temp_ch)) < 1e-5 else np.min(np.abs(temp_ch))
                        temp_ch = temp_ch + np.diag([min(temp_eps, 0.1)] * 2)

                    # para_lat_hat[h_u] = np.linalg.solve(temp_ch, temp_pa)#----------隐节点与隐节点之间的条件概率矩阵

                    # 带非负约束的最小二乘问题（Nonnegative Least Squares, NNLS）。
                    para_lat_hat[h_u] = solve_nnls_2x2(temp_ch, temp_pa)

                    para_lat_hat[h_u][:, 0] = para_lat_hat[h_u][:, 0] / np.sum(para_lat_hat[h_u][:, 0])
                    para_lat_hat[h_u][:, 1] = para_lat_hat[h_u][:, 1] / np.sum(para_lat_hat[h_u][:, 1])

                    para_ih[i_val] = temp_pa  # --------------------------------存储{i|h}的参数

    # 处理 flag==1 的情况（类似逻辑，使用 k2 信息）
    if sum(logi1) != 0:
        print("flag==1")

        for h_u, f in flag.items():  # ----------------------------如果存在 flag=1 的情况(只有极少数情况下才不存在)
            if f == 1:

                # print("h_u: ",h_u)

                i_val = I[h_u]
                ijk_i = ijk[h_u]

                print("h_u: in f==1", h_u, "i_val: ", i_val, "is_root: ", is_root)
                print("ijk_i: ", ijk_i)
                # 从 ijk_i 提取 k2
                k2 = ijk_i.iloc[2, 1]

                print("k2: ", k2)
                para_input_i_ch = para_input_ch[i_val]
                Aijk_i_ch = para_input_i_ch["Aijk"]
                mu_k_ch = para_input_i_ch["mu_k"]
                mu_i_ch = para_input_i_ch["mu_i"]

                para_input_i_pa = para_input_pa[i_val]
                Aijk_i_pa = para_input_i_pa["Aijk"]
                mu_k_pa = para_input_i_pa["mu_k"]
                mu_i_pa = para_input_i_pa["mu_i"]

                # print("from para_hat:", para_hat)
                # print("from para_ih:", para_ih)
                # print("Aijk_i_pa:-如果存在 flag=1 的情况(只有极少数情况下才不存在)")
                # print(Aijk_i_pa)

                if isinstance(Aijk_i_ch, list) and isinstance(Aijk_i_pa,
                                                              list):  # -------------------------------如果Aijk_i中有不止一个
                    p_val = max(len(Aijk_i_ch), len(mu_k_ch), len(Aijk_i_pa), len(mu_k_pa))
                    if not (len(Aijk_i_ch) == p_val and len(mu_k_ch) == p_val and len(Aijk_i_pa) == p_val and len(
                            mu_k_pa) == p_val):
                        raise ValueError("'Aijk_i', 'mu_k' and 'mu_i' have different length!")
                    temp_ch = [None] * p_val
                    temp_pa = [None] * p_val
                    temp_lat = [None] * p_val
                    temp_root_para = np.zeros(p_val)
                    for ll in range(p_val):
                        # print("ll 如果Aijk_i中有不止一个: ",ll)
                        # 根据 k2 是否在 I 中以及 k2 的类型获取 lambda 参数
                        if pa.get(k2, None) == h:
                            temp_mat = para_hat[k2][ll]

                            if k2 not in I:  # -----------------------pa[k2] == h时，k2不一定在I中
                                # 判断k2是否是离散的
                                if model is not None:
                                    if model.loc[k2, model.columns[1]] == "observable_continuous":
                                        lambda_val = temp_mat[0, :]
                                    else:
                                        lambda_val = temp_mat[1, :]
                                elif data is not None:
                                    if abs(data[0, int(k2) - 1] - int(
                                            data[0, int(k2) - 1])) > tol:  # 这里需要注意data的行与列假定是0-based，所以顶点的索引值-1
                                        lambda_val = temp_mat[0, :]
                                    else:
                                        lambda_val = temp_mat[1, :]
                                else:
                                    raise ValueError("Error input of 'data' and 'model'!")

                            elif node_type.get(k2, "") == "continuous":
                                lambda_val = temp_mat[0, :]
                            elif node_type.get(k2, "") == "discrete":
                                lambda_val = temp_mat[1, :]
                        else:
                            if para_ih is None:
                                raise ValueError("The temporary parameter 'para_ih' was lost!")
                            if k2 not in [I[x] for x in list(flag.keys()) if flag[x] == 0]:
                                raise ValueError("The conditional parameters of 'k2' under 'h' were lost!")
                            temp_mat = para_ih[k2][ll]
                            if k2 not in I:
                                raise ValueError(f"The node {k2} should belong to 'I'!")
                            elif node_type.get(k2, "") == "continuous":
                                lambda_val = temp_mat[0, :]
                            elif node_type.get(k2, "") == "discrete":
                                lambda_val = temp_mat[1, :]
                        temp_ch[ll] = para_esti(i=i_val, Aijk=Aijk_i_ch[ll], mu_k=mu_k_ch[ll],
                                                node_type=node_type[i_val],
                                                mu_i=mu_i_ch[0][ll], mu_i2=mu_i_ch[1][ll],
                                                lambda1=None, lambda2=None, root=False)
                        temp = para_esti(i=i_val, Aijk=Aijk_i_pa[ll], mu_k=mu_k_pa[ll],
                                         node_type=node_type[i_val],
                                         mu_i=mu_i_pa[0][ll], mu_i2=mu_i_pa[1][ll],
                                         lambda1=lambda_val[0], lambda2=lambda_val[1], root=is_root)
                        if is_root:
                            temp_root_para[ll] = temp["prob_of_root"]
                            temp = temp["eigenvector"]
                        temp_pa[ll] = temp
                        # P_{h_|h} = Γ^{-1}_{i|h_} %*% Γ_{i|h}; 其中h = pa(h_)
                        if np.linalg.matrix_rank(temp_ch[ll]) < 2:
                            temp_eps = 1 if np.min(np.abs(temp_ch[ll])) < 1e-5 else np.min(np.abs(temp_ch[ll]))
                            temp_ch[ll] = temp_ch[ll] + np.diag([min(temp_eps, 0.1)] * 2)
                        # temp_lat[ll] = np.linalg.solve(temp_ch[ll], temp_pa[ll])

                        ##带非负约束的最小二乘问题（Nonnegative Least Squares, NNLS）。
                        temp_lat[ll] = solve_nnls_2x2(temp_ch[ll], temp_pa[ll])

                        temp_lat[ll][:, 0] = temp_lat[ll][:, 0] / np.sum(temp_lat[ll][:, 0])
                        temp_lat[ll][:, 1] = temp_lat[ll][:, 1] / np.sum(temp_lat[ll][:, 1])
                    para_lat_hat[h_u] = temp_lat  # -------------------------隐节点与隐节点之间的条件概率矩阵

                    print("temp_lat: in 处理 flag==1 的情况")
                    print(temp_lat)
                    if is_root:
                        root_para[h_u] = temp_root_para  # -------------根节点的边际概率
                    para_ih[i_val] = temp_pa  # 存储 {i|h}
                else:  # -------------------------------------------如果Aijk_i中只有一个
                    if not (hasattr(mu_k_ch, '__len__') and len(mu_k_ch) == 1) and not np.isscalar(mu_k_ch):
                        raise ValueError("'Aijk_i' and 'mu_k' should have length 1!")

                    # k2的父节点是h
                    if pa.get(str(k2), None) == h:
                        temp_mat = para_hat[k2]

                        print("para_hat[k2]: k2的父节点是h")
                        print(para_hat[k2])
                        if k2 not in I:  # -----------------------pa[k2] == h时，k2不一定在I中
                            # print("where")
                            # print(type(k2))
                            # 判断k2是否是离散的
                            if model is not None:
                                if model.loc[k2, model.columns[1]] == "observable_continuous":
                                    lambda_val = temp_mat[0, :]
                                else:
                                    lambda_val = temp_mat[1, :]
                            elif data is not None:
                                if abs(data[0, int(k2) - 1] - int(
                                        data[0, int(k2) - 1])) > tol:  # 这里需要注意data的行与列假定是0-based，所以顶点的索引值-1

                                    lambda_val = temp_mat[0, :]
                                else:
                                    lambda_val = temp_mat[1, :]
                            else:
                                raise ValueError("Error input of 'data' and 'model'!")

                        elif node_type.get(k2, "") == "continuous":
                            print("where1")
                            lambda_val = temp_mat[0, :]
                        elif node_type.get(k2, "") == "discrete":
                            print("where2")
                            lambda_val = temp_mat[1, :]
                    else:
                        if para_ih is None or k2 not in para_ih:
                            raise ValueError("Missing 'k2' conditional parameters for hidden node.")
                        temp_mat = para_ih[k2]
                        print("I:", I)
                        if str(k2) not in [str(x) for x in I.values()]:
                            raise ValueError(f"The node {k2} should belong to 'I'!")
                        elif node_type.get(k2, "") == "continuous":
                            lambda_val = temp_mat[0, :]
                        elif node_type.get(k2, "") == "discrete":
                            lambda_val = temp_mat[1, :]

                    mu_i_ch_val = mu_i_ch["mu_i"] if mu_i_ch is not None else None
                    mu_i2_ch_val = mu_i_ch["mu_i2"] if mu_i_ch is not None else None

                    mu_i_pa_val = mu_i_pa["mu_i"] if mu_i_pa is not None else None
                    mu_i2_pa_val = mu_i_pa["mu_i2"] if mu_i_pa is not None else None

                    # print("mu_i_pa_val: ",mu_i_pa_val,"mu_i2_pa_val: ",mu_i2_pa_val,"lambda_val[0]: ",lambda_val[0],"lambda_val[1]: ",lambda_val[1])

                    temp_ch = para_esti(i=i_val, Aijk=Aijk_i_ch, mu_k=mu_k_ch, node_type=node_type[i_val],
                                        mu_i=mu_i_ch_val, mu_i2=mu_i2_ch_val,
                                        lambda1=None, lambda2=None, root=False)
                    temp_pa = para_esti(i=i_val, Aijk=Aijk_i_pa, mu_k=mu_k_pa, node_type=node_type[i_val],
                                        mu_i=mu_i_pa_val, mu_i2=mu_i2_pa_val,
                                        lambda1=lambda_val[0], lambda2=lambda_val[1], root=is_root)

                    # print("temp_ch n 如果Aijk_i中只有一个:")
                    # print(temp_ch)
                    # print("temp_pa n 如果Aijk_i中只有一个:")
                    # print(temp_pa)

                    if is_root:
                        root_para[h_u] = temp_pa["prob_of_root"]  # ----------------------根节点的边际概率
                        temp_pa = temp_pa["eigenvector"]
                        # root_para[h_u] = root_val  # Uncomment to save if needed
                    # P_{h_|h} = Γ^{-1}_{i|h_} %*% Γ_{i|h}; 其中h = pa(h_)
                    if np.linalg.matrix_rank(temp_ch) < 2 or np.linalg.cond(temp_ch) > 1e4:
                        temp_eps = 1 if np.min(np.abs(temp_ch)) < 1e-5 else np.min(np.abs(temp_ch))
                        temp_ch = temp_ch + np.diag([min(temp_eps, 0.1)] * 2)

                    # temp_lat = np.linalg.solve(temp_ch, temp_pa)#----------隐节点与隐节点之间的条件概率矩阵

                    # 带非负约束的最小二乘问题（Nonnegative Least Squares, NNLS）。
                    temp_lat = solve_nnls_2x2(temp_ch, temp_pa)

                    temp_lat[:, 0] /= np.sum(temp_lat[:, 0])
                    temp_lat[:, 1] /= np.sum(temp_lat[:, 1])
                    para_lat_hat[h_u] = temp_lat

                    print("temp_lat in 如果Aijk_i中只有一个:")
                    print(temp_lat)

    if is_root:
        result = {"para_lat_hat": para_lat_hat, "root_para": root_para}  # root_para 需在上述处理中赋值
    else:
        result = para_lat_hat

    gc.collect()
    return result


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
            candidate = order_idx[temp_iter + 1]
            print("candidate1: ", candidate)
            temp_iter += 1

        # print("candidate: ",candidate,"common: ",common)

        # 计算最终索引（0-based）
        i_sub = candidate % len(des_lat1)
        j_sub = candidate // len(des_lat1)
        a = [idx1_0[i_sub], idx2_0[j_sub]]  # 确保 0-based

        a_sorted = sorted(a)

        # print("a_sorted: ",a_sorted)

        return a_sorted


# 三叉变量(固定一个节点k, 选出与其对应的子节点)
# 对于输入的隐节点h与可观测节点k, 以k为一个三叉变量，选出h的与k相关的三叉变量.
# 输入: h, k, tree
# ------h: 字符串，隐节点, 如"H1"
# ------k: 字符串，可观测节点, 如"1"(与h绑定的k, 即假设的 (\mu_{k|h=0} > \mu_{k|h=1}) )
# ------tree: h,k所在的树结构
# 输出: (i,j,k)

def tri_nodes_obs1(h, k, tree):
    """
    针对输入的隐节点 h 与可观测节点 k，选出 h 对应的三叉变量 (i, j, k)。

    输入:
      h: str，隐节点名称（例如 "H1"），必须以 "H" 开头
      k: str，可观测节点名称（例如 "1"），代表与 h 绑定的 k 节点（假设 μ_{k|h=0} > μ_{k|h=1}）
      tree: dict，包含隐树结构，至少包含键：
              "obs_nodes", "lat_nodes", "pa", "child", "des_nodes", "anc_nodes", "D"

    输出:
      返回一个列表 [i, j, k]，其中 i 和 j 为选出的与 h 对应的两个可观测节点，
      k 保持不变。
    """
    # 检查 h 是否为隐节点（必须以 "H" 开头）
    if not h.startswith("H"):
        raise ValueError("Node 'h' should be latent!")
    print("tri_nodes_obs1(h, k, tree): ", h, k)

    obs_nodes = tree["obs_nodes"]
    lat_nodes = tree["lat_nodes"]
    pa = tree["pa"]
    st = tree["str"]
    ch = tree["child"]
    de = tree["des_nodes"]
    an = tree["anc_nodes"]
    D = tree.get("D", None)

    # 构造图 g：利用 pa 构造边集，去除父节点为空的行
    col1 = list(pa.keys())
    col2 = list(pa.values())
    # 去除父节点为空的边
    edge_list = [(u, v) for u, v in zip(col1, col2) if v != ""]
    # 使用 igraph 构造无向图
    g = ig.Graph.TupleList(edge_list, directed=False)

    # 如果 D 为空，则构造默认距离矩阵：
    if D is None:
        n_obs = len(obs_nodes)
        D_arr = np.ones((n_obs, n_obs))
        np.fill_diagonal(D_arr, 0)
        D = D_arr
    else:
        D = np.array(D)

    obs_idx = {v: i for i, v in enumerate(obs_nodes)}
    # 获取 h 的子节点
    child_h = ch.get(h, [])
    # I：h 的可观测子节点 = intersection(ch[h], obs_nodes)
    I = [node for node in child_h if node in obs_nodes]
    j = None
    # H：h 的隐子节点 = intersection(ch[h], lat_nodes)
    H = [node for node in child_h if node in lat_nodes]

    an_k = an.get(k, [])
    # 判断k是否是h的后代
    is_descendant = h in an_k
    if is_descendant:
        # print("k 是 h 的后代")
        # 判断k是 h的子节点 还是 h的隐子节点的后代，T: k是h的子节点; F: k是h的隐子节点的后代
        if k in I:
            # k是h的子节点
            # print("k是h的子节点")
            # I_rm = I 去掉 k
            I_rm = [x for x in I if x != k]
            if len(I_rm) >= 2:  # 如果h除了k，还有2个可观测子节点
                # print("如果h除了k，还有2个可观测子节点")
                d_ik = [(x, D[obs_idx[x], obs_idx[k]]) for x in I_rm]
                # 按值排序，取第一个作为 i，第二个作为 j
                sorted_nodes = [x for x, _ in sorted(d_ik, key=lambda x: x[1])]
                i_sel, j_sel = sorted_nodes[0], sorted_nodes[1]
            elif len(I_rm) == 1:  # 如果h除了k，只有1个可观测子节点
                print("如果h除了k，只有1个可观测子节点")
                i_sel = I_rm[0]
                if len(H) >= 1:  # 如果h有隐子节点,从h的后代中选出j
                    J = [x for x in set(de.get(h, [])) & set(obs_nodes) if x not in I]
                    if not J:
                        raise ValueError("No candidate found for j!")
                    d_jk = [(x, D[obs_idx[x], obs_idx[k]]) for x in J]
                    j_sel = min(d_jk, key=lambda x: x[1])[0]
                elif len(H) == 0:  # 如果h没有隐子节点,则从非h后代的可观测中选出j
                    J = [x for x in obs_nodes if x not in I]
                    if not J:
                        raise ValueError("No candidate found for j!")
                    d_jk = [(x, D[obs_idx[x], obs_idx[k]]) for x in J]
                    j_sel = min(d_jk, key=lambda x: x[1])[0]
            elif len(I_rm) == 0:  # 如果h除了k，无可观测节点
                print("如果h除了k，无可观测节点")
                if len(H) >= 2:
                    # 如果h有2个隐子节点,从H[1,2]的后代中分别选出i,j
                    II = [x for x in set(de.get(H[0], [])) & set(obs_nodes)]
                    JJ = [x for x in set(de.get(H[1], [])) & set(obs_nodes)]
                    if not II or not JJ:
                        raise ValueError("No candidate for i or j!")
                    d_ik = [(x, D[obs_idx[x], obs_idx[k]]) for x in II]
                    i_sel = min(d_ik, key=lambda x: x[1])[0]
                    d_jk = [(x, D[obs_idx[x], obs_idx[k]]) for x in JJ]
                    j_sel = min(d_jk, key=lambda x: x[1])[0]
                elif len(H) == 1:  # 如果h有1隐子节点,从H的后代中选出i,非h后代的叶子中选出j
                    if pa.get(h, "") == "":
                        raise ValueError("The root node 'h' should have 3 child at least!")
                    II = [x for x in set(de.get(H[0], [])) & set(obs_nodes)]
                    JJ = [x for x in obs_nodes if x not in de.get(h, [])]
                    if not II or not JJ:
                        raise ValueError("No candidate for i or j!")
                    d_ik = [(x, D[obs_idx[x], obs_idx[k]]) for x in II]
                    d_jk = [(x, D[obs_idx[x], obs_idx[k]]) for x in JJ]
                    i_sel = min(d_ik, key=lambda x: x[1])[0]
                    j_sel = min(d_jk, key=lambda x: x[1])[0]
                else:
                    raise ValueError("The latent node 'h' should have two child at least!")
            # 结束 k 在 I 分支
        else:  # k是h的隐子节点的后代
            print("k是h的隐子节点的后代")
            h2 = [x for x in an.get(k, []) if x in H]  # h2是k的祖先(h2是h的子节点)
            H_rm = [x for x in H if x not in h2]
            if len(I) >= 2:  # 如果h有2个可观测子节点
                d_ik = [(x, D[obs_idx[x], obs_idx[k]]) for x in I]
                sorted_nodes = [x for x, _ in sorted(d_ik, key=lambda x: x[1])]
                i_sel, j_sel = sorted_nodes[0], sorted_nodes[1]
            elif len(I) == 1:  # 如果h只有1个可观测子节点
                i_sel = I[0]
                if len(H_rm) >= 1:  # 如果h除了h2，至少有1个可观测子节点
                    # h2的后代与h的可观测子节点
                    J = [x for x in set(de.get(h, [])) & set(obs_nodes) if x not in I]
                    if not J:
                        raise ValueError("No candidate for j!")
                    d_jk = [(x, D[obs_idx[x], obs_idx[k]]) for x in J]
                    j_sel = min(d_jk, key=lambda x: x[1])[0]
                else:  # 如果h只有h2一个隐节点
                    if pa.get(h, "") == "":
                        raise ValueError("The root node 'h' should have 3 child at least!")
                    J = [x for x in obs_nodes if x not in de.get(h, [])]
                    if not J:
                        raise ValueError("No candidate for j!")
                    d_jk = [(x, D[obs_idx[x], obs_idx[k]]) for x in J]
                    j_sel = min(d_jk, key=lambda x: x[1])[0]
            elif len(I) == 0:  # 如果h无可观测节点
                if len(H_rm) >= 2:  # 如果h有多于3个隐子节点,从H_rm[1,2]的后代中分别选出i,j
                    II = [x for x in set(de.get(H_rm[0], [])) & set(obs_nodes)]
                    JJ = [x for x in set(de.get(H_rm[1], [])) & set(obs_nodes)]
                    if not II or not JJ:
                        raise ValueError("No candidate for i or j!")
                    d_ik = [(x, D[obs_idx[x], obs_idx[k]]) for x in II]
                    d_jk = [(x, D[obs_idx[x], obs_idx[k]]) for x in JJ]
                    i_sel = min(d_ik, key=lambda x: x[1])[0]
                    j_sel = min(d_jk, key=lambda x: x[1])[0]
                elif len(H_rm) == 1:  # 如果h有2隐子节点,从H_rm的后代中选出i,非h后代的叶子中选出j
                    if pa.get(h, "") == "":
                        raise ValueError("The root node 'h' should have 3 child at least!")
                    II = [x for x in set(de.get(H_rm[0], [])) & set(obs_nodes)]
                    JJ = [x for x in obs_nodes if x not in de.get(h, [])]
                    if not II or not JJ:
                        raise ValueError("No candidate for i or j!")
                    d_ik = [(x, D[obs_idx[x], obs_idx[k]]) for x in II]
                    d_jk = [(x, D[obs_idx[x], obs_idx[k]]) for x in JJ]
                    i_sel = min(d_ik, key=lambda x: x[1])[0]
                    j_sel = min(d_jk, key=lambda x: x[1])[0]
                else:
                    raise ValueError("The latent node 'h' should have two child at least!")
    else:
        # 如果 k 不是 h 的后代，则调用 bifur 函数，从 h 的后代中选出两个节点构成二叉变量
        # 从h的后代中选出二叉变量即可
        print("k 不是 h 的后代")
        ij = bifur(u=h, D=D, tree=tree)
        i_sel = ij[0] + 1  # bifur返回的是0-based,再加1处理一下
        j_sel = ij[1] + 1
        # 转换为字符串
        i_sel = str(i_sel)
        j_sel = str(j_sel)

    # 返回三叉变量：[i, j, k]
    print("i_sel, j_sel, k: ", i_sel, j_sel, k)
    return [i_sel, j_sel, k]


def mse(a, b):
    """Compute sum of absolute differences between arrays a and b."""
    return np.sum(np.abs(a - b))


# 损失函数MSE
# 真实参数与估计参数的MSE
# 输入: para, para_hat, model, tree
# ------para: 真实参数列表
# ------para_hat: 估计参数列表
# --注: 真实参数列表 与 估计参数列表 参数组数必须完全对应
# ------model: 模型
# ------tree: 树结构
# 输出: 真实参数与估计参数的MSE
import copy


def loss_fun(para, para_hat, model, tree=None):
    """
    Compute the mean squared error (MSE) between true parameters (para) and
    estimated parameters (para_hat) over all nodes.

    Parameters:
      para: dict of true parameters.
      para_hat: dict of estimated parameters.
      model: pandas DataFrame; first column: node names, second column: node type.
      tree: dictionary representing the tree structure. If None, it is built externally.

    Returns:
      result_loss: the overall MSE (a scalar)
    """
    # If tree is None, build it (here we assume build_tree is defined elsewhere)
    if tree is None:
        tree = tree_from_edge(model.drop(model.columns[1], axis=1))  # adjust as needed

    # print("loss_fun(para, para_hat, model, tree=None): ")
    # print("tree:")
    # print(tree)

    obs_nodes = tree["obs_nodes"]
    lat_nodes = tree["lat_nodes"]
    pa = tree["pa"]
    # st = tree["str"]
    # ch = tree["child"]
    # de = tree["des_nodes"]
    # D = tree["D"]
    # Assume there is exactly one root in latent nodes
    root = [node for node in lat_nodes if pa.get(node, "") == ""][0]

    # Get discrete and continuous observable nodes from model:
    disc_nodes = model.loc[model.iloc[:, 1] == "observable_discrete", model.columns[0]].tolist()
    cont_nodes = model.loc[model.iloc[:, 1] == "observable_continuous", model.columns[0]].tolist()

    # Determine p: the number of parameter groups from the root node estimate.
    # We assume para_hat[root] is either a NumPy array (if p==1) or a list of arrays.
    # 一进来就给 para_hat 做个深拷贝
    para_hat = copy.deepcopy(para_hat)
    if isinstance(para_hat[root], list):
        p_val = len(para_hat[root])
    else:
        p_val = 1

    # print("para_hat[root]: ",para_hat[root],"p_val: ",p_val)

    # 连续变量估计矩阵的标准化
    for i in cont_nodes:
        ph = para_hat[i]
        if p_val == 1:
            # 如果 para_hat_i 是列表，则提取其第一个元素
            if isinstance(ph, list):
                mat = ph[0]
            else:
                mat = ph
            # 第二行原本表示的是方差
            if mat[1, 0] < 0 or mat[1, 1] < 0:
                warnings.warn(f"The estimation of {i} is error!")
            # 计算标准差.
            mat[1, :] = np.sqrt(np.where(mat[1, :] >= 0, mat[1, :], 0))
            para_hat[i] = mat
        else:
            for l in range(p_val):
                mat = ph[l]
                if mat[1, 0] < 0 or mat[1, 1] < 0:
                    warnings.warn(f"The estimation of {i} is error!")
                mat[1, :] = np.sqrt(np.where(mat[1, :] >= 0, mat[1, :], 0))
                ph[l] = mat
            para_hat[i] = ph

    # print("para_hat: ",para_hat)

    # 离散变量估计矩阵的标准化，总结来说，这段标准化代码就是在对离散变量的参数估计进行“裁剪”，
    # 使得估计的概率值不超出合法的 [0,1]区间，同时对异常情况（比如负值）发出警告。

    for i in disc_nodes:
        ph = para_hat[i]
        if p_val == 1:
            if isinstance(ph, list):
                mat = ph[0]
            else:
                mat = ph
            if np.min(mat) <= 0:
                warnings.warn(f"The estimation of {i} is error!")
            # 可观测离散变量的条件概率矩阵保留第二行，第二行的意义在于离散变量取值为1，而离散变量生成时，对应的条件概率真值保留的也是离散变量取值为1，对应的概率,
            # setting negatives to 0 and values >1 to 1.
            mat = np.where(mat[1, :] < 0, 0, mat[1, :])
            mat = np.where(mat > 1, 1, mat)
            para_hat[i] = mat
        else:
            for l in range(p_val):
                mat = ph[l]
                if np.min(mat) <= 0:
                    warnings.warn(f"The estimation of {i} is error!")
                mat = np.where(mat[1, :] < 0, 0, mat[1, :])
                mat = np.where(mat > 1, 1, mat)
                ph[l] = mat
            para_hat[i] = ph

    # 隐变量估计矩阵的标准化,这段代码对隐变量（latent nodes）的参数估计进行了一种“裁剪”（clipping）标准化。
    # 也就是说，它确保所有估计的参数值都严格落在 (0,1) 的范围内。如果参数值低于 0，就设置为 0；如果参数值高于 1，就设置为 1。
    for h in lat_nodes:
        if h == root:  # -----------------------------如果h是根节点
            if p_val == 1:
                if para_hat[h] <= 0:
                    warnings.warn(f"The estimation of {h} is error!")
                    para_hat[h] = 0
                elif para_hat[h] >= 1:
                    warnings.warn(f"The estimation of {h} is error!")
                    para_hat[h] = 1
            else:
                val = para_hat[h]
                for l in range(p_val):
                    if val[l] <= 0:
                        warnings.warn(f"The estimation of {h} is error!")
                        val[l] = 0
                    elif val[l] >= 1:
                        warnings.warn(f"The estimation of {h} is error!")
                        val[l] = 1
                para_hat[h] = val
        else:
            if p_val == 1:
                vec = para_hat[h]
                if isinstance(vec, list):
                    vec = vec[0]
                vec = np.where(vec <= 0, 0, vec)
                vec = np.where(vec >= 1, 1, vec)
                para_hat[h] = vec
            else:
                vec = para_hat[h]
                for l in range(p_val):
                    vec[l] = np.where(vec[l] <= 0, 0, vec[l])
                    vec[l] = np.where(vec[l] >= 1, 1, vec[l])
                para_hat[h] = vec

    # print("para_hat: ",para_hat)

    # 连续变量估计值与真值的匹配，这段代码实现了对连续变量参数估计值的“对齐匹配”（matching/alignment），以解决可能的隐状态交换问题。
    for i in cont_nodes:
        ph = para_hat[i]
        true_val = para[i]
        if p_val == 1:
            # Convert the row vector to a 1D array. In R: as.numeric(t(...))
            mat1 = np.array(ph).flatten()
            mat2 = np.array(true_val).flatten()
            # Also compute reversed order: R: t(para_hat_i[,2:1])
            if ph.ndim == 2 and ph.shape[1] >= 2:
                mat1_rev = np.array(ph[:, ::-1]).flatten()
            else:
                mat1_rev = mat1
            if mse(mat1, mat2) <= mse(mat1_rev, mat2):
                temp = mat1
            else:
                temp = mat1_rev
        else:
            temp = np.zeros((4, p_val))
            for l in range(p_val):
                mat = ph[l]
                mat1 = np.array(mat).flatten()
                col_true = np.array(true_val)[:, l].flatten()
                mat1_rev = np.array(mat[:, ::-1]).flatten()
                if mse(mat1, col_true) <= mse(mat1_rev, col_true):
                    temp[:, l] = mat1
                else:
                    temp[:, l] = mat1_rev
        para_hat[i] = temp

    for i in disc_nodes:
        para_hat_i = para_hat[i]
        para_i = para[i]
        if p_val == 1:
            vec1 = np.array(para_hat_i).flatten()

            vec1_rev = vec1[::-1]  # 反转顺序

            # print("vec1: ",vec1,"vec1_rev: ",vec1_rev)
            vec2 = np.array(para_i).flatten()
            if mse(vec1, vec2) <= mse(vec1_rev, vec2):
                temp = vec1
            else:
                temp = vec1_rev
        else:
            temp = np.zeros((2, p_val))
            for l in range(p_val):
                vec = para_hat_i[l]
                vec1 = np.array(vec)
                vec1_rev = vec1[::-1]
                vec2 = np.array(para_i)[:, l]
                if mse(vec1, vec2) <= mse(vec1_rev, vec2):
                    temp[:, l] = vec1
                else:
                    temp[:, l] = vec1_rev
        para_hat[i] = temp

    # 隐变量(与根节点)估计值与真值的匹配
    for h in lat_nodes:
        ph = para_hat[h]
        true_val = para[h]
        if h == root:  # -----------------------------如果h是根节点
            if p_val == 1:
                val1 = np.array(ph).flatten()
                vec2 = np.array(true_val).flatten()
                val1_rev = 1 - val1
                if mse(val1, vec2) <= mse(val1_rev, vec2):
                    temp = val1
                else:
                    temp = val1_rev
            else:
                temp = np.zeros(p_val)
                for l in range(p_val):
                    val = ph[l]
                    val1 = np.array(val)
                    val1_rev = 1 - val1
                    if mse(val1, true_val[l]) <= mse(val1_rev, true_val[l]):
                        temp[l] = val1
                    else:
                        temp[l] = val1_rev
            para_hat[h] = temp
        else:
            if p_val == 1:
                mat = np.array(ph)
                vec1 = mat[0, :]
                vec1_rev = mat[1, :]
                # For a 1×n matrix, we simulate reversal by slicing in reverse order
                vec1_alt = mat[0, ::-1]
                vec1_alt2 = mat[1, ::-1]
                vec2 = np.array(true_val).flatten()
                # Compute MSE for four variants:
                mse_values = [
                    mse(vec1, vec2),
                    mse(vec1_rev, vec2),
                    mse(vec1_alt, vec2),
                    mse(vec1_alt2, vec2)
                ]
                # Select the variant with minimum mse:
                if mse_values[0] <= min(mse_values[1:]):
                    temp = vec1
                elif mse_values[1] <= min(mse_values[0], mse_values[2], mse_values[3]):
                    temp = vec1_rev
                elif mse_values[2] <= min(mse_values[0], mse_values[1], mse_values[3]):
                    temp = vec1_alt
                else:
                    temp = vec1_alt2
            else:
                temp = np.zeros((2, p_val))
                for l in range(p_val):
                    mat = para_hat[h][l]
                    vec1 = mat[0, :]
                    vec1_rev = mat[1, :][::-1]
                    vec1_alt = mat[0, ::-1]
                    vec1_alt2 = mat[1, ::-1]
                    mse_values = [
                        mse(vec1, true_val[:, l]),
                        mse(mat[1, :], true_val[:, l]),
                        mse(vec1_alt, true_val[:, l]),
                        mse(vec1_alt2, true_val[:, l])
                    ]
                    if mse_values[0] <= min(mse_values[1:]):
                        temp[:, l] = vec1
                    elif mse_values[1] <= min(mse_values[0], mse_values[2], mse_values[3]):
                        temp[:, l] = mat[1, :]
                    elif mse_values[2] <= min(mse_values[0], mse_values[1], mse_values[3]):
                        temp[:, l] = vec1_alt
                    else:
                        temp[:, l] = vec1_alt2
            para_hat[h] = temp

    # print("para_hat: ")
    # print(para_hat)

    # Compute overall loss (MSE) between true parameters and estimates.
    if p_val == 1:
        l2 = 0
        num = 0
        # Combine obs_nodes and latent nodes
        all_nodes = obs_nodes + lat_nodes
        # print("all_nodes: ",all_nodes)
        for i in all_nodes:
            # print("i: ",i)
            para_hat_i = para_hat[i]
            para_i = para[i]
            vec1 = np.array(para_hat_i).flatten()
            vec2 = np.array(para_i).flatten()
            if i == root:
                num += 1
                temp_val = np.sum(np.abs(vec1 - vec2))
                l2 += temp_val
                # print("i: ",i,"temp_val: ",temp_val)
            elif (i in disc_nodes) or (i in lat_nodes):
                num += 2
                temp_val = np.sum(np.abs(vec1 - vec2))
                l2 += temp_val
                # print("i: ",i,"temp_val: ",temp_val)
            elif i in cont_nodes:
                num += 4
                temp_val = np.sum(np.abs(vec1 - vec2))
                l2 += temp_val
                # print("i: ",i,"temp_val: ",temp_val)
            else:
                raise ValueError("Unknown nodes!")
        result_loss = l2 / num
    else:
        l2 = np.zeros(p_val)
        num = 0
        for l in range(p_val):
            for i in (obs_nodes + lat_nodes):
                if i == root:
                    num += 1
                    temp_val = np.sum(np.abs(para_hat[i][l] - para[i][l]))
                    l2[l] += temp_val
                elif (i in disc_nodes) or (i in lat_nodes):
                    num += 2
                    temp_val = np.sum(np.abs(para_hat[i][:, l] - para[i][:, l]))
                    l2[l] += temp_val
                elif i in cont_nodes:
                    num += 4
                    temp_val = np.sum(np.abs(para_hat[i][:, l] - para[i][:, l]))
                    l2[l] += temp_val
                else:
                    raise ValueError("Unknown nodes!")
        result_loss = np.sum(l2) / num

    gc.collect()
    return result_loss


def PEMLT_online(data_obs, D, model,  EXYZ11_online,EXYZ12_online,EXYZ21_online,EXYZ22_online,EXY11_online, EXY12_online,EXY21_online,EXY22_online, EX2_online, EX_online,
                 EXYZ_update, EXY_update, EX_update, tol=0.0001):
    """
    根据观测数据和模型设定进行参数估计。

    参数：
        data_obs: numpy 数组，观测数据矩阵
        model: pandas DataFrame，模型设定（各节点关系等）
        tol: 容差，用于判断连续变量（默认 0.0001）

    返回：
        para_hat: dict，包含各节点的参数估计结果
    """
    # --- 1. 计算信息距离 ---
    # D = information_distance(data_obs)

    # --- 2. 对 data_obs 进行部分中心化 ---
    # 判断哪些变量为连续型：若首个数据与其整数部分差值大于 tol，则为连续变量
    num_cols = data_obs.shape[1]
    logi = [abs(data_obs[0, u] - int(data_obs[0, u])) > tol for u in
            range(num_cols)]  # -------------------------T: 连续; F: 离散

    print("logi: ", logi)

    # cont_vex 保存连续变量的索引
    cont_vex = [u for u, flag in enumerate(logi) if flag]
    # 计算各列均值
    mu_hat = np.mean(data_obs, axis=0)
    # 计算每列的方差（样本方差，ddof=1）
    var_hat = np.var(data_obs, axis=0, ddof=1)
    # 对于离散变量，将均值置为 0
    mu_hat = np.array([m if flag else 0 for m, flag in zip(mu_hat, logi)])
    # 对每一列做中心化处理
    data_obs = data_obs - mu_hat

    # --- 3. 构造图与树结构 ---
    # R 中：g <- graph_from_data_frame(model[model[,3] != "", -2], directed = F)
    # 假定 model 为 DataFrame，选取第三列不为空的行，并删除第二列
    model_subset = model[model.iloc[:, 2] != ""].drop(model.columns[1], axis=1)
    # 转为列表构造图（这里要求 model_subset 每行记录一条边）
    g = ig.Graph.TupleList(model_subset.values.tolist(), directed=False)

    # 构造树结构：删除模型中的第二列（与 R 中 model[,-2] 等价）
    model_tree = model.drop(model.columns[1], axis=1)
    tree = tree_from_edge(model_tree)  # 假定 tree_from_edge 返回一个 dict
    tree["D"] = D

    # 从树中提取相关信息
    obs_nodes = tree['obs_nodes']  # 观测节点
    lat_nodes = tree['lat_nodes']  # 潜变量节点
    pa = tree['pa']  # 父节点关系（dict）
    st = tree['str']  # 结构信息，假定 st[node] 是一个列表，第四个元素对应 R 中 st[h,4]
    ch = tree['child']  # 子节点关系（dict）
    de = tree['des_nodes']  # 后代节点
    an = tree['anc_nodes']  # 祖先节点
    # 确定根节点：R 中 root <- model[model[,3] == "", 1]
    root = model.loc[model.iloc[:, 2] == "", model.columns[0]].values[0]  # ------------------------根节点
    child = ch[root]  # --------------------------------------根节点的子节点

    # --- 4. 确定节点层次顺序（step） ---
    # 从根节点开始，递归添加潜变量子节点
    step = [root]  # ------------------------------越靠前，节点的层数越高

    while True:
        children = []
        for node in step:
            if node in ch:
                # 仅保留属于潜变量的子节点
                children.extend([c for c in ch[node] if c in lat_nodes])
        # 去除已在 step 中的节点
        step_new = [n for n in children if n not in step]
        if not step_new:
            break
        step.extend(step_new)

    print("step: ", step)

    # --- 5. 初始化参数估计相关变量 ---
    # key_node: 存储每个潜变量关键节点信息（初始值为空字符串）
    key_node = {node: "" for node in lat_nodes}
    # Ijk 与 flag：分别为每个潜变量节点存储辅助矩阵及标志
    Ijk = {node: None for node in lat_nodes}
    flag = {node: None for node in lat_nodes}
    # para_hat: 存储所有节点（观测和潜变量）的参数估计结果，初始为空
    all_nodes = list(obs_nodes) + list(lat_nodes)
    para_hat = {node: None for node in all_nodes}

    # --- 6. 针对具有观测子节点的节点进行参数估计 ---
    # 依层次自底向上遍历（对 step 反向遍历）
    for h in reversed(step):  # ---------从底层到高层
        # print("why:")
        print("h:", h)

        temp = tri_nodes_obs(h, tree)  # 获取与 h 相关的三元组信息，返回值return {'I': Ijk_mat, 'j': j_node, 'k': key_node}

        print("temp:", temp)
        # print("type(temp):", type(temp))

        key_node[h] = temp.get('k', "") if temp is not None else ""
        # 修复 ch_obs_num 取法（如果 st 是 pandas DataFrame）
        ch_obs_num = st.loc[h, st.columns[3]] if h in st.index else 0
        if ch_obs_num == 0:  # ------------------如果h没有可观测子节点，则跳过
            continue

        temp1 = tri_nodes_stand(temp)  # 标准化处理
        print("temp1:", temp1)

        Ijk[h] = temp1['tri_nodes']
        flag[h] = temp1['key_node_type']

        print("Ijk[h]: ", Ijk[h], "flag[h]: ", flag[h])
        # 观测节点参数估计,Ijk[h]进入函数后主要是利用父节点不调用D,所以Ijk[h]不需要是0-based的整数，使用字符串格式的节点名称即可

        para_hat_= para_esti_obs(EXYZ11_online,EXYZ12_online,EXYZ21_online,EXYZ22_online,EXY11_online, EXY12_online,EXY21_online,EXY22_online, EX2_online, EX_online,
                                 EXYZ_update, EXY_update, EX_update, ijk=Ijk[h], h=h, flag=flag[h], tree=tree, data=data_obs, g=g)                                  

        print("para_hat_: ")
        print(para_hat_)

        if pa[h] == "":  # h 是根节点
            # 从 para_hat_ 中一次性拿出所有 prob_of_root
            prob_list = [
                node_info['prob_of_root']
                for node_info in para_hat_.values()
                if isinstance(node_info, dict) and 'prob_of_root' in node_info
            ]
            # 求平均
            if prob_list:
                root_para_mean = sum(prob_list) / len(prob_list)
            else:
                root_para_mean = None

            para_hat[h] = root_para_mean
            # print("para_hat[h]: ",para_hat[h])
            # 如果h是根的话，para_esti_obs返回的是另外一种带有'prob_of_root'的数据结构，para_hat采用特殊的赋值方式
            for j, info in para_hat_.items():
                eig = info['eigenvector']
                # 把 eigenvector 赋给 para_hat[j]
                para_hat[j] = eig
        else:
            # 如果h不是根的话，para_esti_obs返回的是另外一种不带有'prob_of_root'的数据结构，para_hat采用正常赋值方式
            # 将 para_hat_ 中的每个节点的估计结果存入 para_hat（如已存在则输出警告）
            for j, value in para_hat_.items():
                if para_hat.get(j) is not None:
                    print(f"Warning: The node {j} have had estimation!")
                para_hat[j] = value

    # --- 7. 针对具有潜变量子节点的节点进行参数估计 ---
    for h in reversed(step):  # ---------从底层到高层

        print("h: 隐变量之间", h)
        print("key_node: ", key_node)
        # 计算 h 的子节点中有多少为潜变量
        if h in ch:
            ch_lat = [c for c in ch[h] if c in lat_nodes]
        else:
            ch_lat = []
        if len(ch_lat) == 0:  # ------------------如果h没有隐子节点，则跳过
            continue
        temp = tri_nodes_lat(h1=h, key_node=key_node,
                             tree=tree)  # return {"tri_nodes": ijkjk, "key_node": key_node, "key_node_type": flag}

        print("temp: ", temp)

        # print("para_hat:")
        # print(para_hat)

        Ijk[h] = temp["tri_nodes"]
        key_node.update(temp["key_node"])
        flag[h] = temp["key_node_type"]

        # 返回的是h作为父节点，给定h后，h的各个子节点的条件概率
        para_hat_= para_esti_lat(EXYZ11_online,EXYZ12_online,EXYZ21_online,EXYZ22_online,EXY11_online, EXY12_online,EXY21_online,EXY22_online, EX2_online, EX_online,
                                 EXYZ_update, EXY_update, EX_update,ijk=Ijk[h], h=h, flag=flag[h], para_hat=para_hat, tree=tree, data=data_obs, g=g)
        

        print("para_hat_ in 隐变量之间")
        print(para_hat_)
        if pa[h] == "":  # ---------------------------即h是根，para_hat_中含有root_para
            root_para = para_hat_.get("root_para", None)
            print("root_para: ", root_para)
            if root_para is not None:
                root_para_mean = sum(root_para.values()) / len(root_para)  # 若 h 为根节点，对其每个隐子节点对应的边际概率参数求平均
            else:
                root_para_mean = None
            para_hat[h] = root_para_mean
            para_hat_ = para_hat_.get("para_lat_hat", None)
        for h_, value in para_hat_.items():
            if para_hat.get(h_) is not None:
                print(f"Warning: The node {h_} have had estimation!")
            para_hat[h_] = value

    # print("隐变量之间估计结束： ")
    # print(para_hat)

    # --- 8. 对连续变量的参数估计进行后处理 ---
    # 对于每个连续变量（索引在 cont_vex 中），调整估计矩阵以恢复原均值并修正二阶矩
    # 对连续变量的均值估计进行处理（加回中心化时减去的均值）注：对二阶矩也有影响
    obs_nodes = tree["obs_nodes"]

    for u in cont_vex:
        node_name = obs_nodes[u]
        temp_arr = para_hat.get(node_name)

        if temp_arr is None:
            print(f"Warning: para_hat[{node_name}] is None, skipped.")
            continue

        cond_var = temp_arr[1, 0] - temp_arr[0, 0] ** 2
        temp_arr[1, 0] = temp_arr[1, 0] if cond_var < 0 else cond_var  # 条件方差的估计<0的话,将条件方差设置为temp_arr[1, 0]
        cond_var = temp_arr[1, 1] - temp_arr[0, 1] ** 2
        temp_arr[1, 1] = temp_arr[1, 1] if cond_var < 0 else cond_var  # 条件方差的估计<0的话,将条件方差设置为temp_arr[1, 1]

        temp_arr[0, 0] += mu_hat[u]
        temp_arr[0, 1] += mu_hat[u]

        # #条件方差的特殊处理，把temp_arr[1, 0]和temp_arr[1, 1]放到和var_hat同等尺度
        # if abs(temp_arr[1, 0]-var_hat[u])/var_hat[u]<var_hat[u]>0.5:
        #     temp_arr[1, 0]=var_hat[u]
        # if abs(temp_arr[1, 1]-var_hat[u])/var_hat[u]<var_hat[u]>0.5:
        #     temp_arr[1, 1]=var_hat[u]

        para_hat[node_name] = temp_arr

    # 删除 data_obs（Python 自动回收内存，不强制调用 gc）
    del data_obs
    return para_hat


def adjust_para_hat(para_hat, model):
    """
    根据 model 中的节点信息调整 para_hat 参数：
      - 对于 observable_discrete 节点（离散观测变量），若参数为 2×2 矩阵，则只保留参数矩阵的第二行；
      - 对于 observable_continuous 节点（连续观测变量），保留完整的 2×2 矩阵；
      - 对于 latent 节点：
            - 若该节点为根节点（即在 model 中其 father_vertice 为空字符串），则直接保留原参数；
            - 否则（非根隐变量），若参数为 2×2 矩阵，则只保留参数矩阵的第二行。

    参数：
      para_hat: dict，原始参数估计结果，其中键为节点名称，值通常为 2×2 numpy 数组（对于非根节点）或其他形式（例如根节点可能为标量）
      model: pandas DataFrame，至少包含三列："vertice", "vertice_type", "father_vertice"

    返回：
      new_para_hat: dict，调整后的参数估计结果
    """
    # 构造节点到类型及父节点的字典
    type_dict = {}
    father_dict = {}
    for _, row in model.iterrows():
        node = str(row['vertice'])
        type_dict[node] = row['vertice_type']
        father_dict[node] = str(row['father_vertice']) if pd.notnull(row['father_vertice']) else ""
    # print(type_dict)
    new_para_hat = {}
    # print(para_hat["1"])
    for key, value in para_hat.items():
        # print("key: ",key,"value: ",value)
        # 如果该节点为根节点（父节点为空），直接保留原值
        if father_dict[key] == "":
            new_para_hat[key] = np.array([value])
        else:
            # 非根节点：根据节点类型调整
            if type_dict[key] in ("observable_discrete", "latent"):
                new_para_hat[key] = value[1:2, :].T
            elif type_dict[key] == "observable_continuous":
                # 假设 value.shape == (2,2),按行拉伸成4*1的array
                new_para_hat[key] = value.reshape(-1, 1)
    # print("new_para_hat: ",new_para_hat["1"])
    return new_para_hat


from collections import deque


def renumber_latents_df(model: pd.DataFrame) -> pd.DataFrame:
    """
    对 model 中 vertice_type=="latent" 的节点做层次重编号，并返回新的 DataFrame：
      - 根 latent 编号为 H1，依次向下做 BFS 编号
      - 更新 vertice 列和 father_vertice 列中的隐变量名称
      - 保留 observable 节点的原名和原顺序
    """
    # 1) 建立父→子索引
    children = {}
    for _, row in model.iterrows():
        v, father = row['vertice'], row['father_vertice']
        children.setdefault(father, []).append(v)

    # 2) 筛出所有 latent 节点
    latent_nodes = set(model.loc[model['vertice_type'] == 'latent', 'vertice'])

    # 3) 找根 latent（father_vertice==""）
    roots = [v for v in latent_nodes
             if model.loc[model['vertice'] == v, 'father_vertice'].iloc[0] == ""]
    if len(roots) != 1:
        raise ValueError(f"期望 1 个根 latent，发现 {len(roots)} 个：{roots}")

    # 4) BFS 给 latent 重编号
    mapping = {}
    queue = deque(roots)
    counter = 1
    while queue:
        cur = queue.popleft()
        mapping[cur] = f"H{counter}"
        counter += 1
        # 把所有 latent 子节点入队
        for child in children.get(cur, []):
            if child in latent_nodes:
                queue.append(child)

    # 5) 生成新的 DataFrame 副本并替换名称
    new_model = model.copy()
    # vertice 列
    new_model['vertice'] = new_model['vertice'].apply(lambda v: mapping.get(v, v))
    # father_vertice 列（空串保留）
    new_model['father_vertice'] = new_model['father_vertice'].apply(
        lambda f: mapping.get(f, f) if f != "" else ""
    )
    new_model.index = new_model['vertice']
    new_model.index.name = None

    return new_model