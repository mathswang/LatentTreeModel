#Belief Propagation functions on tree
import numpy as np
from collections import defaultdict, OrderedDict
import math

params = {
  "H1": {
    "states": [0,1,2],              # 隐节点 H1 的可能取值
    "prior": np.array([0.2,0.5,0.3]) # P(H1)
  },
  "X3": {
    "states": [0,1],                # 观测节点 X3（二值离散示例）
    "cpt": np.array([[0.8,0.2],     # P(X3=0|H?=0),P(X3=1|H?=0)
                     [0.4,0.6],     # P(X3=0|H?=1),P(X3=1|H?=1)
                     [0.1,0.9]])    # P(X3=0|H?=2),P(X3=1|H?=2)
    # 注意这里 cpt.shape=(n_parent_states, n_child_states)
  },
  
}


def build_tree(df):
    """
    从 df 构建：
      - children: OrderedDict，键为父节点（按数字降序），值为排好序的子节点列表
      - parent: dict，键为所有节点，值为父节点（根节点对应 ''）
    排序规则：
      * children[p] 列表中：先纯数字（升序），后 H+数字（降序）
      * children 的键（父节点）按数字部分降序
    """
    # 1) 收集结构
    children = defaultdict(list)
    parent   = {}
    for _, row in df.iterrows():
        node = row['vertice']
        p    = row['father_vertice']
        if p:
            parent[node] = p
            children[p].append(node)

    # 2) 确保每个节点都有 entry，根节点 parent 记为 ''
    for node in df['vertice']:
        parent.setdefault(node, '')

    # 3) 提取数字部分函数
    def _num(x):
        return int(x[1:]) if x.startswith("H") else int(x)

    # 4) 对每个父节点的孩子列表排序
    for p, ch in children.items():
        # 拆分纯数字节点和 H+数字节点
        num_nodes = [x for x in ch if x.isdigit()]
        h_nodes   = [x for x in ch if not x.isdigit()]
        # 纯数字升序，H节点降序
        num_nodes_sorted = sorted(num_nodes, key=int)
        h_nodes_sorted   = sorted(h_nodes,   key=_num, reverse=True)
        children[p] = num_nodes_sorted + h_nodes_sorted

    # 5) 整体按父节点名数字部分降序，构造 OrderedDict
    children_ordered = OrderedDict(
        sorted(children.items(),
               key=lambda kv: _num(kv[0]),
               reverse=True)
    )

    return children_ordered, parent


# 信息上传时,node是当前要考虑的节点，node只有一个父节点和对应的传给这个父节点的信息。
# upward_messages(node, children, parent, params, evidence, up_memo)执行前up_memo[node]是没有值的
# upward_messages的返回值是node传给父节点的信息up_memo[node]。
def upward_messages(node, children, parent, params, evidence, up_memo):
    """
    对树中单个节点递归计算上行消息（message to parent）。

    Args:
        node (str): 当前节点名。
        children (dict): parent -> [child1, child2, …] 的映射。
        parent (dict): node -> parent 的映射，根节点的 parent[node] == ''。
        params (dict): 每个节点的参数字典。连续观测有 'mean','var'，
                       离散有 'prior' 或 'cpt'，并且都含 'states'。
        evidence (dict): 叶节点的观测值字典，离散观测值为 int，连续观测值为 float。
        up_memo (dict): 记录已计算节点的消息缓存。最终状态会记录每一个节点对应的上传父节点的信息

    Returns:
        np.ndarray: 长度 = |父状态数| node传向父节点的信息向量。
    """
    # 1. 如果已经算过，直接返回
    if node in up_memo:
        return up_memo[node]
    
    # print("node: ",node)

    
    # print("node: ",node)
    # 2. 如果有子节点的话，先递归获取所有孩子节点的消息
    child_msgs = {}
    for c in children.get(node, []):
        # print("children: ",c)
        child_msgs[c] = upward_messages(c, children, parent, params, evidence, up_memo)

        # print("node: ",node,"params[node]: ",params[node],"c: ",c,"child_msgs[c]: ",child_msgs[c])

    # 3. node是连续观测节点（条件高斯）
    #    用 N(x_obs | mean[parent=i], var[parent=i]) 作为 message[i]
    if "mean" in params[node]:
        x_obs = evidence[node]
        means = params[node]["mean"]  # array of shape (n_parent_states,)
        vars_ = params[node]["var"]   # same shape
        msg = np.zeros_like(means, dtype=float)
        for i in range(len(means)):
            μ = means[i]
            σ2 = vars_[i]
            coef = 1.0 / math.sqrt(2 * math.pi * σ2)
            exponent = -0.5 * (x_obs - μ)**2 / σ2
            msg[i] = coef * math.exp(exponent)
        up_memo[node] = msg

        # print("连续观测node: ",node,"up_memo[node]: ",up_memo[node])
        return msg

    states = params[node]["states"]  # 离散状态列表

    # 4. node是根潜变量（有 'prior' 没有 'cpt'）
    if "prior" in params[node] and "cpt" not in params[node]:
        prior = params[node]["prior"]  # array of shape (n_states,)
        # print("根节点的prior: ",prior)
        # 根向父节点（伪父）只发一个标量
        msg_to_parent = np.zeros(len(states), dtype=float)
        for i,s in enumerate(states):
            # children 对应状态 i 的似然乘积
            eps = 1e-12
            sum_c = 0.0
            for c in children.get(node, []):
                p = child_msgs[c][i]
                # avoid log(0)
                sum_c += math.log( max(p, eps) )
            # sum_c = 0.0
            # for c in children.get(node, []):
            #     # print("c: ",c,"child_msgs[c]: ",child_msgs[c])
            #     sum_c += math.log(child_msgs[c][i])  #把信息值取log后相加，避免子节点太多后连乘导致溢出
            msg_to_parent[i] = prior[i] * math.exp(sum_c)
        up_memo[node] = msg_to_parent
        # print("根节点的信息：",msg_to_parent)
        return msg_to_parent

    # 5. node是其它离散节点：都有 'cpt'
    cpt = params[node]["cpt"]  # shape = (n_parent_states, n_node_states)
    n_node_states, n_parent_states = cpt.shape

    # print("node: ",node,"cpt: ",cpt)

    # 5.1 local likelihood：离散观测叶子如果不写这一行，代码在遇到没有观测的节点时就找不到 local_like，
    # 要么出错，要么就必须为“有观测”/“无观测”写完全不同的分支逻辑。把它设置成全 1，就能统一对待，代码也更简洁清晰。
    if node in evidence:
        obs = evidence[node]
        local_like = np.array([1.0 if s == obs else 0.0 for s in states], dtype=float)
    else:
        # 隐变量或内部非观测：likelihood=1
        local_like = np.ones(n_node_states, dtype=float)

    # 5.2 在node的父节点的每个状态下，使用sum-product加和掉关于node的状态
    msg_to_parent = np.zeros(n_parent_states, dtype=float)
    for pi in range(n_parent_states):
        total = 0.0
        for i in range(n_node_states):
            eps = 1e-12
            sum_c = 0.0
            for c in children.get(node, []):
                p = child_msgs[c][i]
                # avoid log(0)
                sum_c += math.log( max(p, eps) )

            # sum_c = 0.0
            # for c in children.get(node, []):
            #     # print("c: ",c,"child_msgs[c]: ",child_msgs[c])
            #     print("child_msgs[c][i]: ",child_msgs[c][i])
            #     sum_c += math.log(child_msgs[c][i])               #把信息值取log后相加，避免子节点太多后连乘导致溢出
            #     # 这里用 cpt[i,pi] —— P(node_state=i | parent_state=pi)
            total += cpt[i,pi] * local_like[i] * math.exp(sum_c)
        msg_to_parent[pi] = total

    up_memo[node] = msg_to_parent
    # print("node: ",node,"up_memo[node]: ",up_memo[node])
    return msg_to_parent

# 信息下传时,node是当前要考虑的节点，down_memo是node的父节点传给node的信息。
# 信息下传时,node的每个子节点child，都有一个自己的down_to_child_memo
# downward_messages(node, children, parent, params, memo, down_to_node_memo, marginals)执行前down_memo[node]是有值的，
# 除了根节点可以看成特殊情况的有值
def downward_pass(node, children, parent, params,
                   up_memo, down_memo, marginals):

    # 1) 先算出当前节点的后验 marg
    if "prior" in params[node] and "cpt" not in params[node]:
        # 根节点：直接把 up_memo[node] 归一化
        root_msg = up_memo[node]                   # 长度 = n_states
        marg = root_msg / np.sum(root_msg)
        # print("根节点的概率：",marg)
    else:
        # 非根节点：1) 用 CPT 和父下行消息计算 S = P(node=j | evidence_above)
        #    params[node]['cpt'] 形状 (n_node_states, n_parent_states)
        cpt = params[node]['cpt']
        # down_memo 形状 (n_parent_states,)
        S = cpt @ down_memo[node]    # → shape = (n_node_states,)

        # 2) 抹除父节点对本节点的影响后（S 已经包含了），乘上所有孩子的上行消息
        joint = S.copy()
        for c in children.get(node, []):
            # up_memo[c] 也是长度 n_node_states
            joint *= up_memo[c]

        # 3) 归一化得到后验
        marg = joint / np.sum(joint)
        marginals[node] = marg

    marginals[node] = marg

    # 2) 选择本节点到孩子的“下行消息”基准 current_msg
    #    根用自己的概率向量，非根用自身的cpt矩阵
    

    # 3) 把 current_msg 分发给每个 child
    for child in children.get(node, []):
        # 如果 child 是连续观测叶子，就不需要再往下
        if "cpt" not in params[child]:
            continue

        if parent[node] == "":
            #如果是根节点，直接使用根节点自身概率
            joint = params[node]["prior"]
        else:
            #如果是非根节点，抹除父节点对本节点的影响
            S = params[node]['cpt']@ down_memo[node]
            joint = S.copy()
  
        #当前node自身的状态数目
        n_node_states=len(joint)
        down_to_child = np.zeros(n_node_states, dtype=float)
        for i in range(n_node_states):  
            prod_sibs = 1.0
            for sib in children[node]:
                # 剔除当前child对node的贡献：prod over siblings k≠child of up_memo[k][j]
                if sib == child:
                    continue
                prod_sibs *= up_memo[sib][i]
            down_to_child[i] = joint[i] * prod_sibs
        down_memo[child]= down_to_child
        # down_to_child_memo是node传递给子节点child的信息
        downward_pass(child, children, parent, params, up_memo, down_memo, marginals)



def infer_latents(tree_df, params, evidence):
    """
    tree_df: ModelEs_renum DataFrame
    params:  上面构造好的参数字典（mixed 离散+连续）
    evidence: 单个样本的观测字典，形如 {"1":0, "2":1, ...}
    返回：各节点的后验分布 marginals[node]
    """
    # 1) 构建 children, parent
    children, parent = build_tree(tree_df)

    # 2) 找根节点（parent[node]==''）
    roots = [n for n, p in parent.items() if p == '']
    assert len(roots) == 1, f"树应只有一个根，但找到 {len(roots)}: {roots}"
    root = roots[0]

    # print("evidence: ",evidence)

    # 3) 上行消息
    up_memo = {} #由node传向node父节点的信息
    # upward_messages(node…) → 返回 node 发给父节点的消息（并把它存入 up_memo）
    upward_messages(root, children, parent, params, evidence, up_memo)

    # print("up_memo: ",up_memo)

    # 4) 下行消息 & 计算后验
    down_memo = {}#由node父节点传向node的信息
    marginals = {}#每个node的后验概率
    # downward_pass(node…) → 就地更新 所有子节点的下行消息和所有节点的后验，不用返回
    downward_pass(root, children, parent, params, up_memo, down_memo, marginals)

    return marginals

def infer_latents_batch(tree_df, params, evidences):
    """
    tree_df:    ModelEs_renum DataFrame  
    params:     CPT/先验字典  
    evidences:  numpy array, shape = (n_samples, n_leaves)
                每行一个观测样本  
    返回:       list of dict，每个 dict 的键按：
                  1) H开头的潜节点由大到小  
                  2) 其余观测叶子由小到大  
                排好序的后验向量
    """
    batch_marginals = []
    for obs_row in evidences:
        # 构造单样本 evidence
        evidence = { str(j+1): val 
                     for j, val in enumerate(obs_row) }
        # evidence = { str(j): val 
        #              for j, val in enumerate(obs_row) }
        # 推断
        marg = infer_latents(tree_df, params, evidence)

        # 排序潜节点键 H<number>（数字大→小）
        latent_keys = sorted(
            (k for k in marg if k.startswith("H")),
            key=lambda x: int(x[1:]),
            reverse=True
        )
        # 排序观测叶子键（数字小→大）
        obs_keys = sorted(
            (k for k in marg if not k.startswith("H")),
            key=lambda x: int(x)
        )

        # 按顺序插入到一个普通 dict
        ordered_dict = {}
        for k in latent_keys + obs_keys:
            ordered_dict[k] = marg[k]

        batch_marginals.append(ordered_dict)
        

    return batch_marginals


