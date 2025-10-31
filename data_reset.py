import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict
from popularity_tr import Pop_NR


def reindex_nodes_with_duplicates(df):
    """
    对重复节点重新编号，并记录重复关系
    """
    # 按 ego_id 和 local_node_id 分组
    df_sorted = df.sort_values(['ego_id', 'local_node_id'])

    # 记录每个原始节点的所有副本
    original_to_copies = defaultdict(list)
    node_mapping = {}  # (ego_id, local_node_id) -> new_global_index
    new_global_indices = []

    original_nodes_count = 4039
    current_new_id = original_nodes_count  # 从4039开始编号重复节点

    for idx, row in df_sorted.iterrows():
        key = (row['ego_id'], row['local_node_id'])
        original_id = row['local_node_id']

        if key not in node_mapping:
            # 首次出现，使用原始的 local_node_id
            node_mapping[key] = original_id
            new_global_indices.append(original_id)
            original_to_copies[original_id].append(original_id)  # 自己也是副本之一
        else:
            # 重复出现，分配新编号
            node_mapping[key] = current_new_id
            new_global_indices.append(current_new_id)
            original_to_copies[original_id].append(current_new_id)
            current_new_id += 1

    # 更新数据框
    df_reindexed = df_sorted.copy()
    df_reindexed['global_index_new'] = new_global_indices

    print(f"原始节点数: 4039")
    print(f"扩充后节点数: {len(df_reindexed)}")
    print(f"重复关系统计:")
    for orig_id, copies in list(original_to_copies.items())[:5]:  # 显示前5个
        print(f"  节点 {orig_id} 有 {len(copies)} 个副本: {copies}")

    return df_reindexed, original_to_copies, node_mapping


# # 读取特征数据
# df = pd.read_csv('data/compressed_features.csv')
# df_reindexed, original_to_copies, node_mapping = reindex_nodes_with_duplicates(df)


def expand_edges_with_copies(original_edges_file, original_to_copies, output_file):
    """
    根据节点副本关系扩充边数据
    """
    # 读取原始边数据
    original_edges = []
    with open(original_edges_file, 'r') as f:
        for line in f:
            if line.strip():
                u, v = map(int, line.strip().split())
                original_edges.append((u, v))

    print(f"原始边数: {len(original_edges)}")

    # 生成扩充的边数据
    expanded_edges = set()

    # 首先添加所有原始边
    for u, v in original_edges:
        expanded_edges.add((u, v))

    # 为每个原始边，为节点的所有副本生成对应的边
    for u_orig, v_orig in original_edges:
        u_copies = original_to_copies.get(u_orig, [u_orig])
        v_copies = original_to_copies.get(v_orig, [v_orig])

        # 为所有副本组合生成边（除了原始边本身）
        for u_copy in u_copies:
            for v_copy in v_copies:
                if (u_copy, v_copy) not in expanded_edges:
                    expanded_edges.add((u_copy, v_copy))

    # 转换为列表并排序
    expanded_edges = sorted(list(expanded_edges))

    print(f"扩充后边数: {len(expanded_edges)}")
    print(f"边数增加比例: {len(expanded_edges) / len(original_edges):.2f}x")

    # 保存扩充后的边数据
    with open(output_file, 'w') as f:
        for u, v in expanded_edges:
            f.write(f"{u} {v}\n")

    print(f"扩充后的边数据已保存到: {output_file}")
    return expanded_edges


# # 使用示例
# original_edges_file = 'data/facebook_combined.txt'
# expanded_edges_file = 'data/facebook_combined_expanded.txt'
# expanded_edges = expand_edges_with_copies(original_edges_file, original_to_copies, expanded_edges_file)


def build_adjacency_from_expanded_edges(expanded_edges_file, total_nodes):
    """
    从扩充后的边数据构建邻接矩阵
    """
    # 读取扩充后的边数据
    edges = []
    with open(expanded_edges_file, 'r') as f:
        for line in f:
            if line.strip():
                u, v = map(int, line.strip().split())
                edges.append((u, v))

    # 分离源节点和目标节点
    sources = [edge[0] for edge in edges]
    targets = [edge[1] for edge in edges]

    # 构建邻接矩阵
    A = csr_matrix((np.ones(len(sources)), (sources, targets)),
                   shape=(total_nodes, total_nodes))

    # 移除自环
    A.setdiag(0)
    A.eliminate_zeros()

    print(f"邻接矩阵形状: {A.shape}")
    print(f"网络边数: {A.nnz}")
    print(f"网络密度: {A.nnz / (total_nodes * (total_nodes - 1)):.6f}")

    return A


# # 构建邻接矩阵
# total_nodes = len(df_reindexed)
# A = build_adjacency_from_expanded_edges(expanded_edges_file, total_nodes)

def complete_data_processing_pipeline():
    """完整的数据处理流程"""

    # 1. 读取特征数据并重新编号
    print("步骤1: 处理特征数据...")
    df = pd.read_csv('data/compressed_features.csv')
    df_reindexed, original_to_copies, node_mapping = reindex_nodes_with_duplicates(df)

    # 2. 构建特征矩阵
    print("\n步骤2: 构建特征矩阵...")
    feature_cols = [f'feature_{i}' for i in range(22)]
    X = df_reindexed[feature_cols].values
    print(f"特征矩阵形状: {X.shape}")

    # 3. 扩充边数据
    print("\n步骤3: 扩充边数据...")
    original_edges_file = 'data/facebook_combined.txt'
    expanded_edges_file = 'data/facebook_combined_expanded.txt'
    expanded_edges = expand_edges_with_copies(original_edges_file, original_to_copies, expanded_edges_file)

    # 4. 构建邻接矩阵
    print("\n步骤4: 构建邻接矩阵...")
    total_nodes = len(df_reindexed)
    A = build_adjacency_from_expanded_edges(expanded_edges_file, total_nodes)

    # 5. 计算入度
    print("\n步骤5: 计算网络统计量...")
    in_degrees = np.array(A.sum(axis=0)).flatten()

    # 网络基本信息
    total_edges = A.sum()
    density = total_edges / (total_nodes * (total_nodes - 1))
    avg_in_degree = in_degrees.mean()

    print(f"最终网络统计:")
    print(f"  节点数: {total_nodes}")
    print(f"  边数: {total_edges}")
    print(f"  网络密度: {density:.6f}")
    print(f"  平均入度: {avg_in_degree:.2f}")

    return X, A, in_degrees, df_reindexed


# 创建数据对象
class RealData:
    def __init__(self, X, A, in_degrees):
        self.X = X
        self.N, self.p = X.shape
        self.A = A
        self.in_degrees = in_degrees
        self.col_prod = A.T @ A

        # 根据网络密度估算 delta
        total_edges = A.sum()
        density = total_edges / (self.N * (self.N - 1))
        if density < 1e-4:
            self.delta = 0
        else:
            self.delta = 0.25  # 根据你的网络特性调整

        print(f"设置网络密度参数 delta = {self.delta}")


if __name__ == "__main__":
    # 执行完整流程
    X, A, in_degrees, df_reindexed = complete_data_processing_pipeline()

    # 创建数据对象
    real_data = RealData(X, A, in_degrees)

    # 在运行估计器之前添加维度检查
    print("数据维度检查:")
    print(f"X.shape: {real_data.X.shape}")  # 应该是 (4167, 22)
    print(f"A.shape: {real_data.A.shape}")  # 应该是 (4167, 4167)
    print(f"d_in.shape: {real_data.d_in.shape}")  # 应该是 (4167,)
    print(f"col_prod.shape: {real_data.col_prod.shape}")  # 应该是 (4167, 4167)

    # 运行 TR 估计器
    print("\n步骤6: 运行 TR 估计器...")
    tr_estimator = Pop_NR(real_data)
    initial_beta = np.zeros(22)  # 22个特征（包括截距）
    beta_hat = tr_estimator.run(running_parameter=initial_beta, max_iter=50)

    print("估计完成!")
    print(f"参数估计结果: {beta_hat.reshape(-1)}")
