
from utils.popularity_tr import Pop_NR_Fixed
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def build_adjacency_from_csv(adjacency_csv_file, total_nodes=None):
    """
    从CSV格式的邻接矩阵文件构建稀疏邻接矩阵

    参数:
    adjacency_csv_file: 邻接矩阵CSV文件路径
    total_nodes: 总节点数，如果为None则自动推断

    返回:
    A: CSR格式的稀疏邻接矩阵
    """
    try:
        # 读取CSV文件
        print(f"Reading adjacency matrix from {adjacency_csv_file}...")
        adj_df = pd.read_csv(adjacency_csv_file)

        # 如果第一列是索引，将其设置为索引
        if adj_df.columns[0] == 'Unnamed: 0' or adj_df.columns[0] == 'index':
            adj_df = adj_df.set_index(adj_df.columns[0])
            print("Set first column as index")

        # 转换为numpy数组
        adj_matrix = adj_df.values

        print(f"原始邻接矩阵形状: {adj_matrix.shape}")
        print(f"原始矩阵数据类型: {adj_matrix.dtype}")

        # 如果未指定总节点数，使用CSV文件中的节点数
        if total_nodes is None:
            total_nodes = adj_matrix.shape[0]
            print(f"自动推断总节点数: {total_nodes}")
        else:
            print(f"使用指定总节点数: {total_nodes}")

        # 如果CSV矩阵大小与指定节点数不匹配，进行调整
        if adj_matrix.shape[0] != total_nodes or adj_matrix.shape[1] != total_nodes:
            print(f"调整矩阵大小从 {adj_matrix.shape} 到 ({total_nodes}, {total_nodes})")
            # 创建新矩阵
            new_adj_matrix = np.zeros((total_nodes, total_nodes), dtype=adj_matrix.dtype)
            # 将原始数据复制到新矩阵中
            min_rows = min(adj_matrix.shape[0], total_nodes)
            min_cols = min(adj_matrix.shape[1], total_nodes)
            new_adj_matrix[:min_rows, :min_cols] = adj_matrix[:min_rows, :min_cols]
            adj_matrix = new_adj_matrix

        # 转换为稀疏矩阵格式
        A = csr_matrix(adj_matrix)

        # 确保矩阵是对称的（如果是无向图）
        # 如果邻接矩阵不对称，可以取消下面的注释
        # A = A + A.T
        # A.data = np.ones_like(A.data)  # 确保值为1

        # 移除自环
        A.setdiag(0)
        A.eliminate_zeros()

        print(f"处理后的邻接矩阵形状: {A.shape}")
        print(f"网络边数: {A.nnz}")
        print(f"网络密度: {A.nnz / (total_nodes * (total_nodes - 1)):.8f}")

        return A

    except Exception as e:
        print(f"Error building adjacency matrix: {e}")
        raise


def save_adjacency_matrix(A, output_file):
    """保存邻接矩阵为.npz格式"""
    save_npz(output_file, A)
    print(f"邻接矩阵已保存到: {output_file}")


def load_adjacency_matrix(input_file):
    """加载.npz格式的邻接矩阵"""
    A = load_npz(input_file)
    print(f"加载邻接矩阵: 形状{A.shape}, 边数{A.nnz}")
    return A

# class CompatibleData:
#     def __init__(self, X, A, in_degrees):
#         self.X = X
#         self.N, self.p = X.shape
#         self.A = A
#
#         # 关键修复：确保 in_degrees 是正确形状的 numpy 数组
#         self.in_degrees = np.asarray(in_degrees).flatten()
#
#         # 关键修复：确保 col_prod 是稀疏矩阵
#         self.col_prod = A.T @ A
#
#         # 基于网络密度的经验规则
#         density = A.sum() / (self.N * (self.N - 1))
#
#         print(f"\n基于密度的经验规则:")
#         if density < 1e-5:  # 极稀疏网络
#             self.delta = 0
#             desc = "极稀疏网络"
#         elif density < 1e-4:  # 很稀疏网络
#             self.delta = 0.1
#             desc = "很稀疏网络"
#         elif density < 1e-3:  # 稀疏网络
#             self.delta = 0.25
#             desc = "稀疏网络"
#         elif density < 0.01:  # 中等稀疏网络
#             self.delta = 0.5
#             desc = "中等稀疏网络"
#         else:  # 相对稠密网络
#             self.delta = 0.75
#             desc = "相对稠密网络"
#
#         print(f"数据兼容性检查:")
#         print(f"  X.shape: {self.X.shape}")
#         print(f"  A.shape: {self.A.shape}")
#         print(f"  in_degrees.shape: {self.in_degrees.shape}")
#         print(f"  col_prod.shape: {self.col_prod.shape}")
#         print(f"  网络密度: {density:.6f} → {desc}")
#         print(f"  设置 delta = {self.delta}")

# 创建数据对象
class CompatibleData:
    def __init__(self, X, A, in_degrees):
        self.X = X
        self.N, self.p = X.shape
        self.A = A
        self.in_degrees = in_degrees
        self.col_prod = A.T @ A

        # 根据网络密度估算 delta
        avg_out_degree = A.sum() / self.N

        # 假设 C 在 1-10 范围内（典型值）
        C_values = [1, 5, 10]

        print(f"\n不同 C 值对应的 delta 估计:")
        for C in C_values:
            delta_est = (np.log(avg_out_degree) - np.log(C)) / np.log(self.N)
            print(f"  C={C}: δ ≈ {delta_est:.4f}")

        self.delta = 0.25

        print(f"数据兼容性检查:")
        print(f"  X.shape: {self.X.shape}")
        print(f"  A.shape: {self.A.shape}")
        print(f"  in_degrees.shape: {self.in_degrees.shape}")
        print(f"  col_prod.shape: {self.col_prod.shape}")
        print(f"  设置 delta = {self.delta}")

def create_compatible_data(X, A, in_degrees):
    """
    创建与 Pop_NR 类兼容的数据对象
    关键修复：确保所有维度匹配
    """

    return CompatibleData(X, A, in_degrees)

def complete_data_processing_pipeline():
    """完整的数据处理流程"""

    # 1. 读取特征数据并重新编号
    print("步骤1: 处理特征数据...")
    df = pd.read_csv('data/Social/compressed_features_expanded.csv')
    # df, original_to_copies, node_mapping = reindex_nodes_with_duplicates(df)

    # 2. 构建特征矩阵
    print("\n步骤2: 构建特征矩阵...")
    feature_cols = [f'feature_{i}' for i in range(1, 22)]
    X = df[feature_cols].values
    X_df = pd.DataFrame(X, columns=feature_cols)
    X_df.to_csv('./data/Social/feature_matrix_with_headers.csv', index=False)
    print(f"特征矩阵形状: {X.shape}")

    # 4. 构建邻接矩阵
    print("\n步骤4: 构建邻接矩阵...")
    adjacency_csv = './data/Social/adjacency_matrix_origin.csv'
    total_nodes = 4171  # 根据你的节点总数调整

    A = build_adjacency_from_csv(adjacency_csv, total_nodes)

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

    # 关键：创建兼容的数据对象
    compatible_data = create_compatible_data(X, A, in_degrees)

    return compatible_data, df


if __name__ == "__main__":
    # 执行完整流程
    compatible_data, df = complete_data_processing_pipeline()

    # 将对象保存为pkl文件
    with open('./data/Social/compatible_data.pkl', 'wb') as f:  # 注意是'wb'二进制写入模式
        pickle.dump(compatible_data, f)

    # 在运行估计器之前添加维度检查
    print("数据维度检查:")
    print(f"X.shape: {compatible_data.X.shape}")  # 应该是 (4171, 21)
    print(f"A.shape: {compatible_data.A.shape}")  # 应该是 (4171, 4171)
    print(f"in_degrees.shape: {compatible_data.in_degrees.shape}")  # 应该是 (4171,)
    print(f"col_prod.shape: {compatible_data.col_prod.shape}")  # 应该是 (4171, 4171)

    # 运行 TR 估计器
    print("\n步骤6: 运行 TR 估计器...")
    tr_estimator = Pop_NR_Fixed(compatible_data)
    initial_beta = np.zeros(21)  # 21个特征（不包括截距）
    beta_hat = tr_estimator.run(running_parameter=initial_beta, max_iter=50)

    beta = {
        "initial_beta": initial_beta,
        "beta_hat": beta_hat,
    }

    # 将对象保存为pkl文件
    with open('./data/Social/beta.pkl', 'wb') as f:  # 注意是'wb'二进制写入模式
        pickle.dump(beta, f)

    print("估计完成!")
    print(f"参数估计结果: {beta_hat.reshape(-1)}")
