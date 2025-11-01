import numpy as np
import pandas as pd
import pickle
from _5_data_reset import CompatibleData


def create_node_mapping(df_reindexed):
    """创建节点索引到 local_node_id 的映射"""
    # 假设 df_reindexed 包含 global_index 和 local_node_id
    mapping = {}

    for idx, row in df_reindexed.iterrows():
        global_idx = row['global_index']
        local_id = row['local_node_id']
        mapping[global_idx] = local_id

    print(f"创建了 {len(mapping)} 个节点的映射")
    return mapping

def compute_edge_probability_matrix(data, beta, alpha_hat=None, node_mapping=None):
    """
    带节点映射的边存在概率计算
    """
    if alpha_hat is None:
        total_edges = data.A.sum()
        C_beta = np.mean(np.exp(data.X @ beta))
        alpha_hat = np.log(total_edges * np.sqrt(2) / (data.N * C_beta)) - np.log(data.N - 1)

    # 计算流行度参数
    gamma = np.exp(data.X @ beta + alpha_hat)  # (N,)

    P_edge = gamma.reshape(1, -1) / np.sqrt(gamma.reshape(1, -1) ** 2 + 2)
    P_edge_matrix = np.tile(P_edge, (data.N, 1))
    P_edge = P_edge_matrix[0, :]
    np.fill_diagonal(P_edge_matrix, 0)  # 移除自环概率

    # 创建流行度记录
    popularity_records = {}
    for global_idx in range(data.N):
        popularity_records[node_mapping.get(global_idx, f'unknown_{global_idx}')] = gamma[global_idx]

    print(f"边概率矩阵形状: {P_edge_matrix.shape}")
    print(f"平均边概率: {np.mean(P_edge_matrix):.6f}")
    print(f"最大边概率: {np.max(P_edge_matrix):.6f}")
    print(f"最小边概率: {np.min(P_edge_matrix):.6f}")

    return P_edge_matrix, P_edge, gamma, alpha_hat, popularity_records

# def compute_edge_probability_matrix(data, beta, alpha_hat=None):
#     """
#     计算边存在概率矩阵 P(a_ij=1 | X, Z)
#     根据论文公式(2): P(a_ij=1) = exp(-||Z_i - Z_j||² / (2γ_j²))
#
#     由于Z是潜在变量，我们使用期望值近似
#     """
#     # 计算流行度参数 γ_j = exp(X_j^T β + α)
#     if alpha_hat is None:
#         # 估计 α (根据网络密度)
#         total_edges = data.A.sum()
#         C_beta = np.mean(np.exp(data.X @ beta))
#         alpha_hat = np.log(total_edges * np.sqrt(2) / (data.N * C_beta)) - np.log(data.N - 1)
#
#     gamma_val = np.exp(data.X @ beta + alpha_hat)
#     gamma = list(zip(index, gamma_val))
#
#     # 由于Z是标准正态分布，E[exp(-||Z_i-Z_j||²/(2γ_j²))] = γ_j / sqrt(γ_j² + 2)
#     # 这是论文中推导的期望概率
#
#     P_edge = gamma_val.reshape(1, -1) / np.sqrt(gamma_val.reshape(1, -1) ** 2 + 2)
#     P_edge_matrix = np.tile(P_edge, (data.N, 1))
#     P_edge = P_edge_matrix[0, :]
#     print(f"形状: {P_edge.shape}")
#
#     # 移除自环概率
#     np.fill_diagonal(P_edge_matrix, 0)
#
#     print(f"边概率矩阵形状: {P_edge_matrix.shape}")
#     print(f"平均边概率: {np.mean(P_edge_matrix):.6f}")
#     print(f"最大边概率: {np.max(P_edge_matrix):.6f}")
#     print(f"最小边概率: {np.min(P_edge_matrix):.6f}")
#
#     return P_edge_matrix, P_edge, gamma, alpha_hat


# def compute_reciprocity_probability_matrix(data, beta):
#     """
#     计算互惠概率矩阵 P(a_ji=1 | a_ij=1, X)
#     根据论文公式(4): P = sqrt( exp(2Δ_ij^T β) / (exp(2Δ_ij^T β) + 1) )
#     """
#     N = data.N
#     P_recip = np.zeros((N, N))
#
#     for i in range(N):
#         for j in range(N):
#             if i != j:
#                 delta_ij = data.X[i] - data.X[j]  # Δ_ij = X_i - X_j
#                 exp_term = np.exp(2 * delta_ij @ beta)
#                 P_recip[i, j] = np.sqrt(exp_term / (exp_term + 1))
#
#     print(f"互惠概率矩阵形状: {P_recip.shape}")
#     print(f"平均互惠概率: {np.mean(P_recip):.6f}")
#
#     return P_recip
#
#
# def compute_transitivity_probability_matrix(data, beta):
#     """
#     计算传递性概率矩阵 P(a_ik=1 | a_ij=1, a_jk=1, X)
#     根据论文公式(5): P = sqrt( exp(2Δ_kj^T β) / (2exp(2Δ_kj^T β) + 1) )
#     """
#     N = data.N
#     # 这是一个三维概率，我们返回每个(i,j,k)三元组的概率
#
#     # 由于计算量很大，我们可以计算代表性的概率或使用采样
#     print("计算传递性概率矩阵...")
#
#     # 方法1: 计算所有(i,j)对的传递性特征
#     P_trans_pairwise = np.zeros((N, N))
#
#     for j in range(N):
#         for k in range(N):
#             if j != k:
#                 delta_kj = data.X[k] - data.X[j]  # Δ_kj = X_k - X_j
#                 exp_term = np.exp(2 * delta_kj @ beta)
#                 P_trans_pairwise[j, k] = np.sqrt(exp_term / (2 * exp_term + 1))
#
#     print(f"传递性概率矩阵形状: {P_trans_pairwise.shape}")
#     print(f"平均传递性概率: {np.mean(P_trans_pairwise):.6f}")
#
#     return P_trans_pairwise
#
#
# def compute_expected_in_degrees(data, beta, alpha_hat=None):
#     """
#     计算期望入度 E[d_i^in | X]
#     根据论文公式(3): E[d_i^in | X] = N * exp(X_i^T β + α_N) / sqrt(2)
#     """
#     if alpha_hat is None:
#         total_edges = data.A.sum()
#         C_beta = np.mean(np.exp(data.X @ beta))
#         alpha_hat = np.log(total_edges * np.sqrt(2) / (data.N * C_beta)) - np.log(data.N - 1)
#
#     expected_in_degree = data.N * np.exp(data.X @ beta + alpha_hat) / np.sqrt(2)
#
#     print(f"期望入度范围: [{np.min(expected_in_degree):.2f}, {np.max(expected_in_degree):.2f}]")
#     print(f"平均期望入度: {np.mean(expected_in_degree):.2f}")
#     print(f"实际平均入度: {np.mean(data.in_degrees):.2f}")
#
#     return expected_in_degree


def run_probability_analysis(data, beta_hat, output_file='probability_analysis.pkl'):
    """
    运行完整的概率分析 pipeline
    """
    print("开始概率分析...")

    # 1. 估计 alpha
    print("\n1. 估计 alpha...")
    total_edges = data.A.sum()
    C_beta = np.mean(np.exp(data.X @ beta_hat))
    alpha_hat = np.log(total_edges * np.sqrt(2) / (data.N * C_beta)) - np.log(data.N - 1)
    print(f"估计的 alpha: {alpha_hat:.6f}")

    # 2. 计算各种概率矩阵
    print("\n2. 计算边存在概率...")
    # P_edge_matrix, P_edge, gamma, alpha_hat = compute_edge_probability_matrix(data, index, beta_hat, alpha_hat)
    # 使用带映射的版本
    P_edge_matrix, P_edge, gamma, alpha_hat, popularity_records = compute_edge_probability_matrix(compatible_data, beta_hat, alpha_hat, node_mapping)

    # print("\n3. 计算互惠概率...")
    # P_recip = compute_reciprocity_probability_matrix(data, beta_hat)
    #
    # print("\n4. 计算传递性概率...")
    # P_trans = compute_transitivity_probability_matrix(data, beta_hat)
    #
    # print("\n5. 计算期望入度...")
    # expected_in_degree = compute_expected_in_degrees(data, beta_hat, alpha_hat)

    # 6. 计算模型拟合优度
    print("\n6. 计算模型拟合优度...")

    # 预测的边数
    predicted_edges = np.sum(P_edge_matrix)
    actual_edges = total_edges
    edge_prediction_error = abs(predicted_edges - actual_edges) / actual_edges

    # # 互惠性拟合
    # actual_reciprocal_pairs = 0
    # total_possible_reciprocal = 0
    #
    # for i in range(data.N):
    #     for j in range(i + 1, data.N):
    #         if data.A[i, j] == 1 and data.A[j, i] == 1:
    #             actual_reciprocal_pairs += 1
    #         if data.A[i, j] == 1:
    #             total_possible_reciprocal += 1
    #
    # actual_reciprocity_rate = actual_reciprocal_pairs / total_possible_reciprocal if total_possible_reciprocal > 0 else 0

    # 7. 保存结果
    results = {
        'beta_hat': beta_hat,
        'alpha_hat': alpha_hat,
        'gamma': gamma,
        'P_edge': P_edge,
        'P_edge_matrix': P_edge_matrix,
        # 'P_reciprocity': P_recip,
        # 'P_transitivity': P_trans,
        # 'expected_in_degree': expected_in_degree,
        'actual_in_degree': data.in_degrees,
        'goodness_of_fit': {
            'predicted_edges': predicted_edges,
            'actual_edges': actual_edges,
            'edge_prediction_error': edge_prediction_error,
            # 'predicted_reciprocity_rate': np.mean(P_recip),
            # 'actual_reciprocity_rate': actual_reciprocity_rate,
            # 'reciprocity_prediction_error': abs(np.mean(P_recip) - actual_reciprocity_rate),
        },
        'data_info': {
            'N': data.N,
            'p': data.p,
            'delta': data.delta
        },
        'popularity_records': popularity_records
    }

    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"\n概率分析结果已保存到: {output_file}")

    # 8. 输出总结
    print("\n=== 概率分析总结 ===")
    print(f"边数预测: {predicted_edges:.0f} (实际: {actual_edges}, 误差: {edge_prediction_error * 100:.2f}%)")
    # print(f"互惠率预测: {np.mean(P_recip) * 100:.2f}% (实际: {actual_reciprocity_rate * 100:.2f}%)")
    # print(f"平均传递性概率: {np.mean(P_trans) * 100:.2f}%")
    # print(f"流行度参数γ范围: [{np.min(gamma_val):.4f}, {np.max(gamma_val):.4f}]")

    return results


if __name__ == "__main__":
    # 从pkl文件读取对象
    with open('./data/Social/compatible_data.pkl', 'rb') as f:  # 注意是'rb'二进制读取模式
        compatible_data = pickle.load(f)

    with open('./data/Social/beta.pkl', 'rb') as f:  # 注意是'rb'二进制读取模式
        beta = pickle.load(f)

    # 从CSV文件读取数据
    data = pd.read_csv('./data/Social/compressed_features_expanded.csv')
    node_mapping = create_node_mapping(data)

    # 运行概率分析
    print("运行概率分析 pipeline...")
    probability_results = run_probability_analysis(
        data=compatible_data,
        beta_hat=np.asarray(beta["beta_hat"]).flatten(),  # 使用之前TR估计器得到的结果
        output_file='./data/Social/edge_probability_matrix.pkl'
    )