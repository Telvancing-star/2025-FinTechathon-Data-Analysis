import numpy as np
import pandas as pd
import pickle
from utils.popularity_tr import Pop
from sklearn.preprocessing import StandardScaler


def create_compatible_pop_instance(results_file_path, external_X, adjacency_csv_file):
    """
    根据保存的Pop数据创建兼容的Pop实例
    """
    with open(results_file_path, 'rb') as f:
        full_results = pickle.load(f)

    # 提取Pop数据（假设B=1，取第一个重复）
    pop_data = full_results['pop_datas'][0]

    # 创建Pop实例
    pop_instance = Pop(
        N=pop_data['N'],
        beta=pop_data['beta'].tolist(),  # 转换为list
        delta=pop_data['delta'],
        C_min=pop_data['C_min'],
        C_max=pop_data['C_max'],
        external_X=external_X,
        adjacency_csv_file=adjacency_csv_file
    )

    # 手动设置估计的alpha值
    pop_instance.alpha = pop_data['alpha']
    pop_instance.C_alpha = pop_data['C_alpha']
    pop_instance.C_beta = pop_data['C_beta']
    pop_instance.beta = full_results['estimators'][0]  # 重置 beta 为 beta_hat

    return pop_instance

def compute_edge_probability_matrix(data, beta, alpha_hat=None):
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
    # 从CSV文件读取数据
    data = pd.read_csv('./data/Social/compressed_features_expanded.csv')
    df_sorted = data.sort_values('local_node_id')
    popularity_records = {}
    for idx, local_node_id in enumerate(df_sorted['local_node_id']):
        popularity_records[local_node_id] = gamma[idx]

    print(f"边概率矩阵形状: {P_edge_matrix.shape}")
    print(f"平均边概率: {np.mean(P_edge_matrix):.6f}")
    print(f"最大边概率: {np.max(P_edge_matrix):.6f}")
    print(f"最小边概率: {np.min(P_edge_matrix):.6f}")

    return P_edge_matrix, P_edge, gamma, popularity_records


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
    P_edge_matrix, P_edge, gamma, popularity_records = compute_edge_probability_matrix(data, beta_hat, alpha_hat)

    # 6. 计算模型拟合优度
    print("\n6. 计算模型拟合优度...")

    # 预测的边数
    predicted_edges = np.sum(P_edge_matrix)
    actual_edges = total_edges
    edge_prediction_error = abs(predicted_edges - actual_edges) / actual_edges

    # 7. 保存结果
    results = {
        'beta_hat': beta_hat,
        'alpha_hat': alpha_hat,
        'gamma': gamma,
        'P_edge': P_edge,
        'P_edge_matrix': P_edge_matrix,
        'actual_in_degree': data.in_degrees,
        'goodness_of_fit': {
            'predicted_edges': predicted_edges,
            'actual_edges': actual_edges,
            'edge_prediction_error': edge_prediction_error,
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
    print(f"流行度参数γ范围: [{np.min(gamma):.4f}, {np.max(gamma):.4f}]")

    return results


if __name__ == "__main__":
    # 从pkl文件读取对象
    file_path = f'./data/results4171delta25.pkl'
    adjacency_csv_file = './data/Social/adjacency_matrix_origin.csv'
    df = pd.read_csv('./data/Social/compressed_features_expanded.csv')

    feature_cols = [f'feature_{i}' for i in range(1, 22)]
    X_or = df[feature_cols].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X_or)
    pop_instance = create_compatible_pop_instance(file_path, external_X=X, adjacency_csv_file=adjacency_csv_file)

    # 运行概率分析
    print("运行概率分析 pipeline...")
    probability_results = run_probability_analysis(
        data=pop_instance,
        beta_hat=pop_instance.beta,
        output_file='./data/Social/edge_probability_matrix.pkl'
    )