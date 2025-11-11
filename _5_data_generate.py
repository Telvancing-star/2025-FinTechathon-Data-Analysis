# 构建为估计器可用的数据格式, 完成 beta_hat 参数估计
from utils.popularity_tr import Pop, Pop_NR, plugin
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


def create_compatible_data(X, A, in_degrees, beta):
    """
    创建与 Pop_NR 类兼容的数据对象
    关键修复：确保所有维度匹配
    """

    return CompatibleData(X, A, in_degrees, beta)


def complete_data_processing_pipeline(filepath, X_output, adj_output, beta):
    """完整的数据处理流程"""

    # 1. 读取特征数据并重新编号
    print("步骤1: 处理特征数据...")
    df = pd.read_csv(filepath)
    # df, original_to_copies, node_mapping = reindex_nodes_with_duplicates(df)

    # 2. 构建特征矩阵
    print("\n步骤2: 构建特征矩阵...")
    feature_cols = [f'feature_{i}' for i in range(1, 22)]
    X = df[feature_cols].values
    X_df = pd.DataFrame(X, columns=feature_cols)
    X_df.to_csv(X_output, index=False)
    print(f"特征矩阵形状: {X.shape}")

    # 4. 构建邻接矩阵
    print("\n步骤3: 构建邻接矩阵...")
    total_nodes = 4171  # 根据你的节点总数调整
    A = build_adjacency_from_csv(adj_output, total_nodes)

    # 5. 计算入度
    print("\n步骤4: 计算网络统计量...")
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
    compatible_data = create_compatible_data(X, A, in_degrees, beta)

    return compatible_data, df


if __name__ == "__main__":
    # 执行完整流程
    filepath = 'data/Social/compressed_features_expanded.csv'
    X_output = './data/Social/feature_matrix_with_headers.csv'
    adj_output = './data/Social/adjacency_matrix_origin.csv'

    np.random.seed(42)
    initial_beta = np.random.normal(0, 0.01, 21)  # 21个特征（不包括截距）

    compatible_data, df = complete_data_processing_pipeline(filepath=filepath, X_output=X_output, adj_output=adj_output, beta=initial_beta)

    # # 将对象保存为pkl文件
    # with open('./data/Social/compatible_data.pkl', 'wb') as f:  # 注意是'wb'二进制写入模式
    #     pickle.dump(compatible_data, f)

    # 在运行估计器之前添加维度检查
    print("数据维度检查:")
    print(f"X.shape: {compatible_data.X.shape}")  # 应该是 (4171, 21)
    print(f"A.shape: {compatible_data.A.shape}")  # 应该是 (4171, 4171)
    print(f"in_degrees.shape: {compatible_data.in_degrees.shape}")  # 应该是 (4171,)
    print(f"col_prod.shape: {compatible_data.col_prod.shape}")  # 应该是 (4171, 4171)

    # 运行 TR 估计器
    print("\n步骤5: 运行 TR 估计器...")
    tr_estimator = Pop_NR_Fixed(compatible_data)
    beta_hat = tr_estimator.run(running_parameter=initial_beta, max_iter=50)

    beta = {
        "initial_beta": initial_beta,
        "beta_hat": beta_hat,
    }

    # 将对象保存为pkl文件
    compatible_data, df = complete_data_processing_pipeline(filepath=filepath, X_output=X_output, adj_output=adj_output,
                                                            beta=beta_hat)
    with open('./data/Social/compatible_data.pkl', 'wb') as f:  # 注意是'wb'二进制写入模式
        pickle.dump(compatible_data, f)

    # # 将对象保存为pkl文件
    # with open('./data/Social/beta.pkl', 'wb') as f:  # 注意是'wb'二进制写入模式
    #     pickle.dump(beta, f)

    print("估计完成!")
    print(f"参数估计结果: {beta_hat.reshape(-1)}")

if __name__ == "__main__":
    from joblib import Parallel, delayed
    import pickle
    from tqdm import tqdm
    import numpy as np

    np.random.seed(42)
    initial_beta = np.random.normal(0, 0.01, 21)  # 21个特征
    beta = initial_beta.tolist()  # 转换为list格式

    feature_cols = [f'feature_{i}' for i in range(1, 22)]
    your_X = df[feature_cols].values
    print(f"特征矩阵形状: {your_X.shape}")

    C_max = 25
    C_min = 9


    def map_fun(b):
        d = Pop(N, beta, delta, C_min, C_max, N + b + int(10000 * delta), external_X=your_X)
        true_beta = np.array(beta).reshape(-1)
        data_NR = Pop_NR(d)
        np.random.seed(b)  # 每个重复使用不同的随机种子
        running_beta = np.random.normal(0, 0.01, 21)
        beta_hat = data_NR.run(running_parameter=running_beta)

        beta_hat = beta_hat.reshape(-1)
        est_std = np.sqrt(np.diag(plugin(d, beta_hat)[0])).reshape(-1)
        bi_cover = np.logical_and(((beta_hat - 1.96 * est_std) < true_beta), ((beta_hat + 1.96 * est_std) > true_beta))
        print(f"{b}:{bi_cover}")
        return beta_hat, true_beta, est_std, bi_cover


    B = 1000
    Tasks = list(range(B))
    N = your_X.shape[0]
    for delta in [0.25]:
        d0 = Pop(N, beta, delta, C_min, C_max, external_X=your_X)
        print("N: ", d0.N)
        print("beta shape: ", len(beta))
        print("C_max: ", C_max)
        print("C_min: ", C_min)
        print("delta: ", delta)
        print("alpha:", d0.alpha)
        print("C_alpha:", d0.C_alpha)
        print("network nonzereos:", np.sum(d0.A))

        Results = Parallel(n_jobs=-1, backend='loky', verbose=10)(delayed(map_fun)(b) for b in range(B))
        estimators = np.array([est.reshape(-1) for est, _, _, _ in Results])  # B x (p+1)
        parameters = np.array([par.reshape(-1) for _, par, _, _ in Results])  # B x (p+1)
        plug_std = np.array([est_std.reshape(-1) for _, _, est_std, _ in Results])  # B x (p+1)
        cover = np.array([cv.reshape(-1) for _, _, _, cv in Results])  # B x (p+1)
        with open(f'./data/results{N}delta{int(100 * delta)}.pkl', 'wb') as f:
            pickle.dump(Results, f)

        covariance = (estimators.T @ estimators) / B - (
                    estimators.mean(axis=0).reshape(-1, 1) @ estimators.mean(axis=0).reshape(1, -1))
        monte_std = np.sqrt(np.diag(covariance))
        ARE = np.mean(np.abs((plug_std / monte_std) - 1), axis=0)
        RMSE = np.sqrt(np.mean((estimators - parameters) ** 2, axis=0))
        std_estimation = np.mean(plug_std, axis=0)
        cover_rate = np.mean(cover, axis=0)

        print(f"N:{N};monte:", monte_std)
        print(f"N:{N};plug:", std_estimation)
        print(f"N:{N};RMSE:", RMSE)
        print(f"N:{N};ARE:", ARE)
        print(f"N:{N};cover:", cover_rate)


    # # results

    # In[13]:
    import pickle

    np.random.seed(42)
    initial_beta = np.random.normal(0, 0.01, 21)
    beta = initial_beta.tolist()
    C_max = 25
    C_min = 9
    for delta in [0.25]:
        B = 1000
        N = your_X.shape[0]
        file_path = f'./data/results{N}delta{int(delta * 100)}.pkl'

        with open(file_path, 'rb') as file:
            Results = pickle.load(file)
            estimators = np.array([est.reshape(-1) for est, _, _, _ in Results])  # B x (p+1)
            parameters = np.array([par.reshape(-1) for _, par, _, _ in Results])  # B x (p+1)
            plug_std = np.array([est_std.reshape(-1) for _, _, est_std, _ in Results])  # B x (p+1)
            cover = np.array([cv.reshape(-1) for _, _, _, cv in Results])  # B x (p+1)

            covariance = (estimators.T @ estimators) / B - (
                        estimators.mean(axis=0).reshape(-1, 1) @ estimators.mean(axis=0).reshape(1, -1))
            monte_std = np.sqrt(np.diag(covariance))
            ARE = np.mean(np.abs((plug_std / monte_std) - 1), axis=0)
            RMSE = np.sqrt(np.mean((estimators - parameters) ** 2, axis=0))
            std_estimation = np.mean(plug_std, axis=0)
            cover_rate = np.mean(cover, axis=0)
            print("N: ", N)
            print("delta: ", delta)
            #             print(f"N:{N};monte:", monte_std)
            #             print(f"N:{N};plug:", std_estimation)
            print(f"N:{N};RMSE:", RMSE)
            print(f"N:{N};ARE:", ARE)
            print(f"N:{N};cover:", cover_rate)