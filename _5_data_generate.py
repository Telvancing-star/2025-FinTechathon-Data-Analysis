# 构建为估计器可用的数据格式, 完成 beta_hat 参数估计
from utils.popularity_tr import Pop, Pop_NR, plugin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from joblib import Parallel, delayed
from tqdm import tqdm

np.random.seed(42)
initial_beta = np.random.normal(0, 0.1, 21)  # 21个特征
beta = initial_beta.tolist()  # 转换为list格式

filepath = 'data/Social/compressed_features_expanded.csv'
adjacency_csv_file = './data/Social/adjacency_matrix_origin.csv'
df = pd.read_csv(filepath)

feature_cols = [f'feature_{i}' for i in range(1, 22)]
X_or = df[feature_cols].values

scaler = StandardScaler()
X = scaler.fit_transform(X_or)

print(f"标准化后特征范围: [{X.min():.3f}, {X.max():.3f}]")
print(f"标准化后特征均值: {np.mean(X, axis=0)}")
print(f"标准化后特征标准差: {np.std(X, axis=0)}")

print(f"特征矩阵形状: {X.shape}")

C_max = 25
C_min = 9
delta = 0.25


def map_fun(b):
    # 创建Pop实例并保存相关信息
    d = Pop(N, beta, delta, C_min, C_max, N + b + int(10000 * delta), external_X=X,
            adjacency_csv_file=adjacency_csv_file)

    # 收集Pop类的关键数据
    pop_data = {
        'N': d.N,
        'beta': np.array(beta).reshape(-1),
        'delta': d.delta,
        'C_min': d.C_min,
        'C_max': d.C_max,
        'alpha': d.alpha,
        'C_alpha': d.C_alpha,
        'C_beta': d.C_beta,
        'gamma': d.gamma,  # 流行度参数
        'in_degrees': d.in_degrees,  # 入度分布
        'network_density': d.A.nnz / (d.N * (d.N - 1)),  # 网络密度
        'total_edges': d.A.nnz,  # 总边数
        'adjacency_matrix_shape': d.A.shape,  # 邻接矩阵形状
        'feature_matrix_shape': d.X.shape,  # 特征矩阵形状
        'seed': d.seed  # 随机种子
    }

    true_beta = np.array(beta).reshape(-1)
    data_NR = Pop_NR(d)
    np.random.seed(b)  # 每个重复使用不同的随机种子
    running_beta = np.random.normal(0, 0.01, 21)
    beta_hat = data_NR.run(running_parameter=running_beta)

    beta_hat = beta_hat.reshape(-1)
    est_std = np.sqrt(np.diag(plugin(d, beta_hat)[0])).reshape(-1)
    bi_cover = np.logical_and(((beta_hat - 1.96 * est_std) < true_beta), ((beta_hat + 1.96 * est_std) > true_beta))
    print(f"{b}:{bi_cover}")

    # 返回扩展的结果，包含Pop数据和alpha估计
    return beta_hat, true_beta, est_std, bi_cover, pop_data


B = 1
Tasks = list(range(B))
N = X_or.shape[0]

# 保存Pop类配置信息
pop_config = {
    'N': N,
    'beta': beta,
    'delta': delta,
    'C_min': C_min,
    'C_max': C_max,
    'feature_source': filepath,
    'adjacency_source': adjacency_csv_file,
    'feature_stats': {
        'min': X.min(),
        'max': X.max(),
        'mean': np.mean(X, axis=0),
        'std': np.std(X, axis=0)
    }
}

for delta in [0.25]:
    d0 = Pop(N, beta, delta, C_min, C_max, external_X=X, adjacency_csv_file=adjacency_csv_file)
    print("N: ", d0.N)
    print("beta shape: ", len(beta))
    print("C_max: ", C_max)
    print("C_min: ", C_min)
    print("delta: ", delta)
    print("alpha:", d0.alpha)
    print("C_alpha:", d0.C_alpha)
    print("network nonzereos:", np.sum(d0.A))

    Results = Parallel(n_jobs=-1, backend='loky', verbose=10)(delayed(map_fun)(b) for b in range(B))

    # 提取各个组件
    estimators = np.array([est.reshape(-1) for est, _, _, _, _ in Results])  # B x (p+1)
    parameters = np.array([par.reshape(-1) for _, par, _, _, _ in Results])  # B x (p+1)
    plug_std = np.array([est_std.reshape(-1) for _, _, est_std, _, _ in Results])  # B x (p+1)
    cover = np.array([cv.reshape(-1) for _, _, _, cv, _ in Results])  # B x (p+1)
    pop_datas = [pop_data for _, _, _, _, pop_data in Results]  # 所有Pop数据

    # 保存完整结果
    full_results = {
        'estimators': estimators,
        'parameters': parameters,
        'plug_std': plug_std,
        'cover': cover,
        'pop_datas': pop_datas,
        'pop_config': pop_config,
        'alpha_estimates': [pop_data['alpha'] for pop_data in pop_datas],  # alpha估计值
        'C_alpha_estimates': [pop_data['C_alpha'] for pop_data in pop_datas],  # C_alpha估计值
        'summary_stats': {
            'N': N,
            'delta': delta,
            'total_simulations': B,
            'mean_alpha': np.mean([pop_data['alpha'] for pop_data in pop_datas]),
            'mean_C_alpha': np.mean([pop_data['C_alpha'] for pop_data in pop_datas]),
            'mean_network_density': np.mean([pop_data['network_density'] for pop_data in pop_datas])
        }
    }

    with open(f'./data/results{N}delta{int(100 * delta)}.pkl', 'wb') as f:
        pickle.dump(full_results, f)

    # 计算统计量
    covariance = (estimators.T @ estimators) / B - (
            estimators.mean(axis=0).reshape(-1, 1) @ estimators.mean(axis=0).reshape(1, -1))
    monte_std = np.sqrt(np.diag(covariance))
    monte_std_safe = np.where(monte_std == 0, 1e-12, monte_std)
    ARE = np.mean(np.abs((plug_std / monte_std_safe) - 1), axis=0)
    RMSE = np.sqrt(np.mean((estimators - parameters) ** 2, axis=0))
    std_estimation = np.mean(plug_std, axis=0)
    cover_rate = np.mean(cover, axis=0)

    print(f"N:{N};monte:", monte_std)
    print(f"N:{N};plug:", std_estimation)
    print(f"N:{N};RMSE:", RMSE)
    print(f"N:{N};ARE:", ARE)
    print(f"N:{N};cover:", cover_rate)

    # 输出alpha相关的统计信息
    print(f"Alpha估计统计:")
    print(f"  平均alpha: {full_results['summary_stats']['mean_alpha']:.6f}")
    print(f"  平均C_alpha: {full_results['summary_stats']['mean_C_alpha']:.6f}")
    print(f"  平均网络密度: {full_results['summary_stats']['mean_network_density']:.8f}")

# 结果读取和分析部分
import pickle

np.random.seed(42)
initial_beta = np.random.normal(0, 0.1, 21)
beta = initial_beta.tolist()
C_max = 25
C_min = 9
for delta in [0.25]:
    B = 1
    N = X.shape[0]
    file_path = f'./data/results{N}delta{int(delta * 100)}.pkl'

    with open(file_path, 'rb') as file:
        full_results = pickle.load(file)

        # 从完整结果中提取数据
        estimators = full_results['estimators']
        parameters = full_results['parameters']
        plug_std = full_results['plug_std']
        cover = full_results['cover']
        pop_datas = full_results['pop_datas']

        # 输出alpha相关信息
        print("\n=== Alpha估计详情 ===")
        for i, pop_data in enumerate(pop_datas):
            print(f"重复 {i}: alpha={pop_data['alpha']:.6f}, C_alpha={pop_data['C_alpha']:.6f}")
            print(
                f"  网络统计: 节点数={pop_data['N']}, 边数={pop_data['total_edges']}, 密度={pop_data['network_density']:.8f}")

        covariance = (estimators.T @ estimators) / B - (
                estimators.mean(axis=0).reshape(-1, 1) @ estimators.mean(axis=0).reshape(1, -1))
        monte_std = np.sqrt(np.diag(covariance))
        monte_std_safe = np.where(monte_std == 0, 1e-12, monte_std)
        ARE = np.mean(np.abs((plug_std / monte_std_safe) - 1), axis=0)
        RMSE = np.sqrt(np.mean((estimators - parameters) ** 2, axis=0))
        std_estimation = np.mean(plug_std, axis=0)
        cover_rate = np.mean(cover, axis=0)
        print("N: ", N)
        print("delta: ", delta)
        print(f"N:{N};RMSE:", RMSE)
        print(f"N:{N};ARE:", ARE)
        print(f"N:{N};cover:", cover_rate)