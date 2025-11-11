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

from joblib import Parallel, delayed
from tqdm import tqdm

np.random.seed(42)
initial_beta = np.random.normal(0, 0.01, 21)  # 21个特征
beta = initial_beta.tolist()  # 转换为list格式

filepath = 'data/Social/compressed_features_expanded.csv'
adjacency_csv_file = './data/Social/adjacency_matrix_origin.csv'
df = pd.read_csv(filepath)

feature_cols = [f'feature_{i}' for i in range(1, 22)]
your_X = df[feature_cols].values
print(f"特征矩阵形状: {your_X.shape}")

C_max = 25
C_min = 9


def map_fun(b):
    d = Pop(N, beta, delta, C_min, C_max, N + b + int(10000 * delta), external_X=your_X, adjacency_csv_file=adjacency_csv_file)
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


B = 1
Tasks = list(range(B))
N = your_X.shape[0]
for delta in [0.25]:
    d0 = Pop(N, beta, delta, C_min, C_max, external_X=your_X, adjacency_csv_file=adjacency_csv_file)
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
    B = 1
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
        # print(f"N:{N};monte:", monte_std)
        # print(f"N:{N};plug:", std_estimation)
        print(f"N:{N};RMSE:", RMSE)
        print(f"N:{N};ARE:", ARE)
        print(f"N:{N};cover:", cover_rate)