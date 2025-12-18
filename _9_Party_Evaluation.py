import numpy as np
from numpy.linalg import eig

# A 组矩阵（政绩、选举、意识形态、内部稳定、透明度）
A_matrix = np.array([
    [1,   1,   3,   5,   6],
    [1,   1,   3,   5,   6],
    [1/3, 1/3, 1,   3,   4],
    [1/5, 1/5, 1/3, 1,   2],
    [1/6, 1/6, 1/4, 1/2, 1]
])

# B 组矩阵（腐败、路线急转、政策失败、候选人争议、势头下滑）
B_matrix = np.array([
    [1,   4,   3,   5,   3],
    [1/4, 1,   1/3, 2,   1/3],
    [1/3, 3,   1,   3,   1],
    [1/5, 1/2, 1/3, 1,   1/3],
    [1/3, 3,   1,   3,   1]
])

# C 组矩阵（政策贴近、新领导、势头上升、清廉透明、地方治理）
C_matrix = np.array([
    [1,   5,   3,   3,   4],
    [1/5, 1,   1/3, 1/3, 1/3],
    [1/3, 3,   1,   2,   3],
    [1/3, 3,   1/2, 1,   2],
    [1/4, 3,   1/3, 1/2, 1]
])

# D 组矩阵（意识形态距离、选举弱、内讧、不透明、极端主义争议）
D_matrix = np.array([
    [1,   1,   2,   3,   4],
    [1,   1,   2,   3,   4],
    [1/2, 1/2, 1,   2,   3],
    [1/3, 1/3, 1/2, 1,   2],
    [1/4, 1/4, 1/3, 1/2, 1]
])

############################################################
# 1. AHP 权重计算函数：求最大特征值对应的特征向量
############################################################
def ahp_weights(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = eig(matrix)
    max_index = np.argmax(eigenvalues.real)
    principal_vec = eigenvectors[:, max_index].real
    principal_vec = np.abs(principal_vec)   # 防止负方向
    return principal_vec / principal_vec.sum()


############################################################
# 2. 信号与权重加权计算
############################################################
def compute_weighted_score(signals, weights):
    return float(np.dot(signals, weights))


############################################################
# 3. 最终 target score 计算
############################################################
def compute_target_scores(A, B, C, D, A_mat=A_matrix, B_mat=B_matrix, C_mat=C_matrix, D_mat=D_matrix):

    # === Step 1: AHP 权重 ===
    wA = ahp_weights(A_mat)
    wB = ahp_weights(B_mat)
    wC = ahp_weights(C_mat)
    wD = ahp_weights(D_mat)

    # === Step 2: 信号评分 ===
    scoreA = compute_weighted_score(A, wA)
    scoreB = compute_weighted_score(B, wB)
    scoreC = compute_weighted_score(C, wC)
    scoreD = compute_weighted_score(D, wD)

    # === Step 3: 合成到最终的 target scores（保持在 -1,1） ===
    target0 = max(min(scoreA + scoreB, 1.0), -1.0)
    target1 = max(min(scoreC + scoreD, 1.0), -1.0)

    return {
        "weights": {
            "A": wA,
            "B": wB,
            "C": wC,
            "D": wD,
        },
        "scores": {
            "target_score_0": target0,
            "target_score_1": target1,
        }
    }, [target0, target1]


if __name__=="__main__":
    print("Test")
