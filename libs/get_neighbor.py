import pandas as pd
import numpy as np
import pickle


def build_adjacency_dict_efficient(csv_file_path):
    """
    高效版本：使用布尔索引
    """
    # 读取邻接矩阵
    adj_matrix = pd.read_csv(csv_file_path, index_col=0)
    node_ids = adj_matrix.index.tolist()
    matrix_values = adj_matrix.values

    # 创建无向图版本
    undirected_matrix = np.maximum(matrix_values, matrix_values.T)

    adjacency_dict = {}

    for i, node_i in enumerate(node_ids):
        # 使用布尔索引直接找到所有邻居
        neighbor_indices = np.where(undirected_matrix[i, :] != 0)[0]
        neighbors = [node_ids[j] for j in neighbor_indices if j != i]  # 排除自身
        adjacency_dict[node_i] = neighbors

    return adjacency_dict


# 使用示例
file_path = '../data/Social/adjacency_matrix_origin.csv'
adj_dict = build_adjacency_dict_efficient(file_path)

with open('../data/Social/adj_neighbor.pkl', 'wb') as f:  # 注意是'rb'二进制读取模式
    pickle.dump(adj_dict, f)

# 查看结果
print(f"总节点数: {len(adj_dict)}")
for node, neighbors in list(adj_dict.items())[:5]:  # 显示前5个节点
    print(f"节点 {node}: 邻居 {neighbors}")