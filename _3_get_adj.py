import pandas as pd
import numpy as np
import networkx as nx


def edges_to_adjacency_matrix_networkx(edges_file, output_csv=None):
    """
    从边文件生成邻接矩阵（使用NetworkX）

    参数:
    edges_file: 边数据文件路径
    output_csv: 输出的CSV文件路径
    """
    # 读取边数据
    edges = []
    node_set = set()

    with open(edges_file, 'r') as f:
        for line in f:
            # 处理每行数据，假设格式为 "源节点 目标节点" 或 "源节点 目标节点 权重"
            parts = line.strip().split()
            if len(parts) >= 2:
                source = parts[0]
                target = parts[1]
                weight = float(parts[2]) if len(parts) >= 3 else 1.0

                edges.append((source, target, weight))
                node_set.add(source)
                node_set.add(target)

    print(f"读取到 {len(edges)} 条边")
    print(f"发现 {len(node_set)} 个唯一节点")

    # 创建图
    G = nx.Graph()

    # 添加带权重的边
    for source, target, weight in edges:
        G.add_edge(source, target, weight=weight)

    # 获取排序后的节点列表
    nodes = sorted(node_set)
    n = len(nodes)

    # 创建节点到索引的映射
    node_to_index = {node: i for i, node in enumerate(nodes)}

    # 创建邻接矩阵
    adj_matrix = np.zeros((n, n))

    for source, target, weight in edges:
        i = node_to_index[source]
        j = node_to_index[target]
        adj_matrix[i, j] = weight
        # adj_matrix[j, i] = weight  # 无向图，对称矩阵

    # 创建DataFrame
    df_adj = pd.DataFrame(adj_matrix, index=nodes, columns=nodes)

    # 保存到CSV
    if output_csv:
        df_adj.to_csv(output_csv)
        print(f"邻接矩阵已保存到: {output_csv}")

    return df_adj, G


# 使用示例
df_adj, G = edges_to_adjacency_matrix_networkx('./data/Social/facebook_combined_expanded.txt', './data/Social/adjacency_matrix_origin.csv')