import pickle
import pandas as pd
import numpy as np
import os


def find_node_id_column(df, sample_nodes):
    """
    自动检测包含节点ID的列

    Args:
        df: 数据框
        sample_nodes: 已知的节点ID样本

    Returns:
        column_name: 检测到的列名
    """
    sample_nodes_set = set(sample_nodes)

    for column in df.columns:
        # 检查该列是否包含样本节点
        column_values_set = set(df[column].unique())
        if sample_nodes_set.issubset(column_values_set):
            return column

    # 如果没有完全匹配，找包含最多样本节点的列
    best_column = None
    max_overlap = 0

    for column in df.columns:
        column_values_set = set(df[column].unique())
        overlap = len(sample_nodes_set.intersection(column_values_set))
        if overlap > max_overlap:
            max_overlap = overlap
            best_column = column

    return best_column


def get_node_neighbors_investment_auto(node_id, neighbor_file_path, investment_file_path, output_dir,
                                       node_id_column=None):
    """
    自动检测节点ID列版本的函数
    """
    # 1. 加载邻接字典
    print("正在加载邻接字典...")
    with open(neighbor_file_path, 'rb') as f:
        adj_dict = pickle.load(f)

    if node_id not in adj_dict:
        raise ValueError(f"节点 {node_id} 不在邻接字典中")

    neighbors = adj_dict[node_id]
    target_nodes = [node_id] + neighbors

    print(f"目标节点: {node_id}")
    print(f"邻居数量: {len(neighbors)}")
    print(f"总节点数: {len(target_nodes)}")

    # 2. 加载投资记录
    print("正在加载投资记录...")
    investment_df = pd.read_csv(investment_file_path, encoding="gb18030")

    # 3. 自动检测节点ID列（如果未指定）
    if node_id_column is None:
        print("自动检测节点ID列...")
        # 使用目标节点和部分邻居作为样本
        sample_nodes = [node_id] + neighbors[:10]  # 取前10个邻居作为样本
        detected_column = find_node_id_column(investment_df, sample_nodes)

        if detected_column is None:
            print(f"投资记录文件的列名: {investment_df.columns.tolist()}")
            print(f"尝试查找的节点样本: {sample_nodes[:5]}...")  # 只显示前5个
            raise ValueError("无法自动检测到节点ID列，请手动指定")
        else:
            node_id_column = detected_column
            print(f"自动检测到节点ID列: {node_id_column}")

    # 验证列名
    if node_id_column not in investment_df.columns:
        print(f"投资记录文件的列名: {investment_df.columns.tolist()}")
        raise ValueError(f"指定的节点ID列名 '{node_id_column}' 不存在")

    # 4. 筛选数据
    filtered_df = investment_df[investment_df[node_id_column].isin(target_nodes)]

    # 5. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, f"node_{node_id}_neighbors_investment.csv")
    filtered_df.to_csv(output_file_path, index=False)

    # 6. 输出统计信息
    print(f"\n处理完成!")
    print(f"找到的投资记录数量: {len(filtered_df)}")
    print(f"原投资记录总数: {len(investment_df)}")
    print(f"输出文件: {output_file_path}")

    node_counts = filtered_df[node_id_column].value_counts()
    print(f"\n各节点的投资记录数量:")
    for node, count in node_counts.items():
        node_type = "目标节点" if node == node_id else "邻居节点"
        print(f"  {node} ({node_type}): {count} 条记录")

    return output_file_path


# 使用增强版
if __name__ == "__main__":
    neighbor_file_path = "../data/Social/adj_neighbor.pkl"
    investment_file_path = "../data/cluster_with_rounds.csv"
    output_dir = "./data/node_neighbors_investment"
    target_node = 0  # 替换为您的目标节点

    try:
        # 使用自动检测版本
        output_file = get_node_neighbors_investment_auto(
            node_id=target_node,
            neighbor_file_path=neighbor_file_path,
            investment_file_path=investment_file_path,
            output_dir=output_dir
            # node_id_column=None  # 设为None自动检测，或手动指定如 'node_id'
        )
        print(f"\n成功生成文件: {output_file}")

    except Exception as e:
        print(f"处理过程中出错: {e}")