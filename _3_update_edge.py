# 更新边数据
import json
import networkx as nx
from collections import defaultdict


def expand_graph_with_duplicate_nodes(edge_file, duplicates_file, output_file):
    """
    扩展图：为重复节点创建新节点并复制相关的边

    参数:
    edge_file: 原始边文件路径
    duplicates_file: 重复节点信息JSON文件路径
    output_file: 输出文件路径
    """

    # 读取重复节点信息
    with open(duplicates_file, 'r') as f:
        duplicates = json.load(f)

    print(f"找到 {len(duplicates)} 个需要复制的节点")

    # 读取原始图
    graph = nx.Graph()
    max_node_id = 0

    with open(edge_file, 'r') as f:
        for line in f:
            if line.strip():
                node1, node2 = map(int, line.strip().split())
                graph.add_edge(node1, node2)
                max_node_id = max(max_node_id, node1, node2)

    print(f"原始图: {graph.number_of_nodes()} 个节点, {graph.number_of_edges()} 条边")
    print(f"最大节点ID: {max_node_id}")

    # 为每个重复节点创建新节点
    new_edges = []
    next_new_id = max_node_id + 1

    # 记录每个原始节点对应的新节点ID
    original_to_new_ids = defaultdict(list)

    for original_id, count in duplicates.items():
        original_id = int(original_id)
        # 为每个重复创建新节点
        for i in range(count - 1):  # 原始节点保留，所以创建count-1个新节点
            new_id = next_new_id
            original_to_new_ids[original_id].append(new_id)
            next_new_id += 1

    print(f"将添加 {next_new_id - max_node_id - 1} 个新节点")

    # 收集所有需要处理的边
    edges_to_process = []

    # 找到所有与重复节点相关的边
    for original_id, new_ids in original_to_new_ids.items():
        if original_id in graph:
            # 找到所有与原始节点相连的边
            neighbors = list(graph.neighbors(original_id))
            for neighbor in neighbors:
                edges_to_process.append((original_id, neighbor))

    # 添加新边
    for node1, node2 in edges_to_process:
        # 如果node1是重复节点，为它的每个新副本创建边
        if node1 in original_to_new_ids:
            for new_id in original_to_new_ids[node1]:
                new_edges.append((new_id, node2))
                graph.add_edge(new_id, node2)

        # 如果node2是重复节点，为它的每个新副本创建边
        if node2 in original_to_new_ids:
            for new_id in original_to_new_ids[node2]:
                new_edges.append((node1, new_id))
                graph.add_edge(node1, new_id)

    print(f"添加了 {len(new_edges)} 条新边")
    print(f"扩展后图: {graph.number_of_nodes()} 个节点, {graph.number_of_edges()} 条边")

    # 保存扩展后的图
    with open(output_file, 'w') as f:
        for edge in graph.edges():
            f.write(f"{edge[0]} {edge[1]}\n")

    # 保存新节点映射信息（可选，用于调试）
    mapping_file = output_file.replace('.txt', '_mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump(dict(original_to_new_ids), f, indent=2)

    print(f"扩展后的图已保存到: {output_file}")
    print(f"节点映射信息已保存到: {mapping_file}")

    return graph, original_to_new_ids


def verify_expansion(original_file, expanded_file, duplicates_file):
    """
    验证扩展结果
    """
    # 读取原始图
    original_graph = nx.Graph()
    with open(original_file, 'r') as f:
        for line in f:
            if line.strip():
                node1, node2 = map(int, line.strip().split())
                original_graph.add_edge(node1, node2)

    # 读取扩展图
    expanded_graph = nx.Graph()
    with open(expanded_file, 'r') as f:
        for line in f:
            if line.strip():
                node1, node2 = map(int, line.strip().split())
                expanded_graph.add_edge(node1, node2)

    # 读取重复节点信息
    with open(duplicates_file, 'r') as f:
        duplicates = json.load(f)

    print("\n验证结果:")
    print(f"原始图: {original_graph.number_of_nodes()} 节点, {original_graph.number_of_edges()} 边")
    print(f"扩展图: {expanded_graph.number_of_nodes()} 节点, {expanded_graph.number_of_edges()} 边")

    # 计算应该添加的新节点数量
    expected_new_nodes = sum(count - 1 for count in duplicates.values())
    actual_new_nodes = expanded_graph.number_of_nodes() - original_graph.number_of_nodes()

    print(f"预期新节点数: {expected_new_nodes}, 实际新节点数: {actual_new_nodes}")

    if expected_new_nodes == actual_new_nodes:
        print("✓ 节点数量验证通过")
    else:
        print("✗ 节点数量验证失败")


if __name__ == "__main__":
    # 文件路径
    edge_file = "data/Social/facebook_combined.txt"
    duplicates_file = "data/Social/multiple_occurrences.json"
    output_file = "data/Social/facebook_combined_expanded.txt"

    # 执行图扩展
    expanded_graph, node_mapping = expand_graph_with_duplicate_nodes(
        edge_file, duplicates_file, output_file
    )

    # 验证结果
    verify_expansion(edge_file, output_file, duplicates_file)

    # 打印一些统计信息
    print("\n扩展统计:")
    for original_id, new_ids in list(node_mapping.items())[:5]:  # 只显示前5个
        print(f"节点 {original_id} -> 新节点 {new_ids}")

    if len(node_mapping) > 5:
        print(f"... 还有 {len(node_mapping) - 5} 个节点的映射")