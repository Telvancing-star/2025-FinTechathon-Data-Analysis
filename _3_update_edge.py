# 更新边数据
import json
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def expand_graph_with_duplicate_nodes(edge_file, duplicates_file, output_file):
    """
    扩展图：为重复节点创建新节点并复制相关的边
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

    return graph, original_to_new_ids, new_edges


def visualize_new_edges(expanded_graph, original_to_new_ids, new_edges, save_path=None):
    """
    可视化新增的边，相同节点的副本边使用相同颜色，不同节点使用不同颜色
    """
    plt.figure(figsize=(15, 10))

    # 创建子图只包含新增的边和相关的节点
    visualization_graph = nx.Graph()

    # 添加所有与新边相关的节点
    for edge in new_edges:
        visualization_graph.add_edge(edge[0], edge[1])

    # 为每个原始节点分配一个独特的颜色
    original_nodes = list(original_to_new_ids.keys())
    n_colors = len(original_nodes)

    # 使用rainbow色系为不同原始节点分配颜色
    color_map = plt.cm.rainbow(np.linspace(0, 1, n_colors))
    node_color_mapping = {}
    edge_color_mapping = {}

    for i, original_node in enumerate(original_nodes):
        color = color_map[i]
        # 为原始节点和所有副本节点分配相同颜色
        node_color_mapping[original_node] = color
        for new_node in original_to_new_ids[original_node]:
            node_color_mapping[new_node] = color

        # 为这个原始节点相关的所有新边分配相同颜色
        for edge in new_edges:
            if original_node in edge:
                edge_color_mapping[edge] = color

    # 准备节点颜色列表
    node_colors = []
    for node in visualization_graph.nodes():
        if node in node_color_mapping:
            node_colors.append(node_color_mapping[node])
        else:
            # 对于非重复节点的邻居，使用灰色
            node_colors.append('lightgray')

    # 准备边颜色列表
    edge_colors = []
    for edge in visualization_graph.edges():
        if edge in edge_color_mapping:
            edge_colors.append(edge_color_mapping[edge])
        elif (edge[1], edge[0]) in edge_color_mapping:  # 无向图边可能顺序相反
            edge_colors.append(edge_color_mapping[(edge[1], edge[0])])
        else:
            edge_colors.append('lightgray')

    # 使用spring布局
    pos = nx.spring_layout(visualization_graph, k=1, iterations=50, seed=42)

    # 绘制节点
    nx.draw_networkx_nodes(visualization_graph, pos,
                           node_color=node_colors,
                           node_size=100,
                           alpha=0.8)

    # 绘制边 - 突出显示新增的边
    nx.draw_networkx_edges(visualization_graph, pos,
                           edge_color=edge_colors,
                           width=2,
                           alpha=0.7)

    # 创建图例
    legend_elements = []
    for i, original_node in enumerate(original_nodes[:10]):  # 只显示前10个以免图例太拥挤
        color = color_map[i]
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=color, markersize=8,
                                          label=f'节点 {original_node} 的副本'))

    if len(original_nodes) > 10:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor='gray', markersize=8,
                                          label=f'... 还有 {len(original_nodes) - 10} 个节点'))

    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))

    plt.title(f'新增边可视化\n(共 {len(new_edges)} 条新增边，涉及 {len(original_nodes)} 个原始节点的副本)')
    plt.axis('off')

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"新增边可视化图已保存到: {save_path}")

    plt.show()


def create_simple_visualization(new_edges, original_to_new_ids, save_path=None):
    """
    创建简化的可视化，只关注新增边的模式
    """
    plt.figure(figsize=(12, 8))

    # 创建简化的图
    simple_graph = nx.Graph()

    # 为每个原始节点创建一个代表性节点
    representative_nodes = {}
    for original_node in original_to_new_ids.keys():
        representative_nodes[original_node] = f"O{original_node}"

    # 添加边到简化图
    for edge in new_edges:
        # 找到这条边属于哪个原始节点
        source_original = None
        for original_node, new_nodes in original_to_new_ids.items():
            if edge[0] in new_nodes or edge[0] == original_node:
                source_original = original_node
                break

        if source_original is not None:
            simple_graph.add_edge(representative_nodes[source_original], f"T{edge[1]}")

    # 为节点分配颜色
    original_nodes = list(original_to_new_ids.keys())
    n_colors = len(original_nodes)
    color_map = plt.cm.rainbow(np.linspace(0, 1, n_colors))

    node_colors = []
    for node in simple_graph.nodes():
        if node.startswith('O'):
            original_id = int(node[1:])
            if original_id in original_nodes:
                idx = original_nodes.index(original_id)
                node_colors.append(color_map[idx])
            else:
                node_colors.append('lightgray')
        else:
            node_colors.append('lightblue')

    # 绘制简化图
    pos = nx.spring_layout(simple_graph, seed=42)

    nx.draw_networkx_nodes(simple_graph, pos, node_color=node_colors,
                           node_size=300, alpha=0.8)
    nx.draw_networkx_edges(simple_graph, pos, alpha=0.6, edge_color='gray')
    nx.draw_networkx_labels(simple_graph, pos, font_size=8)

    plt.title(f'新增边模式可视化\n(显示 {len(original_nodes)} 个原始节点与其目标节点的连接)')
    plt.axis('off')

    if save_path:
        simple_save_path = save_path.replace('.png', '_simple.png')
        plt.savefig(simple_save_path, dpi=300, bbox_inches='tight')
        print(f"简化可视化图已保存到: {simple_save_path}")

    plt.show()


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
    expanded_graph, node_mapping, new_edges = expand_graph_with_duplicate_nodes(
        edge_file, duplicates_file, output_file
    )

    # 验证结果
    verify_expansion(edge_file, output_file, duplicates_file)

    # 可视化新增的边
    visualize_new_edges(expanded_graph, node_mapping, new_edges,
                        save_path="data/Social/new_edges_visualization.png")

    # 创建简化可视化
    create_simple_visualization(new_edges, node_mapping,
                                save_path="data/Social/new_edges_visualization.png")

    # 打印一些统计信息
    print("\n扩展统计:")
    for original_id, new_ids in list(node_mapping.items())[:5]:  # 只显示前5个
        print(f"节点 {original_id} -> 新节点 {new_ids} (新增 {len(new_ids)} 条边)")

    if len(node_mapping) > 5:
        print(f"... 还有 {len(node_mapping) - 5} 个节点的映射")