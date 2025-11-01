import networkx as nx
import os, pickle

def compare_edges(file_edges, graph_edges):
    """
    比较两个边列表
    """
    file_set = set(file_edges)
    graph_set = set(graph_edges)

    missing_edges = list(file_set - graph_set)  # 在文件中但不在图中
    extra_edges = list(graph_set - file_set)  # 在图中但不在文件中

    return missing_edges, extra_edges

def load_edges_from_txt(filename):
    """
    从txt文件加载边列表
    """
    edges = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('['):  # 跳过空行和文件头
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        edge = (int(parts[0]), int(parts[1]))
                        edges.append(edge)
                    except ValueError:
                        continue  # 跳过非数字行
    return edges

# def load_combined_network(self, file_path=None):
#     """加载之前保存的合并网络"""
#     if file_path is None:
#         file_path = f"{self.output_dir}/combined_network.pkl"
#
#     if os.path.exists(file_path):
#         print(f"正在从 {file_path} 加载网络数据...")
#         from collections import defaultdict
#         with open(file_path, 'rb') as f:
#             data = pickle.load(f)
#             self.combined_graph = data['graph']
#             self.node_to_egos = defaultdict(set, data['node_to_egos'])
#             self.circle_info = defaultdict(set, data['circle_info'])
#             self.processed_ego_nodes = set(data['processed_ego_nodes'])
#         print("网络数据加载完成！")
#         return True
#     else:
#         print(f"错误: 找不到文件 {file_path}")
#         return False

def load_edges_from_txt(filename):
    """
    从txt文件加载边列表（无向图处理）
    """
    edges = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('['):  # 跳过空行和文件头
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        # 对无向图，统一存储为 (min, max) 格式
                        node1, node2 = int(parts[0]), int(parts[1])
                        edge = (min(node1, node2), max(node1, node2))
                        edges.append(edge)
                    except ValueError:
                        continue  # 跳过非数字行
    return edges

def graph_edges_to_normalized(G):
    """
    将图的边列表也转换为统一格式
    """
    edges = []
    for edge in G.edges():
        # 对无向图，统一存储为 (min, max) 格式
        node1, node2 = edge
        normalized_edge = (min(node1, node2), max(node1, node2))
        edges.append(normalized_edge)
    return edges

# 主比较函数
def main_comparison():
    # file_edges = load_edges_from_txt('data/facebook_combined.txt')

    # 假设 G 是你的合并图
    print(f"正在从 {"./data/combined_network.pkl"} 加载网络数据...")
    from collections import defaultdict
    with open("./data/combined_network.pkl", 'rb') as f:
        data = pickle.load(f)
        G = data['graph']
    print("网络数据加载完成！")

    # graph_edges = list(G.edges())
    file_edges = load_edges_from_txt('./data/facebook_combined.txt')
    graph_edges = graph_edges_to_normalized(G)

    print(f"参考文件边数: {len(file_edges)}")
    print(f"你的图边数: {len(graph_edges)}")

    missing_edges, extra_edges = compare_edges(file_edges, graph_edges)

    print(f"\n缺失的边 ({len(missing_edges)} 条):")
    for edge in sorted(missing_edges):
        print(f"{edge[0]} {edge[1]}")

    print(f"\n多余的边 ({len(extra_edges)} 条):")
    for edge in sorted(extra_edges):
        print(f"{edge[0]} {edge[1]}")

    # 验证结果
    if len(missing_edges) == 0 and len(extra_edges) == 0:
        print("\n✓ 两个图的边完全一致！")
    else:
        print("\n✗ 两个图的边不一致")


# 运行比较
if __name__ == "__main__":
    main_comparison()