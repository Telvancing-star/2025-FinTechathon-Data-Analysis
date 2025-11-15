
import pickle
import numpy as np
import pandas as pd
from collections import deque
import os

class GetPath:
    def __init__(self, adjacency_dict, output_dir):
        self.adjacency_dict, self.output_dir = adjacency_dict, output_dir

    def compute_all_pairs_shortest_paths(self):
        """
        使用多源BFS计算所有点对之间的最短路径长度

        Args:
            adjacency_dict: 邻接字典，key为节点ID，value为邻居列表

        Returns:
            distance_matrix: 距离矩阵，dist[i][j]表示节点i到j的最短距离
            node_index_map: 节点ID到矩阵索引的映射
        """
        # 获取所有节点并排序以确保一致性
        all_nodes = sorted(self.adjacency_dict.keys())
        n = len(all_nodes)

        # 创建节点ID到矩阵索引的映射
        node_to_index = {node: idx for idx, node in enumerate(all_nodes)}
        index_to_node = {idx: node for idx, node in enumerate(all_nodes)}

        # 初始化距离矩阵为无穷大
        distance_matrix = np.full((n, n), np.inf)

        # 对每个节点执行BFS
        for source_idx, source_node in enumerate(all_nodes):
            # 初始化距离数组和访问标记
            dist_from_source = np.full(n, np.inf)
            visited = np.zeros(n, dtype=bool)

            # BFS队列：(节点索引, 距离)
            queue = deque()
            queue.append((source_idx, 0))
            visited[source_idx] = True
            dist_from_source[source_idx] = 0

            while queue:
                current_idx, current_dist = queue.popleft()

                # 遍历所有邻居
                current_node = index_to_node[current_idx]
                for neighbor_node in self.adjacency_dict[current_node]:
                    neighbor_idx = node_to_index[neighbor_node]

                    if not visited[neighbor_idx]:
                        visited[neighbor_idx] = True
                        new_dist = current_dist + 1  # 无权图，每条边权重为1
                        dist_from_source[neighbor_idx] = new_dist
                        queue.append((neighbor_idx, new_dist))

            # 将结果存入距离矩阵
            distance_matrix[source_idx, :] = dist_from_source

        return distance_matrix, node_to_index, index_to_node

    def compute_all_pairs_shortest_paths_optimize(self):
        """
        优化版本：使用更高效的BFS实现
        """
        all_nodes = sorted(self.adjacency_dict.keys())
        n = len(all_nodes)
        node_to_index = {node: idx for idx, node in enumerate(all_nodes)}

        distance_matrix = np.full((n, n), np.inf)

        for source_idx, source_node in enumerate(all_nodes):
            if source_idx % 100 == 0:  # 进度显示
                print(f"处理进度: {source_idx}/{n}")

            dist = np.full(n, np.inf)
            visited = np.zeros(n, dtype=bool)

            queue = deque([source_idx])
            visited[source_idx] = True
            dist[source_idx] = 0

            while queue:
                current_idx = queue.popleft()
                current_dist = dist[current_idx]
                current_node = all_nodes[current_idx]

                for neighbor in self.adjacency_dict[current_node]:
                    neighbor_idx = node_to_index[neighbor]
                    if not visited[neighbor_idx]:
                        visited[neighbor_idx] = True
                        dist[neighbor_idx] = current_dist + 1
                        queue.append(neighbor_idx)

            distance_matrix[source_idx] = dist

        return distance_matrix, node_to_index

    def save_distance_matrix(self, distance_matrix, node_index_map, output_path):
        """
        保存距离矩阵和节点映射
        """
        # 保存距离矩阵
        np.save(output_path, distance_matrix)

        # 保存节点映射
        mapping_path = output_path.replace('.npy', '_mapping.pkl')
        with open(mapping_path, 'wb') as f:
            pickle.dump(node_index_map, f)

        print(f"距离矩阵已保存至: {output_path}")
        print(f"节点映射已保存至: {mapping_path}")

    def load_distance_matrix(self, matrix_path, mapping_path):
        """
        加载距离矩阵和节点映射
        """
        distance_matrix = np.load(matrix_path)
        with open(mapping_path, 'rb') as f:
            node_index_map = pickle.load(f)

        return distance_matrix, node_index_map

    def get_shortest_distance(self, distance_matrix, node_index_map, node_a, node_b):
        """
        查询两个节点之间的最短距离

        Args:
            distance_matrix: 距离矩阵
            node_index_map: 节点ID到矩阵索引的映射
            node_a, node_b: 要查询的节点ID

        Returns:
            shortest_distance: 最短距离，如果不连通则返回np.inf
        """
        idx_a = node_index_map.get(node_a)
        idx_b = node_index_map.get(node_b)

        if idx_a is None or idx_b is None:
            raise ValueError(f"节点 {node_a} 或 {node_b} 不在图中")

        return distance_matrix[idx_a, idx_b]

    # 使用示例
    def main(self):
        # 计算所有点对最短路径
        print("开始计算所有点对最短路径...")
        # distance_matrix, node_index_map, index_to_node = self.compute_all_pairs_shortest_paths()
        distance_matrix, node_index_map = self.compute_all_pairs_shortest_paths_optimize()

        # 保存结果
        output_path = os.path.join(self.output_dir, "shortest_path_distances.npy")
        self.save_distance_matrix(distance_matrix, node_index_map, output_path)

        # 示例查询
        sample_nodes = list(adjacency_dict.keys())[:3]  # 取前3个节点作为示例
        if len(sample_nodes) >= 2:
            dist = Getpath.get_shortest_distance(distance_matrix, node_index_map, sample_nodes[0], sample_nodes[1])
            print(f"\n示例查询: {sample_nodes[0]} -> {sample_nodes[1]} = {dist}")

        # 验证结果
        print("\n计算完成！验证一些统计信息：")

        # 统计连通性
        connected_pairs = np.sum(distance_matrix < np.inf)
        total_pairs = distance_matrix.shape[0] * distance_matrix.shape[1]
        connectivity_ratio = connected_pairs / total_pairs

        print(f"总节点对数: {total_pairs}")
        print(f"连通节点对数: {connected_pairs}")
        print(f"连通比例: {connectivity_ratio:.4f}")

        # 统计平均最短路径长度（仅限连通对）
        connected_distances = distance_matrix[distance_matrix < np.inf]
        if len(connected_distances) > 0:
            avg_shortest_path = np.mean(connected_distances)
            print(f"平均最短路径长度: {avg_shortest_path:.4f}")


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


if __name__ == "__main__":
    # 使用示例
    file_path = './data/Social/adjacency_matrix_origin.csv'
    neighbor_file_path = "./data/Social/adj_neighbor.pkl"
    adj_dict = build_adjacency_dict_efficient(file_path)

    with open(neighbor_file_path, 'wb') as f:  # 注意是'wb'二进制写入模式
        pickle.dump(adj_dict, f)

    # 查看结果
    print(f"总节点数: {len(adj_dict)}")
    for node, neighbors in list(adj_dict.items())[:5]:  # 显示前5个节点
        print(f"节点 {node}: 邻居 {neighbors}")

    # # 文件路径
    # output_dir = "./data/Social/"
    #
    # # 确保输出目录存在
    # os.makedirs(output_dir, exist_ok=True)
    #
    # # 加载邻接字典
    # print("正在加载邻接字典...")
    # with open(neighbor_file_path, 'rb') as f:
    #     adjacency_dict = pickle.load(f)
    #
    # print(f"加载完成，共有 {len(adjacency_dict)} 个节点")
    #
    # Getpath = GetPath(adjacency_dict, output_dir)
    # Getpath.main()