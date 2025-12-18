import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def analyze_graph_from_csv(csv_file_path):
    """
    从CSV文件读取邻接矩阵并分析图的基本性质
    """
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_file_path, index_col=0)
        print("成功读取CSV文件")
        print(f"邻接矩阵形状: {df.shape}")
    except Exception as e:
        print(f"读取文件错误: {e}")
        return

    # 转换为numpy数组
    adj_matrix = df.values

    # 创建图对象
    G = nx.from_numpy_array(adj_matrix)

    # 基本图性质分析
    print("\n=== 图基本性质分析 ===")
    print(f"节点数量: {G.number_of_nodes()}")
    print(f"边数量: {G.number_of_edges()}")

    # 度分析
    degrees = [d for n, d in G.degree()]
    print(f"最小度: {min(degrees)}")
    print(f"最大度: {max(degrees)}")
    print(f"平均度: {np.mean(degrees):.2f}")
    print(f"度标准差: {np.std(degrees):.2f}")

    # 获取最大度节点的编号
    max_degree = max(degrees)
    max_degree_nodes = [node for node, degree in G.degree() if degree == max_degree]
    print(f"最大度节点编号: {max_degree_nodes}")

    # 网络密度
    density = nx.density(G)
    print(f"网络密度: {density:.4f}")

    # 连通性
    print(f"是否连通图: {nx.is_connected(G)}")

    # 如果是非连通图，显示连通分量信息
    if not nx.is_connected(G):
        connected_components = list(nx.connected_components(G))
        print(f"连通分量数量: {len(connected_components)}")
        print(f"最大连通分量大小: {max(len(cc) for cc in connected_components)}")

    return G, adj_matrix


def plot_degree_distribution(G, save_path=None):
    """
    绘制度分布图
    """
    degrees = [d for n, d in G.degree()]

    plt.figure(figsize=(12, 4))

    # 度分布直方图
    plt.subplot(1, 2, 1)
    plt.hist(degrees, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('度')
    plt.ylabel('频数')
    plt.title('度分布直方图')
    plt.grid(True, alpha=0.3)

    # 度分布箱线图
    plt.subplot(1, 2, 2)
    plt.boxplot(degrees)
    plt.ylabel('度')
    plt.title('度分布箱线图')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"度分布图已保存到: {save_path}")

    plt.show()


def plot_network_graph(G, save_path=None):
    """
    绘制网络图并保存
    """
    plt.figure(figsize=(10, 8))

    # 使用spring布局
    pos = nx.spring_layout(G, k=1, iterations=50)

    # 绘制节点和边
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='lightblue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')

    # 可选：添加节点标签（对于大图可能太密集）
    # nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(f'网络结构图\n(节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()})')
    plt.axis('off')

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"网络图已保存到: {save_path}")

    plt.show()


def plot_degree_rank(G, save_path=None):
    """
    绘制度排序图
    """
    degrees = [d for n, d in G.degree()]
    sorted_degrees = sorted(degrees, reverse=True)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(sorted_degrees) + 1), sorted_degrees, 'o-', linewidth=1, markersize=3)
    plt.xlabel('节点排序')
    plt.ylabel('度')
    plt.title('节点度排序图')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 可选：使用对数坐标
    plt.xscale('log')  # 可选：使用对数坐标

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"度排序图已保存到: {save_path}")

    plt.show()


def save_analysis_results(G, output_dir='./data/network_analysis'):
    """
    保存所有分析结果和图片
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n=== 保存分析结果到: {output_dir} ===")

    # 1. 保存度分布图
    degree_dist_path = os.path.join(output_dir, 'degree_distribution.png')
    plot_degree_distribution(G, save_path=degree_dist_path)

    # 2. 保存网络图
    network_path = os.path.join(output_dir, 'network_graph.png')
    plot_network_graph(G, save_path=network_path)

    # 3. 保存度排序图
    degree_rank_path = os.path.join(output_dir, 'degree_rank.png')
    plot_degree_rank(G, save_path=degree_rank_path)

    # 4. 保存基本统计信息到文本文件
    stats_path = os.path.join(output_dir, 'network_statistics.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("网络基本性质分析结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"节点数量: {G.number_of_nodes()}\n")
        f.write(f"边数量: {G.number_of_edges()}\n")

        degrees = [d for n, d in G.degree()]
        f.write(f"最小度: {min(degrees)}\n")
        f.write(f"最大度: {max(degrees)}\n")
        f.write(f"平均度: {np.mean(degrees):.2f}\n")
        f.write(f"度标准差: {np.std(degrees):.2f}\n")
        f.write(f"网络密度: {nx.density(G):.4f}\n")
        f.write(f"是否连通图: {nx.is_connected(G)}\n")

        if not nx.is_connected(G):
            connected_components = list(nx.connected_components(G))
            f.write(f"连通分量数量: {len(connected_components)}\n")
            f.write(f"最大连通分量大小: {max(len(cc) for cc in connected_components)}\n")

    print(f"统计信息已保存到: {stats_path}")
    print("所有分析完成！")


# 使用示例
if __name__ == "__main__":
    # 替换为您的CSV文件路径
    csv_file = "../data/Social/adjacency_matrix_origin.csv"

    # 分析图
    G, adj_matrix = analyze_graph_from_csv(csv_file)

    # 可选：单独绘制度分布图
    # plot_degree_distribution(G)

    # 保存所有分析结果和图片
    save_analysis_results(G)