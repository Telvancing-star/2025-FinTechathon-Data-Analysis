import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_graph_from_file(file_path):
    """
    从边集文件加载图数据
    """
    try:
        # 读取边数据，假设文件格式为 "源节点 目标节点"
        edges = []
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip():  # 跳过空行
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        source = int(parts[0])
                        target = int(parts[1])
                        edges.append((source, target))

        # 创建图
        G = nx.Graph()
        G.add_edges_from(edges)

        return G, edges

    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return None, None
    except Exception as e:
        print(f"读取文件时出错：{e}")
        return None, None


def visualize_graph(G, layout='spring', figsize=(12, 8), node_size=50, font_size=8):
    """
    可视化图
    """
    plt.figure(figsize=figsize)

    # 选择布局算法
    if layout == 'spring':
        pos = nx.spring_layout(G, k=1, iterations=50)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'random':
        pos = nx.random_layout(G)
    elif layout == 'shell':
        pos = nx.shell_layout(G)
    else:
        pos = nx.spring_layout(G)

    # 绘制图
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='lightblue',
                           alpha=0.7, linewidths=0.5)
    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray', width=0.5)

    # 可选：显示节点标签（对于大图可能太密集）
    if len(G.nodes()) < 100:  # 只在节点数较少时显示标签
        nx.draw_networkx_labels(G, pos, font_size=font_size, font_color='darkblue')

    plt.title(f"图可视化 (节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()})")
    plt.axis('off')  # 关闭坐标轴
    plt.tight_layout()
    plt.show()


def analyze_graph(G):
    """
    分析图的基本属性
    """
    print("=" * 50)
    print("图分析报告")
    print("=" * 50)
    print(f"节点数量: {G.number_of_nodes()}")
    print(f"边数量: {G.number_of_edges()}")
    print(f"图是否连通: {nx.is_connected(G)}")

    if not nx.is_connected(G):
        print(f"连通分量数量: {nx.number_connected_components(G)}")
        # 显示最大的连通分量
        largest_cc = max(nx.connected_components(G), key=len)
        print(f"最大连通分量大小: {len(largest_cc)}")

    print(f"平均度: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    print(f"图密度: {nx.density(G):.6f}")

    # 度分布
    degrees = [deg for node, deg in G.degree()]
    print(f"最大度: {max(degrees)}")
    print(f"最小度: {min(degrees)}")


def visualize_degree_distribution(G):
    """
    可视化度分布
    """
    degrees = [deg for node, deg in G.degree()]

    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('度')
    plt.ylabel('频率')
    plt.title('度分布')
    plt.grid(True, alpha=0.3)
    plt.show()


def main():
    # 文件路径
    file_path = "./data/facebook_combined.txt"

    # 加载图数据
    print("正在加载图数据...")
    G, edges = load_graph_from_file(file_path)

    if G is None:
        print("无法加载图数据")
        return

    # 分析图
    analyze_graph(G)

    # 可视化整个图
    print("\n正在生成图可视化...")
    visualize_graph(G, layout='spring', figsize=(14, 10), node_size=30)

    # 可视化度分布
    print("正在生成度分布图...")
    visualize_degree_distribution(G)

    # 如果图不连通，可以选择只可视化最大连通分量
    if not nx.is_connected(G):
        print("\n图不连通，正在可视化最大连通分量...")
        # 找到最大连通分量
        largest_cc = max(nx.connected_components(G), key=len)
        G_largest = G.subgraph(largest_cc).copy()

        print(f"最大连通分量 - 节点数: {G_largest.number_of_nodes()}, 边数: {G_largest.number_of_edges()}")
        visualize_graph(G_largest, layout='spring', figsize=(12, 8), node_size=50)


def advanced_visualization(G, save_path=None):
    """
    高级可视化选项
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Spring layout
    pos_spring = nx.spring_layout(G, k=1, iterations=50)
    nx.draw_networkx_nodes(G, pos_spring, ax=ax1, node_size=20, node_color='red', alpha=0.6)
    nx.draw_networkx_edges(G, pos_spring, ax=ax1, alpha=0.3, edge_color='gray')
    ax1.set_title('Spring Layout')
    ax1.axis('off')

    # 2. Circular layout
    pos_circular = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos_circular, ax=ax2, node_size=20, node_color='blue', alpha=0.6)
    nx.draw_networkx_edges(G, pos_circular, ax=ax2, alpha=0.3, edge_color='gray')
    ax2.set_title('Circular Layout')
    ax2.axis('off')

    # 3. 根据度的节点大小
    node_sizes = [deg * 10 for _, deg in G.degree()]
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax3, node_size=node_sizes, node_color='green', alpha=0.6)
    nx.draw_networkx_edges(G, pos, ax=ax3, alpha=0.2, edge_color='gray')
    ax3.set_title('节点大小与度成正比')
    ax3.axis('off')

    # 4. 度分布直方图
    degrees = [deg for _, deg in G.degree()]
    ax4.hist(degrees, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax4.set_xlabel('度')
    ax4.set_ylabel('频率')
    ax4.set_title('度分布')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")

    plt.show()


# 在main函数中添加这个调用
# advanced_visualization(G, save_path="./graph_visualization.png")


if __name__ == "__main__":
    main()
