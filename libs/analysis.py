import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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


def plot_degree_distribution(G):
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
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 替换为您的CSV文件路径
    csv_file = "../data/Social/adjacency_matrix_origin.csv"

    # 分析图
    G, adj_matrix = analyze_graph_from_csv(csv_file)

    # 绘制度分布
    plot_degree_distribution(G)