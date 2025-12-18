import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def read_facebook_data(data_dir, ego_id):
    """
    读取Facebook ego-network数据
    根据论文描述，ego-network只包含好友之间的连接，不包括ego自己
    """
    # 构建文件路径
    edges_file = os.path.join(data_dir, f"{ego_id}.edges")
    circles_file = os.path.join(data_dir, f"{ego_id}.circles")
    feat_file = os.path.join(data_dir, f"{ego_id}.feat")
    egofeat_file = os.path.join(data_dir, f"{ego_id}.egofeat")
    featnames_file = os.path.join(data_dir, f"{ego_id}.featnames")

    print(f"正在处理ego用户 {ego_id}...")

    # 1. 读取边数据 - 只包含好友之间的连接
    G = nx.Graph()
    if os.path.exists(edges_file):
        with open(edges_file, 'r') as f:
            for line in f:
                node1, node2 = map(int, line.strip().split())
                G.add_edge(node1, node2)
        print(f"  边数据: {G.number_of_edges()} 条边")
    else:
        print(f"  警告: 未找到边文件 {edges_file}")

    # 2. 读取圈子数据
    circles = {}
    if os.path.exists(circles_file):
        with open(circles_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    circle_id = parts[0]
                    circle_members = list(map(int, parts[1:]))
                    circles[circle_id] = circle_members
        print(f"  圈子数据: {len(circles)} 个圈子")

    # 3. 读取特征数据
    node_features = {}
    if os.path.exists(feat_file):
        with open(feat_file, 'r') as f:
            for line in f:
                parts = list(map(int, line.strip().split()))
                if parts:
                    node_id = parts[0]
                    features = parts[1:]
                    node_features[node_id] = features
        print(f"  节点特征: {len(node_features)} 个节点")

    # 4. 读取ego特征
    ego_features = None
    if os.path.exists(egofeat_file):
        with open(egofeat_file, 'r') as f:
            line = f.readline()
            ego_features = list(map(int, line.strip().split()))
        print(f"  Ego特征: 已读取")

    # 5. 读取特征名称
    feature_names = []
    if os.path.exists(featnames_file):
        with open(featnames_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    feature_names.append(' '.join(parts[1:]))
        print(f"  特征名称: {len(feature_names)} 个特征")

    return G, circles, node_features, ego_features, feature_names, ego_id


def create_complete_ego_network(G, circles, node_features, ego_features, ego_id, feature_names):
    """
    创建完整的ego-network可视化
    """
    # 创建图形布局
    plt.figure(figsize=(20, 6))

    # 子图1: 网络结构图
    plt.subplot(131)

    # 为ego节点创建特殊标记
    # 注意: 在原始数据中，ego节点不在边文件中，但我们需要在可视化中包含它
    pos = nx.spring_layout(G, k=1, iterations=50)

    # 绘制节点 - 根据圈子分配颜色
    node_colors = []
    node_sizes = []

    # 为每个圈子分配颜色
    circle_colors = {}
    color_map = plt.cm.Set3
    for i, circle_name in enumerate(circles.keys()):
        circle_colors[circle_name] = color_map(i / len(circles))

    # 处理节点颜色和大小
    all_nodes = list(G.nodes())
    for node in all_nodes:
        # 查找节点所属的圈子
        node_circles = [name for name, members in circles.items() if node in members]

        if node_circles:
            # 如果节点属于多个圈子，选择第一个
            node_colors.append(circle_colors[node_circles[0]])
            node_sizes.append(100)
        else:
            # 不属于任何圈子的节点
            node_colors.append('lightgray')
            node_sizes.append(60)

    # 绘制网络
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.8)

    # 可选: 添加节点标签（对于小网络）
    if len(all_nodes) <= 50:
        nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(f"Ego-Network {ego_id}\n(只包含好友间连接)")
    plt.axis('off')

    # 添加图例
    legend_elements = []
    for circle_name, color in circle_colors.items():
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=color, markersize=8,
                                          label=circle_name))
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor='lightgray', markersize=8,
                                      label='无圈子'))

    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))

    # 子图2: 圈子分布
    plt.subplot(132)
    if circles:
        circle_sizes = [len(members) for members in circles.values()]
        circle_names = list(circles.keys())

        colors = [circle_colors[name] for name in circle_names]
        bars = plt.bar(range(len(circle_names)), circle_sizes, color=colors, alpha=0.7)
        plt.xticks(range(len(circle_names)), circle_names, rotation=45, ha='right')
        plt.title("圈子大小分布")
        plt.ylabel("成员数量")

        # 添加数值标签
        for bar, size in zip(bars, circle_sizes):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     str(size), ha='center', va='bottom')
    else:
        plt.text(0.5, 0.5, "无圈子数据", ha='center', va='center', transform=plt.gca().transAxes)
        plt.title("圈子分布")

    # # 子图3: 网络统计信息
    # plt.subplot(133)
    # plt.axis('off')
    #
    # # 显示统计信息
    # stats_text = f"网络统计信息 - Ego {ego_id}\n\n"
    # stats_text += f"节点数量: {G.number_of_nodes()}\n"
    # stats_text += f"边数量: {G.number_of_edges()}\n"
    # stats_text += f"网络密度: {nx.density(G):.4f}\n"
    #
    # if G.number_of_nodes() > 0:
    #     stats_text += f"平均度: {2 * G.number_of_edges() / G.number_of_nodes():.2f}\n"
    #
    # if circles:
    #     stats_text += f"圈子数量: {len(circles)}\n"
    #     total_circled_nodes = sum(len(members) for members in circles.values())
    #     unique_circled_nodes = len(set().union(*circles.values()))
    #     stats_text += f"有圈子节点: {unique_circled_nodes}/{G.number_of_nodes()}\n"
    #     stats_text += f"圈子重叠度: {total_circled_nodes / unique_circled_nodes:.2f}\n"
    #
    # if node_features:
    #     stats_text += f"特征维度: {len(next(iter(node_features.values())))}\n"
    #
    # plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes,
    #          fontfamily='monospace', verticalalignment='top')

    plt.tight_layout()
    return plt.gcf()


def analyze_circle_overlap(circles):
    """
    分析圈子的重叠情况
    """
    if not circles:
        return {}

    overlap_info = {}
    circle_names = list(circles.keys())
    circle_sets = {name: set(members) for name, members in circles.items()}

    for i, name1 in enumerate(circle_names):
        overlap_info[name1] = {}
        set1 = circle_sets[name1]

        for j, name2 in enumerate(circle_names):
            if i != j:
                set2 = circle_sets[name2]
                overlap = len(set1 & set2)
                if overlap > 0:
                    overlap_ratio = overlap / len(set1)
                    overlap_info[name1][name2] = {
                        'overlap_count': overlap,
                        'overlap_ratio': overlap_ratio
                    }

    return overlap_info


def visualize_feature_space(node_features, circles, ego_id):
    """
    可视化特征空间（如果特征数据可用）
    """
    if not node_features or len(node_features) < 3:
        return None

    # 准备特征矩阵
    nodes = list(node_features.keys())
    feature_matrix = np.array([node_features[node] for node in nodes])

    # 降维
    if len(nodes) > 10:
        # 使用PCA预处理
        pca = PCA(n_components=min(50, feature_matrix.shape[1]))
        features_pca = pca.fit_transform(feature_matrix)

        # t-SNE降维
        perplexity = min(30, len(nodes) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        features_2d = tsne.fit_transform(features_pca)
    else:
        # 对于小网络，使用PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(feature_matrix)

    # 创建颜色映射
    circle_colors = {}
    color_map = plt.cm.Set3
    circle_names = list(circles.keys())
    for i, circle_name in enumerate(circle_names):
        circle_colors[circle_name] = color_map(i / len(circle_names))

    # 绘制特征空间
    plt.figure(figsize=(10, 8))

    # 为每个节点分配颜色
    node_colors = []
    for node in nodes:
        node_circles = [name for name, members in circles.items() if node in members]
        if node_circles:
            node_colors.append(circle_colors[node_circles[0]])
        else:
            node_colors.append('lightgray')

    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=node_colors, alpha=0.7, s=60)
    plt.title(f"Ego {ego_id} - 特征空间可视化")

    # 添加图例
    legend_elements = []
    for circle_name, color in circle_colors.items():
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=color, markersize=8,
                                          label=circle_name))
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor='lightgray', markersize=8,
                                      label='无圈子'))

    plt.legend(handles=legend_elements)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True, alpha=0.3)

    return plt.gcf()


def main():
    """
    主函数 - 处理多个ego网络
    """
    data_dir = "../data/Social"  # 数据文件所在目录
    ego_nodes = [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]  # 示例ego节点ID

    # 创建输出目录
    os.makedirs('./visualize', exist_ok=True)

    all_stats = []

    for ego_id in ego_nodes:
        try:
            # 读取数据
            G, circles, node_features, ego_features, feature_names, ego_id = read_facebook_data(data_dir, ego_id)

            if G.number_of_nodes() == 0:
                print(f"  Ego {ego_id}: 无数据，跳过")
                continue

            # 创建主可视化
            fig1 = create_complete_ego_network(G, circles, node_features, ego_features, ego_id, feature_names)
            plt.savefig(f'./visualize/ego_{ego_id}_network.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 创建特征空间可视化（如果特征数据足够）
            if len(node_features) >= 5:
                fig2 = visualize_feature_space(node_features, circles, ego_id)
                if fig2:
                    plt.savefig(f'./visualize/ego_{ego_id}_features.png', dpi=300, bbox_inches='tight')
                    plt.close()

            # 收集统计信息
            stats = {
                'ego_id': ego_id,
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'density': nx.density(G),
                'circles': len(circles),
                'features': len(node_features) if node_features else 0
            }
            all_stats.append(stats)

            # 分析圈子重叠
            overlap_info = analyze_circle_overlap(circles)
            if overlap_info:
                print(f"  圈子重叠分析:")
                for circle, overlaps in overlap_info.items():
                    if overlaps:
                        print(f"    {circle}: 与 {len(overlaps)} 个其他圈子重叠")

            print(f"  Ego {ego_id}: 处理完成\n")

        except Exception as e:
            print(f"  处理ego {ego_id}时出错: {e}")
            continue

    # 输出总体统计
    print("\n" + "=" * 60)
    print("总体统计信息")
    print("=" * 60)

    if all_stats:
        df_stats = pd.DataFrame(all_stats)
        print(df_stats.to_string(index=False))

        # 保存统计信息
        df_stats.to_csv('./visualize/ego_network_statistics.csv', index=False)
        print(f"\n统计信息已保存至: ./visualize/ego_network_statistics.csv")

    print(f"\n所有可视化结果已保存至 ./visualize")


if __name__ == "__main__":
    main()