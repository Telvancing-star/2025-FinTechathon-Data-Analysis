import os
from pathlib import Path


def _plot_network(self, G, node_investments, edge_data, current_round, new_investments, fig, ax, save_frames=False,
                  frame_dir=None):
    """绘制网络图 - 椭圆形内部均匀分布，并可选保存帧"""
    ax.clear()

    # 获取所有节点
    all_nodes = list(G.nodes())
    n_nodes = len(all_nodes)

    # 创建椭圆形内部的均匀分布
    if n_nodes > 0:
        pos = {}
        a, b = 1.0, 0.6  # 椭圆的长短轴

        # 在椭圆内生成随机均匀分布
        i = 0
        while i < n_nodes:
            # 在矩形区域内生成随机点
            x = np.random.uniform(-a, a)
            y = np.random.uniform(-b, b)

            # 检查点是否在椭圆内: (x/a)^2 + (y/b)^2 <= 1
            if (x / a) ** 2 + (y / b) ** 2 <= 1:
                pos[all_nodes[i]] = (x, y)
                i += 1
    else:
        pos = {}

    # 准备节点颜色（根据投资金额）
    node_colors = []
    for node in all_nodes:
        if node in node_investments:
            node_colors.append(node_investments[node])
        else:
            node_colors.append(0)

    # 归一化颜色值用于着色
    if node_colors and max(node_colors) > min(node_colors):
        normalized_colors = [(color - min(node_colors)) / (max(node_colors) - min(node_colors))
                             for color in node_colors]
        colors = [plt.cm.Blues(val) for val in normalized_colors]
    else:
        colors = ['lightblue'] * len(all_nodes)

    # 绘制边
    new_edges = [(u, v) for u, v, style in edge_data if style == 'new']
    existing_edges = [(u, v) for u, v, style in edge_data if style != 'new']

    # 绘制现有边（灰色）
    if existing_edges:
        nx.draw_networkx_edges(G, pos, edgelist=existing_edges,
                               edge_color='gray', width=1, alpha=0.5, arrows=True, ax=ax)

    # 绘制新传播边（红色高亮）
    if new_edges:
        nx.draw_networkx_edges(G, pos, edgelist=new_edges,
                               edge_color='red', width=2, alpha=0.8, arrows=True, ax=ax)

    # 分离现有节点和新节点
    new_nodes = [inv['对应的local_node_id'] for inv in new_investments]
    existing_nodes = [node for node in all_nodes if node not in new_nodes]

    # 为现有节点和新节点分别准备颜色
    existing_colors = [colors[all_nodes.index(node)] for node in existing_nodes]
    new_node_colors = ['red'] * len(new_nodes)

    # 绘制现有投资者节点
    if existing_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=existing_nodes,
                               node_color=existing_colors, node_size=100, alpha=0.8, ax=ax)

    # 高亮新投资者节点
    if new_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=new_nodes,
                               node_color=new_node_colors, node_size=150, alpha=0.9, ax=ax)

    # 设置坐标轴范围
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.8, 0.8)
    ax.set_aspect('equal')

    # 添加标题和信息
    ax.set_title(f'Investment Diffusion - Round {current_round}\n'
                 f'Total Investors: {len(G.nodes())}, New Investors: {len(new_investments)}',
                 fontsize=12)

    # 图例
    ax.text(0.02, 0.98, '● Existing Investor (Blue)\n● New Investor (Red)\n→ Propagation Path',
            transform=ax.transAxes, verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_axis_off()

    # 保存当前帧
    if save_frames and frame_dir:
        self._save_frame(fig, current_round, frame_dir)


def _save_frame(self, fig, current_round, frame_dir):
    """保存当前帧为图片"""
    # 确保目录存在
    os.makedirs(frame_dir, exist_ok=True)

    # 生成文件名
    filename = f"round_{current_round:03d}.png"
    filepath = os.path.join(frame_dir, filename)

    # 保存图片
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"帧已保存: {filepath}")


# 在主调用函数中添加保存功能
def run_visualization_pipeline(self, df, save_frames=True):
    """运行可视化pipeline，可选择保存所有帧"""

    # 创建帧保存目录
    if save_frames:
        frame_dir = "./animation_frames"
        print(f"帧将保存到: {frame_dir}")
    else:
        frame_dir = None

    # 原有的轮次循环
    rounds = sorted(df['Round'].unique())
    fig, ax = plt.subplots(figsize=(12, 8))

    for current_round in rounds:
        # 获取当前轮次的新投资
        current_investments = df[df['Round'] == current_round].to_dict('records')

        # 创建网络图
        G, node_investments, edge_data = self._create_network_graph(df, current_round, current_investments)

        # 绘制网络图（传入保存参数）
        self._plot_network(G, node_investments, edge_data, current_round,
                           current_investments, fig, ax,
                           save_frames=save_frames, frame_dir=frame_dir)

        # 显示当前帧（可选）
        plt.pause(0.5)  # 暂停0.5秒以便观察

    plt.close()

    if save_frames:
        print(f"所有帧已保存到: {frame_dir}")
        self._create_animation(frame_dir)  # 可选：创建动画


def _create_animation(self, frame_dir):
    """从帧创建动画（可选功能）"""
    try:
        from matplotlib.animation import PillowWriter

        # 获取所有帧文件
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])

        if not frame_files:
            print("未找到帧文件")
            return

        # 创建动画
        fig, ax = plt.subplots(figsize=(12, 8))
        writer = PillowWriter(fps=2)  # 2帧/秒

        animation_path = os.path.join(frame_dir, "investment_diffusion_animation.gif")

        with writer.saving(fig, animation_path, dpi=150):
            for frame_file in frame_files:
                frame_path = os.path.join(frame_dir, frame_file)
                img = plt.imread(frame_path)
                ax.imshow(img)
                ax.axis('off')
                writer.grab_frame()
                ax.clear()

        print(f"动画已创建: {animation_path}")

    except ImportError:
        print("Pillow 未安装，无法创建动画")
    except Exception as e:
        print(f"创建动画时出错: {e}")