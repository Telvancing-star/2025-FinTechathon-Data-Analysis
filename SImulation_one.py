import pickle, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
from utils.popularity_tr import Pop


class Diffusion:
    def __init__(self, target, delta=0.25, xi=0.93, threshold=0.6, num=3, iter=10):
        self.target = target  # 产品
        self.num = num
        self.iter = iter
        self.xi = xi
        self.threshold = threshold
        self.a = 0.6
        self.c = 0.4
        self.investment = {}
        self.delta = delta

        with open('./data/Social/adj_neighbor.pkl', 'rb') as f:  # 注意是'rb'二进制读取模式
            self.neighbor = pickle.load(f)

        with open('./data/Social/edge_probability_matrix.pkl', 'rb') as f:  # 注意是'rb'二进制读取模式
            results = pickle.load(f)
        self.beta_hat = results['beta_hat']
        self.P = results['P_edge']

        # 存储每轮的可视化数据
        self.visualization_data = []

    def _investment(self, source_node):
        return self.a * self.investment[source_node] + self.c * self.mean

    def _get_round(self, df):
        return df['Round'].unique()

    def _create_network_graph(self, df, current_round, new_investments):
        """创建当前轮次的网络图数据"""
        # 获取当前轮次及之前的所有投资记录
        records = df[df['Round'] <= current_round]

        # 创建有向图
        G = nx.DiGraph()

        # 添加节点和投资金额
        node_investments = {}
        for _, row in records.iterrows():
            node_id = row['对应的local_node_id']
            investment = row['Denominations']
            G.add_node(node_id)
            node_investments[node_id] = investment

        # 添加边（传播关系）
        edge_data = []

        # 对于当前轮次的新投资者，添加从传播者到他们的边
        for new_inv in new_investments:
            new_node = new_inv['对应的local_node_id']
            # 找到影响这个新投资者的传播者
            if new_node in self.neighbor:
                spread_nodes = [node for node in self.neighbor[new_node]
                                if node in G.nodes]
                if spread_nodes:
                    # 随机选择一个传播者
                    source_node = np.random.choice(spread_nodes)
                    G.add_edge(source_node, new_node)
                    edge_data.append((source_node, new_node, 'new'))

        return G, node_investments, edge_data

    def _plot_network(self, G, node_investments, edge_data, current_round, new_investments, fig, ax):
        """绘制网络图 - 椭圆形内部均匀分布"""
        ax.clear()

        # 获取所有节点
        all_nodes = list(G.nodes())
        n_nodes = len(all_nodes)

        # 创建椭圆形内部的均匀分布
        if n_nodes > 0:
            pos = {}
            a, b = 1.0, 0.6  # 椭圆的长短轴

            # 方法1: 在椭圆内生成随机均匀分布
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

    def main(self):
        data = pd.read_csv('./data/cluster_with_rounds.csv', encoding='gb18030')
        df = data[data['Name of the Political Party'] == self.target]
        self.oiter = max(self._get_round(df))

        # 准备动画
        fig, ax = plt.subplots(figsize=(12, 8))
        frames = []

        # 创建帧保存目录
        frame_dir = f'./data/frames_{self.target.replace(" ", "_")}_{self.xi}'
        os.makedirs(frame_dir, exist_ok=True)

        for round in range(self.oiter + 1, self.oiter + self.iter + 1):

            records = df[df['Round'] < round]  # 获取本轮存在的投资记录
            self.investment = dict(zip(records['对应的local_node_id'], records['Denominations']))
            self.mean = records['Denominations'].mean()
            nodes = records['对应的local_node_id'].unique()  # 获取本轮之前存在的已有投资者

            if len(nodes) == 0:  # 若本轮没有投资者
                # 记录空帧
                frames.append((df, round, []))
                continue
            else:
                # 获取所有邻居并合并, 这些是本轮的潜在投资对象
                all_neighbors = set()
                for node in nodes:
                    if node in self.neighbor:
                        all_neighbors.update(self.neighbor[node])

                # 在循环外部收集新投资者
                new_investments = []

                for potential_investor in all_neighbors:
                    spread_nodes = [node for node in self.neighbor[potential_investor] if node in nodes]  # 找到潜在传播者
                    if not spread_nodes:
                        continue
                    spread_record = records[records['对应的local_node_id'].isin(spread_nodes)]  # 找到潜在传播记录
                    grouped = spread_record.groupby('Round')  # 按已投资者的投资发生轮次分组

                    # 收集所有组的伯努利试验结果
                    invest_prob = 0
                    all_bernoulli_results = []

                    for round_name, round_group in grouped:
                        round_value = float(round_name) if isinstance(round_name, str) else round_name
                        trail = round - round_value

                        n_trials = round_group.shape[0]  # 获取该轮次的记录数, 也就是这一轮有多少影响到当前潜在投资者的已投资者

                        # 计算该轮次的衰减后概率, 更新投资概率
                        adjusted_prob = self.P[potential_investor] * self.xi ** trail
                        invest_prob += adjusted_prob * n_trials ** (1 - self.delta)

                        # 进行该轮次的伯努利试验
                        round_bernoulli_results = np.random.binomial(1, adjusted_prob, n_trials)

                        # 添加到总结果中
                        all_bernoulli_results.extend(round_bernoulli_results)

                    if np.any(all_bernoulli_results) and invest_prob >= self.threshold:
                        # 收集新投资信息，稍后统一添加到df
                        source_node = np.random.choice(spread_record['对应的local_node_id'])
                        new_investments.append({
                            'Name of the Political Party': self.target,
                            'Prefix': '/',
                            'Round': round,
                            'Denominations': self._investment(source_node),
                            '对应的local_node_id': potential_investor,
                            'Journal Date': '/'
                        })

                # 在循环外部统一更新df
                if new_investments:
                    new_rows = pd.DataFrame(new_investments)
                    df = pd.concat([df, new_rows], ignore_index=True)

            # 记录当前轮次的可视化数据
            frames.append((df.copy(), round, new_investments))

            fig_frame, ax_frame = plt.subplots(figsize=(12, 8))
            G_frame, node_investments_frame, edge_data_frame = self._create_network_graph(df, round, new_investments)
            self._plot_network(G_frame, node_investments_frame, edge_data_frame, round, new_investments, fig_frame,
                               ax_frame)

            frame_filename = f'round_{round}.png'
            fig_frame.savefig(os.path.join(frame_dir, frame_filename), dpi=150, bbox_inches='tight')
            plt.close(fig_frame)
            print(f"帧已保存: {frame_filename}")

        # [原有的动画代码保持不变...]
        def update(frame_idx):
            df_frame, round, new_investments = frames[frame_idx]
            G, node_investments, edge_data = self._create_network_graph(df_frame, round, new_investments)
            self._plot_network(G, node_investments, edge_data, round, new_investments, fig, ax)
            return ax

        anim = FuncAnimation(fig, update, frames=len(frames), interval=1000, repeat=False)
        gif_filename = frame_dir + f'/diffusion_animation_{self.target.replace(" ", "_")}_{self.xi}_{round}.gif'
        anim.save(gif_filename, writer='pillow', fps=1)
        print(f"动图已保存到: {gif_filename}")

        plt.close()

        # 保存到新文件
        output_filename = frame_dir + f'/simulation_results_{self.target.replace(" ", "_")}_{round}.csv'
        df.to_csv(output_filename, index=False, encoding='gb18030')
        print(f"模拟结果已保存到: {output_filename}")

        print(f"所有帧图像已保存到: {frame_dir}")

        return df


if __name__ == '__main__':
    terget = 'ALL INDIA TRINAMOOL CONGRESS'
    # for xi in [0.93, 0.935, 0.94, 0.945]:
    for xi in [0.95, 0.955]:
        run = Diffusion(terget, xi=xi, iter=15)
        run.main()
