import pickle
import pandas as pd
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
import os
from _5_data_reset import CompatibleData


class Diffusion:
    def __init__(self, target):
        self.xi = 0.5
        self.threshold = 0.6
        self.target = target  # 产品
        self.a = 0.6
        self.c = 0.4
        self.investment = {}

        # 用于存储每轮的投资传播数据
        self.visualization_data = []

        # 从pkl文件读取对象
        with open('./data/Social/compatible_data.pkl', 'rb') as f:  # 注意是'rb'二进制读取模式
            self.data = pickle.load(f)

        with open('./data/Social/adj_neighbor.pkl', 'rb') as f:  # 注意是'rb'二进制读取模式
            self.neighbor = pickle.load(f)

        with open('./data/Social/edge_probability_matrix.pkl', 'rb') as f:  # 注意是'rb'二进制读取模式
            results = pickle.load(f)
        self.beta_hat = results['beta_hat']
        self.P = results['P_edge']

    def _investment(self, source_node):
        return self.a * self.investment[source_node] + self.c * self.mean

    def _get_round(self, df):
        return df['Round'].unique()

    def _create_investment_graph(self, records, round_num, new_investments=None):
        """
        创建投资传播图
        """
        G = nx.DiGraph()

        # 添加节点
        current_investors = records['对应的local_node_id'].unique()
        investment_amounts = records.groupby('对应的local_node_id')['Denominations'].mean().to_dict()

        for node in current_investors:
            G.add_node(node, amount=investment_amounts.get(node, 0))

        # 添加边（传播关系）
        if new_investments:
            for new_investor, source_investor in new_investments:
                if source_investor in G.nodes and new_investor in G.nodes:
                    G.add_edge(source_investor, new_investor)

        return G

    def _plot_investment_round(self, G, round_num, ax):
        """
        绘制单轮投资图
        """
        ax.clear()

        if len(G.nodes) == 0:
            ax.text(0.5, 0.5, f'Round {round_num}\nNo Investors',
                    ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return

        # 获取节点投资金额
        node_amounts = [G.nodes[node].get('amount', 0) for node in G.nodes]

        if node_amounts:
            max_amount = max(node_amounts)
            min_amount = min(node_amounts)
        else:
            max_amount = 1
            min_amount = 0

        # 节点颜色基于投资金额
        if max_amount > min_amount:
            node_colors = [(G.nodes[node].get('amount', 0) - min_amount) / (max_amount - min_amount)
                           for node in G.nodes]
        else:
            node_colors = [0.5 for _ in G.nodes]

        # 布局
        pos = nx.spring_layout(G, k=1, iterations=50)

        # 绘制边
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.6, edge_color='gray',
                               arrows=True, arrowsize=20, arrowstyle='->')

        # 绘制节点
        nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                                       cmap='YlOrRd', node_size=300, alpha=0.8)

        # 添加节点标签（投资金额）
        node_labels = {node: f'{G.nodes[node].get("amount", 0):.0f}'
                       for node in G.nodes}
        nx.draw_networkx_labels(G, pos, ax=ax, labels=node_labels, font_size=8)

        # 设置标题和颜色条
        ax.set_title(f'Investment Diffusion - Round {round_num}\n'
                     f'Total Investors: {len(G.nodes)}', fontsize=12)

        # 添加颜色条
        if node_amounts and max_amount > min_amount:
            sm = plt.cm.ScalarMappable(cmap='YlOrRd',
                                       norm=plt.Normalize(vmin=min_amount, vmax=max_amount))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label='Investment Amount')

        ax.axis('off')

    def _create_animation(self):
        """
        创建投资传播动图
        """
        if not self.visualization_data:
            print("No visualization data available")
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        def update(frame):
            round_data = self.visualization_data[frame]
            G, round_num, new_investments = round_data
            self._plot_investment_round(G, round_num, ax)
            return []

        anim = FuncAnimation(fig, update, frames=len(self.visualization_data),
                             interval=1000, blit=False, repeat=True)

        # 保存动图
        os.makedirs('./output', exist_ok=True)
        anim.save('./output/investment_diffusion.gif', writer='pillow', fps=1, dpi=100)
        print("Animation saved as './output/investment_diffusion.gif'")

        plt.close()

    def main(self):
        data = pd.read_csv('./data/cluster_with_rounds.csv', encoding='gb18030')
        df = data[data['Name of the Political Party'] == self.target]  # 初始已投资记录, 此时只考虑一个产品

        # 存储初始投资金额
        initial_investments = df.groupby('对应的local_node_id')['Denominations'].first().to_dict()
        self.investment.update(initial_investments)

        # 存储每轮的新投资关系
        investment_edges = {}

        for round in range(26):
            records = df[df['Round'] <= round]  # 获取本轮存在的投资记录
            self.mean = records['Denominations'].mean()
            nodes = records['对应的local_node_id'].unique()  # 获取本轮存在的已有投资者

            # 存储本轮的新投资关系
            round_new_investments = []

            if len(nodes) == 0:  # 若本轮没有投资者
                # 记录空轮次
                G = self._create_investment_graph(records, round)
                self.visualization_data.append((G, round, []))
                continue
            else:
                # 获取所有邻居并合并, 这些是本轮的潜在投资对象
                all_neighbors = list(chain.from_iterable(
                    self.neighbor[node] for node in nodes if node in self.neighbor
                ))

                for potential_investor in all_neighbors:  # 重命名循环变量
                    if potential_investor not in nodes and self.P[potential_investor] >= self.threshold:
                        spread_nodes = [node for node in self.neighbor[potential_investor] if node in nodes]  # 找到潜在传播者
                        spread_record = records[records['对应的local_node_id'].isin(spread_nodes)]  # 找到潜在传播记录
                        grouped = spread_record.groupby('Round')  # 按已投资者的投资发生轮次分组

                        # 收集所有组的伯努利试验结果
                        all_bernoulli_results = []

                        for round_name, round_group in grouped:
                            # 获取该轮次的记录数
                            n_trials = round_group.shape[0]

                            # 计算该轮次的衰减后概率
                            adjusted_prob = self.P[potential_investor] * self.xi ** (26 - round_group['Round'])

                            # 进行该轮次的伯努利试验
                            round_bernoulli_results = np.random.binomial(1, adjusted_prob, n_trials)

                            # 添加到总结果中
                            all_bernoulli_results.extend(round_bernoulli_results)

                        if np.any(all_bernoulli_results) and self.P[potential_investor] >= self.threshold:
                            # 随机选择一个传播者作为投资来源
                            source_investor = np.random.choice(spread_record['对应的local_node_id'])
                            investment_amount = self._investment(source_investor)

                            # 更新投资记录
                            new_row = pd.DataFrame({
                                'Name of the Political Party': self.target,
                                'Prefix': '/',
                                'Round': round + 1,
                                'Denominations': investment_amount,
                                'Journal Date': '/'
                            })
                            df = pd.concat([df, new_row], ignore_index=True)

                            # 存储投资金额
                            self.investment[potential_investor] = investment_amount

                            # 记录投资关系
                            round_new_investments.append((potential_investor, source_investor))

                # 创建当前轮次的图
                current_records = df[df['Round'] <= round]
                G = self._create_investment_graph(current_records, round, round_new_investments)
                self.visualization_data.append((G, round, round_new_investments))

                print(f"Round {round}: {len(current_records)} total records, "
                      f"{len(round_new_investments)} new investments")

        # 创建最终轮次的图（第26轮）
        final_records = df[df['Round'] <= 26]
        G_final = self._create_investment_graph(final_records, 26, [])
        self.visualization_data.append((G_final, 26, []))

        # 创建动图
        self._create_animation()

        # 同时保存静态图片
        self._save_static_plots()

        return df

    def _save_static_plots(self):
        """
        保存每轮的静态图片
        """
        os.makedirs('./output/rounds', exist_ok=True)

        for i, (G, round_num, new_investments) in enumerate(self.visualization_data):
            fig, ax = plt.subplots(figsize=(10, 8))
            self._plot_investment_round(G, round_num, ax)
            plt.tight_layout()
            plt.savefig(f'./output/rounds/round_{round_num:02d}.png', dpi=150, bbox_inches='tight')
            plt.close()

        print(f"Static plots saved in './output/rounds/' directory")


if __name__ == '__main__':
    target = 'BHARATIYA JANATA PARTY'
    run = Diffusion(target)
    result_df = run.main()