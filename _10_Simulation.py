import pickle, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
from utils.popularity_tr import Pop


class Diffusion:
    def __init__(self, target, target_score, effect_strength, delta=0.25, xi=0.75, threshold=0.6, iter=10):
        """
        初始化扩散模型

        参数:
        target: 目标产品/政党名称
        delta: 网络稀疏参数，控制网络密度
        xi: 衰减系数，控制信息传播的衰减速度
        threshold: 投资概率阈值，只有超过该阈值才会投资
        iter: 仿真迭代轮次
        """
        self.target = target  # 产品/政党名称
        self.target_score = target_score
        self.effect_strength = effect_strength
        self.iter = iter
        self.xi = xi  # 衰减系数：控制历史影响的衰减速度
        self.threshold = threshold  # 投资决策阈值
        self.a = 0.5  # 投资金额计算参数：源节点投资权重
        self.b = 0.1  # 投资金额计算参数：邻居节点投资权重
        self.c = 0.4  # 投资金额计算参数：平均投资权重
        self.investment = {}  # 存储节点投资金额的字典
        self.delta = delta  # 网络稀疏参数

        # 加载邻居关系数据（社交网络结构）
        with open('./data/Social/adj_neighbor.pkl', 'rb') as f:
            self.neighbor = pickle.load(f)

        # 加载预计算的边概率矩阵（基于流行度模型）
        with open('./data/Social/edge_probability_matrix.pkl', 'rb') as f:
            results = pickle.load(f)
        self.beta_hat = results['beta_hat']  # 估计的特征参数
        self.P = results['P_edge']  # 边存在概率向量

        # 存储每轮的可视化数据
        self.visualization_data = []

    def _investment(self, source_node):
        """
        计算新投资者的投资金额

        基于源节点投资金额和平均投资金额的加权组合
        公式: a * source_investment + b * neighbor_investment + c * mean_investment

        参数:
        source_node: 传播源节点

        返回:
        新投资者的投资金额
        """
        neighbor_investment, cnt = 0, 0

        for neighbor in self.neighbor[source_node]:
            cnt += 1
            if neighbor in self.investment:
                neighbor_investment += self.investment[neighbor]
        neighbor_investment /= cnt if cnt > 0 else 1  # 防止除零

        return self.a * self.investment[source_node] + self.b * neighbor_investment + self.c * self.mean

    def _get_round(self, df):
        """
        获取数据集中所有轮次的唯一值

        参数:
        df: 包含投资记录的数据框

        返回:
        轮次数组
        """
        return df['Round'].unique()

    def _create_network_graph(self, df, current_round, new_investments):
        """
        创建当前轮次的网络图数据

        参数:
        df: 完整投资记录数据框
        current_round: 当前轮次
        new_investments: 本轮新增的投资记录

        返回:
        G: 网络图对象
        node_investments: 节点投资金额字典
        edge_data: 边数据列表
        """
        # 获取当前轮次及之前的所有投资记录
        records = df[df['Round'] <= current_round]

        # 创建有向图（表示投资传播方向）
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
            # 找到影响这个新投资者的传播者（邻居中已投资的节点）
            if new_node in self.neighbor:
                spread_nodes = [node for node in self.neighbor[new_node]
                                if node in G.nodes]
                if spread_nodes:
                    # 随机选择一个传播者作为影响源
                    source_node = np.random.choice(spread_nodes)
                    G.add_edge(source_node, new_node)
                    edge_data.append((source_node, new_node, 'new'))

        return G, node_investments, edge_data

    def _plot_network(self, G, node_investments, edge_data, current_round, new_investments, fig, ax):
        """
        绘制网络图 - 椭圆形内部均匀分布

        可视化当前轮次的投资传播网络状态
        """
        ax.clear()

        # 获取所有节点
        all_nodes = list(G.nodes())
        n_nodes = len(all_nodes)

        # 创建椭圆形内部的均匀分布（美观的节点布局）
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

        # 归一化颜色值用于着色（蓝色渐变表示投资金额大小）
        if node_colors and max(node_colors) > min(node_colors):
            normalized_colors = [(color - min(node_colors)) / (max(node_colors) - min(node_colors))
                                 for color in node_colors]
            colors = [plt.cm.Blues(val) for val in normalized_colors]
        else:
            colors = ['lightblue'] * len(all_nodes)

        # 绘制边
        new_edges = [(u, v) for u, v, style in edge_data if style == 'new']  # 新增传播边
        existing_edges = [(u, v) for u, v, style in edge_data if style != 'new']  # 已有边

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
        new_node_colors = ['red'] * len(new_nodes)  # 新节点用红色高亮

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

        # 图例说明
        ax.text(0.02, 0.98, '● Existing Investor (Blue)\n● New Investor (Red)\n→ Propagation Path',
                transform=ax.transAxes, verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_axis_off()

    def _calculate_investment_decision(self, potential_investor, df, grouped, current_round):
        """
        考虑投资金额的个性化基准倾向
        """
        # 原有的概率计算
        invest_prob = 0
        all_bernoulli_results = []
        for round_name, round_group in grouped:
            round_value = float(round_name) if isinstance(round_name, str) else round_name
            trail = current_round - round_value
            n_trials = round_group.shape[0]
            adjusted_prob = self.P[potential_investor] * self.xi ** trail
            invest_prob += adjusted_prob * n_trials ** (1 - self.delta)

            # 进行该轮次的伯努利试验（模拟投资决策）
            round_bernoulli_results = np.random.binomial(1, adjusted_prob, n_trials)
            all_bernoulli_results.extend(round_bernoulli_results)

        # 检查投资状态
        previous_round_records = df[df['Round'] == current_round - 1]
        previously_invested = potential_investor in previous_round_records['对应的local_node_id'].values

        # 计算个性化的基准投资倾向（考虑投资金额）
        investor_history = df[df['对应的local_node_id'] == potential_investor]
        invested_rounds_count = len(investor_history)

        # 计算加权mu：考虑投资金额的轮次比例
        if current_round > 0 and len(investor_history) > 0:
            # 计算平均投资金额（归一化）
            avg_investment = investor_history['Denominations'].mean()
            max_investment = df['Denominations'].max() if len(df) > 0 else 1.0
            investment_weight = avg_investment / max_investment

            # 基础mu + 投资金额权重
            base_mu = invested_rounds_count / (current_round + invested_rounds_count)
            weighted_mu = base_mu * (1 + 0.5 * investment_weight)  # 投资金额大的用户倾向更高
        else:
            weighted_mu = 0.0

        # 计算sigma^2（与加权mu相关）
        sigma_squared = (weighted_mu ** 2) / 10.0 if weighted_mu > 0 else 0.01

        # 生成个性化基准倾向
        baseline_tendency = 0.1 + np.random.normal(weighted_mu, np.sqrt(sigma_squared))
        baseline_tendency = np.clip(baseline_tendency, 0.01, 0.5)

        if previously_invested:
            product_effect, effect_strength = self.target_score[0], self.effect_strength[0]
        else:
            product_effect, effect_strength = self.target_score[1], self.effect_strength[1]

        # 综合计算
        total_influence = (
                invest_prob +
                baseline_tendency +
                product_effect * effect_strength
        )

        final_probability = 1 / (1 + np.exp(-total_influence))

        # print(f"节点 {potential_investor}: {investor_type}, "
        #       f"历史{invested_rounds_count}轮, 加权mu={weighted_mu:.3f}, "
        #       f"基准倾向: {baseline_tendency:.3f}, 最终概率: {final_probability:.4f}")

        return final_probability, all_bernoulli_results

    def main(self):
        """
        主仿真函数

        执行多轮投资扩散仿真，包括：
        1. 数据加载和初始化
        2. 多轮投资决策仿真
        3. 网络动态可视化
        4. 结果保存
        """
        # 加载基础数据
        data = pd.read_csv('./data/cluster_with_rounds.csv', encoding='gb18030')
        df = data[data['Name of the Political Party'] == self.target]  # 筛选目标产品的初始投资记录
        self.oiter = max(self._get_round(df))  # 获取初始最大轮次

        # 准备动画
        fig, ax = plt.subplots(figsize=(12, 8))
        frames = []  # 存储每轮的可视化数据

        # 主仿真循环：从当前轮次开始进行多轮扩散
        for current_round in range(self.oiter + 1, self.oiter + self.iter + 1):
            # 获取本轮之前的所有投资记录
            records = df[df['Round'] < current_round]
            self.investment = dict(zip(records['对应的local_node_id'], records['Denominations']))
            self.mean = records['Denominations'].mean()  # 计算平均投资金额
            nodes = records['对应的local_node_id'].unique()  # 获取本轮之前存在的已有投资者

            if len(nodes) == 0:  # 若本轮没有投资者
                # 记录空帧
                frames.append((df, current_round, []))
                continue
            else:
                # 获取所有邻居并合并，这些是本轮的潜在投资对象
                all_neighbors = set()
                for node in nodes:
                    if node in self.neighbor:
                        all_neighbors.update(self.neighbor[node])

                # 在循环外部收集新投资者
                new_investments = []

                # 对每个潜在投资者进行投资决策
                for potential_investor in all_neighbors:
                    # 找到潜在传播者（邻居中已投资的节点）
                    spread_nodes = [node for node in self.neighbor[potential_investor] if node in nodes]
                    if not spread_nodes:
                        continue

                    # 找到潜在传播记录
                    spread_record = records[records['对应的local_node_id'].isin(spread_nodes)]
                    grouped = spread_record.groupby('Round')  # 按已投资者的投资发生轮次分组
                    invest_prob, all_bernoulli_results = self._calculate_investment_decision(potential_investor, df,
                                                                                             grouped, current_round)

                    # 投资决策条件：
                    # 1. 至少一次伯努利试验成功
                    # 2. 累计投资概率超过阈值
                    if np.any(all_bernoulli_results) and invest_prob >= self.threshold:
                        # 随机选择一个传播者作为影响源
                        source_node = np.random.choice(spread_record['对应的local_node_id'])
                        # 收集新投资信息
                        new_investments.append({
                            'Name of the Political Party': self.target,
                            'Prefix': '/',
                            'Round': current_round,
                            'Denominations': self._investment(source_node),  # 计算投资金额
                            '对应的local_node_id': potential_investor,
                            'Journal Date': '/'
                        })

                # 在循环外部统一更新df
                if new_investments:
                    new_rows = pd.DataFrame(new_investments)
                    df = pd.concat([df, new_rows], ignore_index=True)

                # 记录当前轮次的可视化数据
                frames.append((df.copy(), current_round, new_investments))

        # 创建动画函数
        def update(frame_idx):
            """动画更新函数"""
            df_frame, current_round, new_investments = frames[frame_idx]
            G, node_investments, edge_data = self._create_network_graph(df_frame, current_round, new_investments)
            self._plot_network(G, node_investments, edge_data, current_round, new_investments, fig, ax)
            return ax

        # 生成动画
        anim = FuncAnimation(fig, update, frames=len(frames), interval=1000, repeat=False)

        # 保存动图
        gif_filename = f'./data/gif_1/diffusion_animation_{self.target.replace(" ", "_")}_{self.xi}_{current_round}.gif'
        anim.save(gif_filename, writer='pillow', fps=1)
        print(f"动图已保存到: {gif_filename}")

        plt.close()

        # 保存仿真结果到CSV文件
        output_filename = f'./data/gif_1/simulation_results_{self.target.replace(" ", "_")}_{current_round}.csv'
        df.to_csv(output_filename, index=False, encoding='gb18030')
        print(f"模拟结果已保存到: {output_filename}")

        return df


if __name__ == '__main__':
    """
    对不同的衰减系数xi进行参数敏感性分析
    """
    terget = 'BHARATIYA JANATA PARTY'
    target_score = [0.1, 0.2]  # 产品: [使已投资的人仍然想投资, 使未投资的人想投资] 的得分(-1~1)
    effect_strength = [0.6, 0.4]  # 产品得分的影响系数

    # 参数扫描：测试不同的衰减系数
    # for xi in [0.93, 0.935, 0.94, 0.945]:
    for xi in range(943, 950):
        run = Diffusion(terget, target_score=target_score, effect_strength=effect_strength, xi=xi, iter=15)
        run.main()