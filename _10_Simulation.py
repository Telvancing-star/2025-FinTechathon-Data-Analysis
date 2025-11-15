import pickle, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
from utils.popularity_tr import Pop
from collections import defaultdict


class Diffusion:
    def __init__(self, target, target_score, effect_strength, alpha=0.3, beta=0.1, xi=0.8, threshold=0.6,
                 iter=10):
        self.target = target  # 产品
        self.target_score = target_score
        self.effect_strength = effect_strength
        self.iter = iter
        self.alpha, self.beta, self.xi, self.base_mu = alpha, beta, xi, 0.01
        self.threshold = threshold
        self.a = 0.5  # 投资金额计算参数：源节点投资权重
        self.b = 0.1  # 投资金额计算参数：邻居节点投资权重
        self.c = 0.4  # 投资金额计算参数：平均投资权重
        self.investment = {}

        with open('./data/Social/adj_neighbor.pkl', 'rb') as f:
            self.neighbor = pickle.load(f)

        with open('./data/Social/edge_probability_matrix.pkl', 'rb') as f:
            results = pickle.load(f)
        self.beta_hat = results['beta_hat']
        self.P = results['P_edge']

        # 存储每轮的可视化数据
        self.visualization_data = []

        # === 新增：统计分析数据结构 ===
        self.analysis_data = {
            'weighted_mu': [],
            'invest_prob': [],
            'baseline_tendency': [],
            'effect': [],
            'final_probability': [],
            'rounds': [],
            'investor_types': []  # 'new' 或 'existing'
        }

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

    def _record_analysis_data(self, weighted_mu, invest_prob, baseline_tendency, effect,
                              final_probability, current_round, investor_type):
        """记录分析数据"""
        self.analysis_data['weighted_mu'].append(weighted_mu)
        self.analysis_data['invest_prob'].append(invest_prob)
        self.analysis_data['baseline_tendency'].append(baseline_tendency)
        self.analysis_data['effect'].append(effect)
        self.analysis_data['final_probability'].append(final_probability)
        self.analysis_data['rounds'].append(current_round)
        self.analysis_data['investor_types'].append(investor_type)

    def _calculate_statistics(self, data_list):
        """计算均值和标准差"""
        if not data_list:
            return 0, 0
        data_array = np.array(data_list)
        return np.mean(data_array), np.std(data_array)

    def _analyze_variables(self):
        """分析所有变量的统计特性"""
        print("\n" + "=" * 60)
        print("变量统计分析报告")
        print("=" * 60)

        # 整体统计
        variables = ['weighted_mu', 'invest_prob', 'baseline_tendency', 'effect', 'final_probability']
        for var in variables:
            mean_val, std_val = self._calculate_statistics(self.analysis_data[var])
            print(
                f"{var:20} 均值: {mean_val:8.4f} 标准差: {std_val:8.4f} 变异系数: {std_val / max(mean_val, 0.001):8.4f}")

        # 按投资者类型分析
        print("\n--- 按投资者类型分析 ---")
        for investor_type in ['new', 'existing']:
            indices = [i for i, t in enumerate(self.analysis_data['investor_types']) if t == investor_type]
            if indices:
                print(f"\n{investor_type.upper()}投资者:")
                for var in variables:
                    values = [self.analysis_data[var][i] for i in indices]
                    mean_val, std_val = self._calculate_statistics(values)
                    print(f"  {var:18} 均值: {mean_val:6.4f} 标准差: {std_val:6.4f}")

        # 按轮次分析
        print("\n--- 按轮次趋势分析 ---")
        rounds = sorted(set(self.analysis_data['rounds']))
        if rounds:
            print("轮次趋势 (均值):")
            for var in ['final_probability', 'invest_prob']:
                round_means = []
                for r in rounds:
                    indices = [i for i, round_val in enumerate(self.analysis_data['rounds']) if round_val == r]
                    if indices:
                        values = [self.analysis_data[var][i] for i in indices]
                        round_means.append(np.mean(values))
                    else:
                        round_means.append(0)

                # 计算趋势
                if len(round_means) > 1:
                    trend = np.polyfit(range(len(round_means)), round_means, 1)[0]
                    trend_direction = "上升" if trend > 0.001 else "下降" if trend < -0.001 else "平稳"
                    print(f"  {var:18} 趋势: {trend:7.4f} ({trend_direction})")

    def _plot_variable_analysis(self, output_dir):
        """绘制变量分析图"""
        if not self.analysis_data['rounds']:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        variables = ['weighted_mu', 'invest_prob', 'baseline_tendency', 'effect', 'final_probability']
        titles = ['加权Mu', '传播概率', '基准倾向', '产品效应', '投资概率']

        for i, (var, title) in enumerate(zip(variables, titles)):
            if i < len(axes):
                # 直方图
                axes[i].hist(self.analysis_data[var], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].set_xlabel(title)
                axes[i].set_ylabel('频数')
                axes[i].set_title(f'{title}分布')
                axes[i].grid(True, alpha=0.3)

                # 添加统计信息
                mean_val, std_val = self._calculate_statistics(self.analysis_data[var])
                axes[i].axvline(mean_val, color='red', linestyle='--', label=f'均值: {mean_val:.3f}')
                axes[i].axvline(mean_val + std_val, color='orange', linestyle=':', label=f'±1标准差')
                axes[i].axvline(mean_val - std_val, color='orange', linestyle=':')
                axes[i].legend()

        # 第六个图：变量随时间变化
        if len(axes) > 5:
            rounds = self.analysis_data['rounds']
            for var, label in zip(['final_probability', 'invest_prob'], ['投资概率', '传播概率']):
                axes[5].scatter(rounds, self.analysis_data[var], alpha=0.6, label=label, s=20)
            axes[5].set_xlabel('轮次')
            axes[5].set_ylabel('概率值')
            axes[5].set_title('概率随时间变化')
            axes[5].legend()
            axes[5].grid(True, alpha=0.3)

        plt.tight_layout()
        analysis_plot_path = os.path.join(output_dir, 'variable_analysis.png')
        plt.savefig(analysis_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"变量分析图已保存: {analysis_plot_path}")

    def _calculate_weighted_mu(self, potential_investor, df, current_round):
        """
        使用对数变换处理极端值，更稳定的加权mu计算
        """
        investor_history = df[df['对应的local_node_id'] == potential_investor]
        invested_rounds_count = len(investor_history)

        # 传播频率（使用对数变换避免极端值）
        if current_round > 0 and invested_rounds_count > 0:
            raw_frequency = invested_rounds_count / (current_round + invested_rounds_count)
            # 对数变换：压缩极端值，保持中间区域的敏感性
            log_frequency = np.log1p(raw_frequency * 10) / np.log1p(10)  # 映射到0-1
        else:
            log_frequency = 0.0

        # 投资金额影响
        if len(investor_history) > 0:
            avg_investment = investor_history['Denominations'].mean()
            max_investment = df['Denominations'].max() if len(df) > 0 else 1.0
            investment_weight = avg_investment / max_investment
        else:
            investment_weight = 0.0

        # 新投资者基础倾向
        if invested_rounds_count == 0:
            base_mu = self.base_mu  # 固定的新投资者基础倾向
        else:
            base_mu = self.base_mu + self.alpha * log_frequency  # 0.1-0.4之间

        # 温和的金额权重
        weighted_mu = base_mu + self.beta * investment_weight

        return np.clip(weighted_mu, self.base_mu, 0.5)

    def _calculate_investment_decision(self, potential_investor, df, grouped, current_round):
        """
        考虑投资金额的个性化基准倾向，增加不投资行为传播逻辑
        """
        # 检查投资状态
        previous_round_records = df[df['Round'] == current_round - 1]
        previously_invested = potential_investor in previous_round_records['对应的local_node_id'].values

        weighted_mu = self._calculate_weighted_mu(potential_investor, df, current_round)
        # 计算sigma^2（与加权mu相关）
        sigma_squared = (weighted_mu ** 2) / 10.0 if weighted_mu > 0 else 0.01

        # 生成个性化基准投资倾向
        baseline_tendency = self.base_mu + np.random.normal(weighted_mu, np.sqrt(sigma_squared))
        baseline_tendency = np.clip(baseline_tendency, 0.01, 0.5)

        if previously_invested:
            product_effect, effect_strength = self.target_score[0], self.effect_strength[0]
            investor_type = 'existing'
        else:
            product_effect, effect_strength = self.target_score[1], self.effect_strength[1]
            investor_type = 'new'

        effect_value = product_effect * effect_strength

        # === 新增：不投资行为传播逻辑 ===
        investment_interruption_effect = 0.0
        if current_round >= 2:
            round_a_minus_2_records = df[df['Round'] == current_round - 2]
            round_a_minus_1_records = df[df['Round'] == current_round - 1]

            invested_at_a_minus_2 = potential_investor in round_a_minus_2_records['对应的local_node_id'].values
            invested_at_a_minus_1 = potential_investor in round_a_minus_1_records['对应的local_node_id'].values

            if invested_at_a_minus_2 and not invested_at_a_minus_1:
                interruption_noise = np.random.normal(weighted_mu / 2, np.sqrt(sigma_squared / 4))
                investment_interruption_effect = -abs(interruption_noise)

        # 原有的概率计算
        invest_prob = 0
        all_bernoulli_results = []
        for round_name, round_group in grouped:
            round_value = float(round_name) if isinstance(round_name, str) else round_name
            trail = current_round - round_value
            n_trials = round_group.shape[0]

            # 基础adjusted_prob计算
            adjusted_prob = self.P[potential_investor] * self.xi ** trail

            # === 应用不投资行为传播效应 ===
            adjusted_prob += investment_interruption_effect
            adjusted_prob = max(0, adjusted_prob)

            invest_prob += adjusted_prob * n_trials

            # 进行该轮次的伯努利试验（模拟投资决策）
            round_bernoulli_results = np.random.binomial(1, adjusted_prob, n_trials)
            all_bernoulli_results.extend(round_bernoulli_results)

        # 综合计算
        total_influence = (
                invest_prob +
                baseline_tendency +
                effect_value
        )

        final_probability = 1 / (1 + np.exp(-total_influence))

        # === 新增：记录分析数据 ===
        self._record_analysis_data(weighted_mu, invest_prob, baseline_tendency,
                                   effect_value, final_probability, current_round, investor_type)

        # 调试信息（抽样输出，避免过多日志）
        if np.random.random() < 0.05:  # 5%的概率输出调试信息
            print(f"轮次{current_round} - 节点{potential_investor}({investor_type}): "
                  f"投资概率={final_probability:.3f}, 传播概率={invest_prob:.3f}")

        return final_probability, all_bernoulli_results

    def main(self):
        data = pd.read_csv('./data/cluster_with_rounds.csv', encoding='gb18030')
        df = data[data['Name of the Political Party'] == self.target]
        self.oiter = max(self._get_round(df))

        # 准备动画
        fig, ax = plt.subplots(figsize=(12, 8))
        frames = []

        # 创建帧保存目录
        frame_dir = f'./output/frames_{self.target.replace(" ", "_")}_{self.xi}'
        os.makedirs(frame_dir, exist_ok=True)

        for current_round in range(self.oiter + 1, self.oiter + self.iter + 1):
            records = df[df['Round'] < current_round]
            self.investment = dict(zip(records['对应的local_node_id'], records['Denominations']))
            self.mean = records['Denominations'].mean()
            nodes = records['对应的local_node_id'].unique()

            if len(nodes) == 0:
                frames.append((df, current_round, []))
                continue
            else:
                all_neighbors = set()
                for node in nodes:
                    if node in self.neighbor:
                        all_neighbors.update(self.neighbor[node])

                new_investments = []

                for potential_investor in all_neighbors:
                    spread_nodes = [node for node in self.neighbor[potential_investor] if node in nodes]
                    if not spread_nodes:
                        continue
                    spread_record = records[records['对应的local_node_id'].isin(spread_nodes)]
                    grouped = spread_record.groupby('Round')
                    invest_prob, all_bernoulli_results = self._calculate_investment_decision(potential_investor, df,
                                                                                             grouped, current_round)

                    if np.any(all_bernoulli_results) and invest_prob >= self.threshold:
                        source_node = np.random.choice(spread_record['对应的local_node_id'])
                        new_investments.append({
                            'Name of the Political Party': self.target,
                            'Prefix': '/',
                            'Round': current_round,
                            'Denominations': self._investment(source_node),
                            '对应的local_node_id': potential_investor,
                            'Journal Date': '/'
                        })

                if new_investments:
                    new_rows = pd.DataFrame(new_investments)
                    df = pd.concat([df, new_rows], ignore_index=True)

            frames.append((df.copy(), current_round, new_investments))

            # 保存帧图像
            fig_frame, ax_frame = plt.subplots(figsize=(12, 8))
            G_frame, node_investments_frame, edge_data_frame = self._create_network_graph(df, current_round,
                                                                                          new_investments)
            self._plot_network(G_frame, node_investments_frame, edge_data_frame, current_round, new_investments,
                               fig_frame, ax_frame)
            frame_filename = f'round_{current_round}.png'
            fig_frame.savefig(os.path.join(frame_dir, frame_filename), dpi=150, bbox_inches='tight')
            plt.close(fig_frame)
            print(f"帧已保存: {frame_filename}")

        def update(frame_idx):
            df_frame, current_round, new_investments = frames[frame_idx]
            G, node_investments, edge_data = self._create_network_graph(df_frame, current_round, new_investments)
            self._plot_network(G, node_investments, edge_data, current_round, new_investments, fig, ax)
            return ax

        anim = FuncAnimation(fig, update, frames=len(frames), interval=1000, repeat=False)
        gif_filename = frame_dir + f'/diffusion_animation_{self.target.replace(" ", "_")}_{self.xi}_{current_round}.gif'
        anim.save(gif_filename, writer='pillow', fps=1)
        print(f"动图已保存到: {gif_filename}")

        plt.close()

        # 保存到新文件
        output_filename = frame_dir + f'/simulation_results_{self.target.replace(" ", "_")}_{current_round}.csv'
        df.to_csv(output_filename, index=False, encoding='gb18030')
        print(f"模拟结果已保存到: {output_filename}")

        # === 新增：执行变量分析 ===
        self._analyze_variables()
        self._plot_variable_analysis(frame_dir)

        print(f"所有帧图像已保存到: {frame_dir}")

        return df


if __name__ == '__main__':
    terget = 'BHARATIYA JANATA PARTY'
    target_score = [0.1, 0.2]
    effect_strength = [0.6, 0.4]

    for xi in [0.8, 0.81, 0.82]:
        run = Diffusion(terget, target_score=target_score, effect_strength=effect_strength, xi=xi, iter=15)
        run.main()