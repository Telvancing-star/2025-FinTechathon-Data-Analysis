
import pickle


with open('./data/Social/edge_probability_matrix.pkl', 'rb') as f:  # 注意是'rb'二进制读取模式
    results = pickle.load(f)


gamma = results['popularity_records']
gamma_sorted = sorted([(float(key), float(val)) for key,val in gamma.items()], key=lambda x:x[1], reverse=True)
print(gamma_sorted[:5])

gamma = results['gamma']
gamma_sorted = sorted([(float(key), float(val)) for key,val in enumerate(gamma)], key=lambda x:x[1], reverse=True)

print(gamma_sorted[:5])

import pandas as pd
import numpy as np
import scipy.stats as stats



class CausalEffectAnalysis:
    def __init__(self):
        # 读取数据
        self.investment_data = pd.read_csv('./data/cluster_with_rounds.csv', encoding='gb18030')
        with open('./data/Social/adj_neighbor.pkl', 'rb') as f:
            self.neighbor = pickle.load(f)

    def define_treatment_outcome(self, product_j, time_window=None):
        """
        定义处理变量和结果变量
        T_i,j: 意见领袖i是否投资产品j
        Y_i,j: 与i相连节点的投资情况
        """
        # 筛选特定产品的投资记录
        product_data = self.investment_data[
            self.investment_data['Name of the Political Party'] == product_j
            ].copy()

        if time_window:
            product_data = product_data[product_data['Round'] <= time_window]

        return product_data

    def identify_opinion_leaders(self, product_data, min_degree=5):
        """识别意见领袖（度数较高的投资者）"""
        leader_candidates = {}

        for node in product_data['对应的local_node_id'].unique():
            if node in self.neighbor:
                degree = len(self.neighbor[node])
                if degree >= min_degree:
                    leader_candidates[node] = degree

        # 按度数排序
        leaders = sorted(leader_candidates.items(), key=lambda x: x[1], reverse=True)
        return leaders[:50]  # 取前50个作为意见领袖

    def get_neighbor_investment_status(self, leader_node, product_data, product_j):
        """获取意见领袖邻居的投资状态"""
        if leader_node not in self.neighbor:
            return []

        neighbor_status = []
        for neighbor_node in self.neighbor[leader_node]:
            # 检查邻居是否投资了产品j
            has_invested = neighbor_node in product_data['对应的local_node_id'].values
            neighbor_status.append({
                'neighbor_node': neighbor_node,
                'Y_ij': 1 if has_invested else 0,
                'leader_invested': leader_node in product_data['对应的local_node_id'].values
            })

        return neighbor_status

    def calculate_causal_effect(self, product_j, time_window=None):
        """计算意见领袖对产品j的因果效应"""
        product_data = self.define_treatment_outcome(product_j, time_window)
        leaders = self.identify_opinion_leaders(product_data)

        causal_effects = []

        for leader_node, degree in leaders:
            neighbor_status = self.get_neighbor_investment_status(leader_node, product_data, product_j)

            if not neighbor_status:
                continue

            df_status = pd.DataFrame(neighbor_status)

            # 计算条件期望
            Y1_mean = df_status[df_status['leader_invested'] == True]['Y_ij'].mean()
            Y0_mean = df_status[df_status['leader_invested'] == False]['Y_ij'].mean()

            # 处理缺失值
            Y1_mean = 0 if pd.isna(Y1_mean) else Y1_mean
            Y0_mean = 0 if pd.isna(Y0_mean) else Y0_mean

            # 计算因果效应估计量
            tau_hat = Y1_mean - Y0_mean

            causal_effects.append({
                'leader_node': leader_node,
                'degree': degree,
                'tau_hat': tau_hat,
                'Y1_mean': Y1_mean,
                'Y0_mean': Y0_mean,
                'n_treated': len(df_status[df_status['leader_invested'] == True]),
                'n_control': len(df_status[df_status['leader_invested'] == False])
            })

        return pd.DataFrame(causal_effects)

    def statistical_test(self, causal_effects_df):
        """进行统计检验"""
        # 检验因果效应是否显著不为0
        significant_effects = causal_effects_df[
            (causal_effects_df['n_treated'] >= 10) &
            (causal_effects_df['n_control'] >= 10)
            ].copy()

        # 计算标准误（简化版本）
        def calculate_se(row):
            p1 = row['Y1_mean']
            p0 = row['Y0_mean']
            n1 = row['n_treated']
            n0 = row['n_control']

            var = (p1 * (1 - p1) / n1) + (p0 * (1 - p0) / n0)
            return np.sqrt(var) if var > 0 else 0

        significant_effects['se'] = significant_effects.apply(calculate_se, axis=1)
        significant_effects['z_score'] = significant_effects['tau_hat'] / significant_effects['se']
        significant_effects['p_value'] = 2 * (1 - stats.norm.cdf(np.abs(significant_effects['z_score'])))

        return significant_effects

    def visualize_results(self, causal_effects_df, product_j):
        """可视化因果效应结果"""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 因果效应分布
        ax1.hist(causal_effects_df['tau_hat'], bins=20, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Causal Effect (τ̂)')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Distribution of Causal Effects - {product_j}')
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)

        # 因果效应 vs 节点度数
        ax2.scatter(causal_effects_df['degree'], causal_effects_df['tau_hat'], alpha=0.6)
        ax2.set_xlabel('Node Degree')
        ax2.set_ylabel('Causal Effect (τ̂)')
        ax2.set_title('Causal Effect vs Node Degree')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f'./data/causal_effects_{product_j.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def main_analysis(self, target_party='BHARATIYA JANATA PARTY'):
        """主分析函数"""
        print(f"分析产品: {target_party}")

        # 计算因果效应
        causal_effects = self.calculate_causal_effect(target_party)

        print(f"找到 {len(causal_effects)} 个意见领袖")
        print(f"平均因果效应: {causal_effects['tau_hat'].mean():.4f}")
        print(f"因果效应标准差: {causal_effects['tau_hat'].std():.4f}")

        # 统计检验
        significant_effects = self.statistical_test(causal_effects)
        print(f"显著的意见领袖数量: {len(significant_effects)}")

        # 可视化
        self.visualize_results(causal_effects, target_party)

        # 保存结果
        causal_effects.to_csv(f'./data/causal_effects_{target_party.replace(" ", "_")}.csv', index=False)

        return causal_effects, significant_effects


# 执行分析
if __name__ == '__main__':
    analyzer = CausalEffectAnalysis()
    all_effects, sig_effects = analyzer.main_analysis()

    # 输出前10个最有影响力的意见领袖
    print("\nTop 10 最有影响力的意见领袖:")
    top_leaders = sig_effects.nlargest(10, 'tau_hat')[['leader_node', 'tau_hat', 'p_value', 'degree']]
    print(top_leaders.round(4))