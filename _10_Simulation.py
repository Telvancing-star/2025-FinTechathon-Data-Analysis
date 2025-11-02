import pickle, math
import pandas as pd
import numpy as np
from itertools import chain
from _5_data_reset import CompatibleData


class Diffusion:
    def __init__(self, target, num=3):
        self.target = target  # 产品
        self.num = num
        self.xi = 0.5
        self.threshold = 0.6
        self.a = 0.6
        self.c = 0.4
        self.investment = {}

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

    def main(self):
        data = pd.read_csv('./data/cluster_with_rounds.csv', encoding='gb18030')
        df = data[data['Name of the Political Party'] == self.target]  # 初始已投资记录, 此时只考虑一个产品

        for round in range(26):

            records = df[df['Round'] <= round]  # 获取本轮存在的投资记录
            self.investment = dict(zip(records['对应的local_node_id'], records['Denominations']))
            self.mean = records['Denominations'].mean()
            nodes = records['对应的local_node_id'].unique()  # 获取本轮存在的已有投资者

            if len(nodes) == 0:  # 若本轮没有投资者
                # plot 执行等待
                continue
            else:
                # 获取所有邻居并合并, 这些是本轮的潜在投资对象
                all_neighbors = set()
                for node in nodes:
                    if node in self.neighbor:
                        all_neighbors.update(self.neighbor[node])
                # 移除已经是投资者的人
                all_neighbors = all_neighbors - set(nodes)

                for potential_investor in all_neighbors:  # 重命名循环变量

                    spread_nodes = [node for node in self.neighbor[potential_investor] if node in nodes]  # 找到潜在传播者
                    spread_record = records[records['对应的local_node_id'].isin(spread_nodes)]  # 找到潜在传播记录
                    grouped = spread_record.groupby('Round')  # 按已投资者的投资发生轮次分组

                    # 收集所有组的伯努利试验结果
                    invest_prob = 0
                    all_bernoulli_results = []

                    for round_name, round_group in grouped:

                        round_value = float(round_name) if isinstance(round_name, str) else round_name
                        trail = round - round_value

                        # if trail < self.num:  # 只考虑最近 num 轮的影响

                        n_trials = round_group.shape[0]  # 获取该轮次的记录数, 也就是有

                        # 计算该轮次的衰减后概率, 更新投资概率
                        adjusted_prob = self.P[potential_investor] * self.xi ** trail
                        invest_prob += adjusted_prob * n_trials ** (1-self.data.delta)  # 同一人重复投资对潜在投资者的影响是衰减的

                        # 进行该轮次的伯努利试验
                        round_bernoulli_results = np.random.binomial(1, adjusted_prob, n_trials)

                        # 添加到总结果中
                        all_bernoulli_results.extend(round_bernoulli_results)

                    if invest_prob >= self.threshold:
                        print(round, potential_investor, invest_prob)
                    # if round > 1:
                    #     print(round, potential_investor, invest_prob)

                    if np.any(all_bernoulli_results) and invest_prob >= self.threshold:
                        # 更新df
                        new_row = pd.DataFrame([{
                            'Name of the Political Party': self.target,
                            'Prefix': '/',
                            'Round': round + 1,
                            'Denominations': self._investment(
                                np.random.choice(spread_record['对应的local_node_id'])),
                            'Journal Date': '/'
                        }])
                        # 使用concat合并
                        df = pd.concat([df, new_row], ignore_index=True)
        return df


if __name__ == '__main__':
    terget = 'BHARATIYA JANATA PARTY'
    run = Diffusion(terget)
    run.main()

'''
帮我添加一个可视化：对每一轮的存在投资行为（records）进行绘图，用节点表示投资者，用有向边表示上一轮投资的传播（所以最初轮的投资是没有边的独立集），用颜色深浅体现不同投资者投入资金的多少。
如果可以，将27张图（26轮，每次绘图要用到本轮投资者和预测结果，体现投资的传播）形成一张动图并保存
'''
