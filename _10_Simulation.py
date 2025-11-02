import pickle, random
import pandas as pd
import numpy as np
from itertools import chain


class Diffusion:
    def __init__(self, target):
        self.xi = 0.5
        self.threshold = 0.6
        self.target = target  # 产品
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
        data = pd.read_csv('./data/cluster_with_rounds.csv')
        df = data[data['Name of the Political Party'] == self.target]  # 初始已投资记录, 此时只考虑一个产品
        # 疑问：什么叫把多个同一节点视为不同的？
        xi, rounds = self.xi, self._get_round(df)
        for round in range(26):
            records = df[df['Round'] == round]  # 获取本轮存在的投资记录
            self.mean = df[df['Round'] <= round]['Denominations'].mean()
            nodes = records['对应的local_node_id'].unique()  # 获取本轮存在的已有投资者
            if not nodes:  # 若本轮没有投资者
                # plot 执行等待
                continue
            else:
                # 获取所有邻居并合并, 这些是本轮的潜在投资对象
                all_neighbors = list(chain.from_iterable(
                    self.neighbor[node] for node in nodes if node in self.neighbor
                ))

                for neighbor in all_neighbors:  # 遍历潜在投资者
                    if neighbor not in nodes and self.P[neighbor] >= self.threshold:  # 不能是已投资者且 neighbor 具备投资倾向
                        spread_nodes = [node for node in self.neighbor[neighbor] if node in nodes]  # 找到潜在传播者
                        spread_record = records[records['对应的local_node_id'].isin(spread_nodes)]  # 找到潜在传播记录

                        # 批量生成 潜在传播的记录数 次伯努利试验结果
                        bernoulli_results = np.random.binomial(1, self.P[neighbor], spread_record.shape[0])

                        if np.any(bernoulli_results) and self.P[neighbor] >= self.threshold:
                            # 更新df
                            new_row = pd.DataFrame({
                                'Name of the Political Party': self.target,
                                'Prefix': '/',
                                'Round': round + 1,
                                'Denominations': self._investment(np.random.choice(spread_record['对应的local_node_id'])),
                                'Journal Date': '/'
                            })
                            # 使用concat合并
                            df = pd.concat([df, new_row], ignore_index=True)
            self.P *= xi  # 更新概率向量
        return


if __name__ == '__main__':
    terget = 'BHARATIYA JANATA PARTY'
    run = Diffusion(terget)
    run.main()
