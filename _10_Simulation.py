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
        df = data[data['Name of the Political Party'] == self.target]  # 初始已投资人
        self.mean = df['Denominations'].mean()  # 此时只考虑一个产品
        # 疑问：什么叫把多个同一节点视为不同的？
        xi, rounds = self.xi, self._get_round(df)
        for round in range(26):
            nodes = df['对应的local_node_id'].unique()  # 疑问：什么叫把多个同一节点视为不同的？
            # 获取所有邻居并合并
            all_neighbors = list(chain.from_iterable(
                self.neighbor[node] for node in nodes if node in self.neighbor
            ))
            if round not in rounds:
                # plot 执行等待
                continue
            else:
                for i in all_neighbors:
                    if random.random() < self.P[i] and self.P[i] >= self.threshold:
                        # 更新df
                        new_row = pd.DataFrame({
                            'Name of the Political Party': self.target,
                            'Prefix': '/',
                            'Round': round + 1,
                            'Denominations': self._investment(np.random.choice(nodes)),
                            'Journal Date': '/'
                        })
                        # 使用concat合并
                        df = pd.concat([df, new_row], ignore_index=True)
                        rounds = self._get_round(df)  # 更新有投资者的轮次
            self.P *= xi  # 更新概率向量
        return


if __name__ == '__main__':
    terget = 'BHARATIYA JANATA PARTY'
    run = Diffusion(terget)
    run.main()
