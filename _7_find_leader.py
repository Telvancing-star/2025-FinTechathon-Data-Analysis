import pickle
import pandas as pd
import openpyxl

with open('./data/Social/edge_probability_matrix.pkl', 'rb') as f:  # 注意是'rb'二进制读取模式
    results = pickle.load(f)

gamma = results['gamma']
gamma_sorted = sorted([(float(key), float(val)) for key, val in enumerate(gamma)], key=lambda x: x[1], reverse=True)

print(gamma_sorted[:5])


class CausalEffectAnalysis:
    def __init__(self):
        # 读取数据
        self.investment_data = pd.read_csv('./data/cluster_with_rounds.csv', encoding='gb18030')
        with open('./data/Social/adj_neighbor.pkl', 'rb') as f:
            self.neighbor = pickle.load(f)

        self.eigen_data = pd.read_csv('./data/Social/compressed_features_expanded.csv', encoding='gb18030')

    def get_record(self, ego_node, save_to_csv=True, filename=None):
        neighbors = self.neighbor[ego_node]  # 列表
        df = self.investment_data.copy()
        record = df[(df['对应的local_node_id'].isin(neighbors)) | (df['对应的local_node_id'] == ego_node)]

        # 保存为CSV文件
        if save_to_csv:
            if filename is None:
                filename = f'./data/ego_network_records_node_{ego_node}.csv'
            record.to_csv(filename, index=False, encoding='gb18030')
            print(f"Ego网络记录已保存到: {filename}")

        return record


# 执行分析
if __name__ == '__main__':
    analyzer = CausalEffectAnalysis()

    data = pd.read_excel('./data/组合赋权法结果.xlsx')
    data_sorted = data.sort_values('综合得分_组合赋权', ascending=False)

    for index, row in data_sorted.head(5).iterrows():
        analyzer.get_record(row['local_node_id'], save_to_csv=True)
