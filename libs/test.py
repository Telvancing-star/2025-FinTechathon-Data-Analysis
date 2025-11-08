import json, pickle
import pandas as pd

# duplicates_file = "../data/Social/multiple_occurrences.json"
# # 读取重复节点信息
# with open(duplicates_file, 'r') as f:
#     duplicates = json.load(f)
#
# cnt = 0
# for key, value in duplicates.items():
#     cnt += value - 1
#
# print(cnt)

# # 读取CSV文件
# csv_file_path = "../data/Social/compressed_features.csv"
# df = pd.read_csv(csv_file_path)
# existing_ids = set(df["local_node_id"])
#
# missing_ids = []
# for i in range(4039):
#     if i not in existing_ids:
#         missing_ids.append(i)
#
# print(f"缺失的ID: {missing_ids}")

with open('../data/Social/edge_probability_matrix.pkl', 'rb') as f:  # 注意是'rb'二进制读取模式
    results = pickle.load(f)

print("流行度向量为:", results["gamma"])
print(results["gamma"].shape)
print("边预测概率矩阵为:", results["P_edge"])
print(results["P_edge"].shape)
print(type(results["P_edge"]))