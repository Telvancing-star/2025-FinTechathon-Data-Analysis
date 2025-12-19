import pandas as pd
import random

# 设置随机种子，保证结果可复现
random.seed(42)

# --------------------------
# 1. 读取文件并处理编码问题
# --------------------------
# 读取sample_cluster_labels.csv（根据实际列名调整）
df_labels = pd.read_csv(
    './data/Aligned Paired Data/sample_cluster_labels.csv',
    encoding='gbk'  # 解决中文编码问题
)

# 读取purchaser_subcluster_mapping.csv
df_purchasers = pd.read_csv(
    './data/Aligned Paired Data/purchaser_subcluster_mapping.csv',
    encoding='gbk'  # 解决中文编码问题
)

# 打印列名，确认合并键（调试用）
print("sample_cluster_labels.csv 列名：", df_labels.columns.tolist())
print("purchaser_subcluster_mapping.csv 列名：", df_purchasers.columns.tolist())

# --------------------------
# 2. 匹配相同聚类标签的记录
# --------------------------
# 根据实际列名调整合并键（这里假设：
# sample_cluster_labels.csv 中有 'cluster_label' 列
# purchaser_subcluster_mapping.csv 中有 '子簇标签' 列（对应聚类标签）
merge_keys = {
    'left_on': 'cluster_label',    # 第一个文件的聚类标签列名
    'right_on': '子簇标签'         # 第二个文件的聚类标签列名（根据实际调整）
}

# 合并两个DataFrame，保留匹配的记录
merged = pd.merge(
    df_labels,
    df_purchasers,
    how='inner',  # 只保留两边都有的聚类标签
    **merge_keys
)

# 提取需要的列
matched_data = merged[['local_node_id', 'Name of the Purchaser']].drop_duplicates()
print(f"\n匹配到 {len(matched_data)} 条有效记录")

# --------------------------
# 3. 随机对应（非满射，以购买者数量为准）
# --------------------------
# 去重并获取列表
local_node_ids = matched_data['local_node_id'].unique().tolist()
purchasers = matched_data['Name of the Purchaser'].unique().tolist()

# 随机打乱
random.shuffle(local_node_ids)
random.shuffle(purchasers)

# 建立对应关系（以数量少的为准）
mapping = {}
for node_id, purchaser in zip(local_node_ids[:len(purchasers)], purchasers):
    mapping[purchaser] = node_id  # 购买者 -> local_node_id

print(f"建立了 {len(mapping)} 组对应关系")

# --------------------------
# 4. 更新merged_three_keys.xlsx
# --------------------------
# 读取Excel文件
df_merged = pd.read_excel('./data/Bonds/merged_three_keys.xlsx')

# 添加对应列
df_merged['对应的local_node_id'] = df_merged['Name of the Purchaser'].map(mapping)

# 保存结果
output_path = './data/Aligned Paired Data/merged_three_keys_with_node_id.xlsx'
df_merged.to_excel(output_path, index=False)
print(f"\n结果已保存至：{output_path}")