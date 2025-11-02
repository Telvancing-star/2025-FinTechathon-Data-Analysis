import pandas as pd
import numpy as np

# 读取两个CSV文件
df1 = pd.read_csv('../data/Social/compressed_features_expanded.csv')
df2 = pd.read_csv('../data/feature_matrix_with_headers.csv')

# 定义特征列
feature_cols = [f'feature_{i}' for i in range(1, 22)]

# 检查列名是否一致
print("文件1的列:", df1.columns.tolist())
print("文件2的列:", df2.columns.tolist())
print("特征列是否一致:", set(df1.columns) == set(df2.columns))

# 检查形状
print(f"文件1形状: {df1.shape}")
print(f"文件2形状: {df2.shape}")

# 检查特征数据是否完全相等
if df1[feature_cols].equals(df2[feature_cols]):
    print("✅ 特征数据完全一致")
else:
    print("❌ 特征数据不一致")

    # 找出差异
    diff_mask = ~np.isclose(df1[feature_cols].values, df2[feature_cols].values, equal_nan=True)
    diff_indices = np.where(diff_mask)

    if len(diff_indices[0]) > 0:
        print(f"发现 {len(diff_indices[0])} 处差异")
        # 显示前几个差异
        for i in range(min(5, len(diff_indices[0]))):
            row, col = diff_indices[0][i], diff_indices[1][i]
            print(f"行{row}, 列{feature_cols[col]}: {df1.iloc[row, col]} vs {df2.iloc[row, col]}")