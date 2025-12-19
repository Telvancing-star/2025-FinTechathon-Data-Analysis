import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import os

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

# --------------------------
# 参数设置（仅保留指定特征）
# --------------------------
INPUT_FILE = './data/Bonds/merged_three_keys.xlsx'  # 输入文件路径
SHEET_NAME = 'Sheet1'  # 工作表名称
TARGET_COL = 'Name of the Purchaser'  # 聚类目标：购买者
FEATURE_COLS = [  # 仅保留指定特征
    'Date of\nEncashment',
    'Name of the Political Party',
    'Prefix',
    'Denominations',
    'Journal Date',
    'Date of\nPurchase',
    'Date of Expiry'
]
INITIAL_K = 10  # 初始聚类数量
K_RANGE = range(10, 60)  # 轮廓系数评估的k值范围
OUTPUT_DIR = '/data/Aligned Paired Data'  # 结果保存目录

# --------------------------
# 数据读取与购买者特征聚合（仅处理指定特征）
# --------------------------
def load_and_aggregate_data(file_path, sheet_name):
    """读取数据并聚合每个购买者的行为特征（仅保留指定特征）"""
    excel_file = pd.ExcelFile(file_path)
    df = excel_file.parse(sheet_name)

    # 检查必要列是否存在
    required_cols = FEATURE_COLS + [TARGET_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据中缺少必要列：{missing_cols}")

    # 日期列转换为datetime
    date_cols = ['Date of\nEncashment', 'Date of\nPurchase', 'Journal Date', 'Date of Expiry']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # 按购买者聚合特征（仅处理指定特征）
    def aggregate_purchaser(group):
        # 数值特征：金额相关统计
        denom_stats = {
            '总投资金额': group['Denominations'].sum(),
            '平均每次投资金额': group['Denominations'].mean(),
            '最大投资金额': group['Denominations'].max(),
            '投资次数': group['Denominations'].count()
        }

        # 日期特征：时间范围统计
        purchase_dates = group['Date of\nPurchase'].dropna()
        encash_dates = group['Date of\nEncashment'].dropna()
        journal_dates = group['Journal Date'].dropna()
        expiry_dates = group['Date of Expiry'].dropna()
        date_stats = {
            '首次投资日期': purchase_dates.min(),
            '末次投资日期': purchase_dates.max(),
            '投资时间跨度(天)': (purchase_dates.max() - purchase_dates.min()).days if not purchase_dates.empty else 0,
            '平均投资间隔(天)': purchase_dates.diff().dt.days.mean() if len(purchase_dates) >= 2 else 0,
            '首次兑现日期': encash_dates.min(),
            '末次兑现日期': encash_dates.max(),
            '兑现时间跨度(天)': (encash_dates.max() - encash_dates.min()).days if not encash_dates.empty else 0,
            '平均兑现间隔(天)': encash_dates.diff().dt.days.mean() if len(encash_dates) >= 2 else 0,
            '首次日志日期': journal_dates.min(),
            '末次日志日期': journal_dates.max(),
            '日志时间跨度(天)': (journal_dates.max() - journal_dates.min()).days if not journal_dates.empty else 0,
            '平均日志间隔(天)': journal_dates.diff().dt.days.mean() if len(journal_dates) >= 2 else 0,
            '首次到期日期': expiry_dates.min(),
            '末次到期日期': expiry_dates.max(),
            '到期时间跨度(天)': (expiry_dates.max() - expiry_dates.min()).days if not expiry_dates.empty else 0,
            '平均到期间隔(天)': expiry_dates.diff().dt.days.mean() if len(expiry_dates) >= 2 else 0
        }

        # 分类特征：最常出现的值（仅保留政党和前缀）
        prefix_mode = group['Prefix'].mode()
        party_mode = group['Name of the Political Party'].mode()
        category_stats = {
            '主要前缀': prefix_mode[0] if not prefix_mode.empty else '无',
            '主要投资政党': party_mode[0] if not party_mode.empty else '无'
        }

        # 合并所有特征
        stats = {**denom_stats, **date_stats, **category_stats}
        return pd.Series(stats)

    # 执行聚合
    purchaser_features = df.groupby(TARGET_COL, group_keys=False).apply(aggregate_purchaser).reset_index()
    print(f"聚合后共得到 {len(purchaser_features)} 个独立购买者的特征数据（仅保留指定特征）")
    return purchaser_features

# --------------------------
# 特征预处理（仅处理指定特征的编码）
# --------------------------
def preprocess_features(purchaser_df):
    """预处理特征（仅处理指定特征）"""
    # 处理日期特征：转换为数值（距最早日期的天数）
    earliest_purchase = purchaser_df['首次投资日期'].min()
    purchaser_df['首次投资距最早天数'] = (purchaser_df['首次投资日期'] - earliest_purchase).dt.days
    purchaser_df['末次投资距最早天数'] = (purchaser_df['末次投资日期'] - earliest_purchase).dt.days
    earliest_encash = purchaser_df['首次兑现日期'].min()
    purchaser_df['首次兑现距最早天数'] = (purchaser_df['首次兑现日期'] - earliest_encash).dt.days
    purchaser_df['末次兑现距最早天数'] = (purchaser_df['末次兑现日期'] - earliest_encash).dt.days
    earliest_journal = purchaser_df['首次日志日期'].min()
    purchaser_df['首次日志距最早天数'] = (purchaser_df['首次日志日期'] - earliest_journal).dt.days
    purchaser_df['末次日志距最早天数'] = (purchaser_df['末次日志日期'] - earliest_journal).dt.days
    earliest_expiry = purchaser_df['首次到期日期'].min()
    purchaser_df['首次到期距最早天数'] = (purchaser_df['首次到期日期'] - earliest_expiry).dt.days
    purchaser_df['末次到期距最早天数'] = (purchaser_df['末次到期日期'] - earliest_expiry).dt.days

    # 分类特征编码（仅保留前缀和政党）
    df_encoded = pd.get_dummies(
        purchaser_df,
        columns=['主要前缀', '主要投资政党'],
        prefix=['前缀', '政党']
    )

    # 选择最终特征列（仅保留指定特征的衍生变量）
    feature_cols = [
        '总投资金额', '平均每次投资金额', '最大投资金额', '投资次数',
        '投资时间跨度(天)', '平均投资间隔(天)', '首次投资距最早天数', '末次投资距最早天数',
        '兑现时间跨度(天)', '平均兑现间隔(天)', '首次兑现距最早天数', '末次兑现距最早天数',
        '日志时间跨度(天)', '平均日志间隔(天)', '首次日志距最早天数', '末次日志距最早天数',
        '到期时间跨度(天)', '平均到期间隔(天)', '首次到期距最早天数', '末次到期距最早天数'
    ] + [col for col in df_encoded.columns if col.startswith('前缀_') or col.startswith('政党_')]

    X = df_encoded[feature_cols]

    # 处理缺失值
    X = X.fillna(X.median(numeric_only=True))

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df_encoded, X_scaled, feature_cols, scaler

# --------------------------
# 聚类分析与评估（逻辑不变）
# --------------------------
def perform_clustering(X_scaled, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=2024, n_init='auto')
    clusters = kmeans.fit_predict(X_scaled)
    return clusters, kmeans, X_scaled

def evaluate_clusters(X_scaled, k_range):
    silhouette_scores = []
    for k in k_range:
        _, kmeans, _ = perform_clustering(X_scaled, k)
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)
        print(f"k={k} 时的轮廓系数：{score:.4f}")
    return silhouette_scores

# --------------------------
# 计算簇中心节点（逻辑不变）
# --------------------------
def get_cluster_centroid_nodes(purchaser_df, X_scaled, clusters, kmeans, target_col):
    centroid_nodes = []
    for cluster_id in range(kmeans.n_clusters):
        cluster_mask = (clusters == cluster_id)
        cluster_X = X_scaled[cluster_mask]
        cluster_purchasers = purchaser_df.loc[cluster_mask, target_col].values
        centroid = kmeans.cluster_centers_[cluster_id:cluster_id + 1]
        distances = euclidean_distances(cluster_X, centroid).flatten()
        min_dist_idx = np.argmin(distances)
        centroid_node = cluster_purchasers[min_dist_idx]
        min_distance = distances[min_dist_idx]
        centroid_nodes.append({
            '簇ID': cluster_id,
            '中心节点（购买者）': centroid_node,
            '到质心的距离（标准化后）': round(min_distance, 4)
        })
    return pd.DataFrame(centroid_nodes)

# --------------------------
# 结果可视化与保存（逻辑不变）
# --------------------------
def visualize_results(k_range, silhouette_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, marker='o', color='b')
    plt.xlabel('聚类数量 (k)', fontsize=12)
    plt.ylabel('轮廓系数', fontsize=12)
    plt.title('不同k值的聚类效果评估（仅保留指定特征）', fontsize=14)
    plt.grid(alpha=0.3)
    plt.xticks(k_range)
    plt.tight_layout()
    return plt

def save_results(purchaser_df, centroid_nodes_df, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # 保存购买者-聚类标签对应表
    result_df = purchaser_df[[TARGET_COL, 'Cluster']]
    result_df.to_csv(f"{output_dir}/purchaser_cluster.csv", index=False)
    # 统计每个簇的购买者数量
    cluster_stats = result_df['Cluster'].value_counts().sort_index()
    cluster_stats.to_csv(f"{output_dir}/cluster_purchaser_count.csv", header=['购买者数量'])
    # 保存每个簇的特征均值
    cluster_features = purchaser_df.groupby('Cluster').mean(numeric_only=True)
    cluster_features.to_csv(f"{output_dir}/cluster_feature_mean.csv")
    # 保存簇中心节点
    centroid_nodes_df.to_csv(f"{output_dir}/cluster_centroid_nodes.csv", index=False)
    print(f"\n所有结果已保存至：{output_dir}")

# --------------------------
# 主函数（逻辑不变）
# --------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 读取数据并聚合特征
    print("正在读取数据并聚合购买者特征（仅保留指定特征）...")
    try:
        purchaser_df = load_and_aggregate_data(INPUT_FILE, SHEET_NAME)
    except Exception as e:
        print(f"数据聚合出错：{str(e)}")
        return
    
    # 2. 特征预处理
    print("\n正在预处理购买者特征...")
    try:
        purchaser_df, X_scaled, feature_cols, scaler = preprocess_features(purchaser_df)
        print(f"特征矩阵形状：{X_scaled.shape}，包含 {len(feature_cols)} 个特征（仅保留指定特征）")
    except Exception as e:
        print(f"特征预处理出错：{str(e)}")
        return
    
    # 3. 评估不同k值
    print("\n正在评估不同聚类数量的效果...")
    try:
        silhouette_scores = evaluate_clusters(X_scaled, K_RANGE)
    except Exception as e:
        print(f"聚类评估出错：{str(e)}")
        return
    
    # 4. 可视化评估结果
    print("\n生成轮廓系数可视化图...")
    try:
        plt = visualize_results(K_RANGE, silhouette_scores)
        plt.savefig(f"{OUTPUT_DIR}/silhouette_scores.png")
        plt.show()
    except Exception as e:
        print(f"可视化保存出错：{str(e)}")
        return
    
    # 5. 执行最终聚类
    print(f"\n使用k={INITIAL_K}执行最终聚类...")
    try:
        clusters, kmeans, X_scaled = perform_clustering(X_scaled, INITIAL_K)
        purchaser_df['Cluster'] = clusters
    except Exception as e:
        print(f"最终聚类出错：{str(e)}")
        return
    
    # 6. 计算中心节点
    print("\n正在计算每个簇的中心节点...")
    try:
        centroid_nodes_df = get_cluster_centroid_nodes(purchaser_df, X_scaled, clusters, kmeans, TARGET_COL)
    except Exception as e:
        print(f"中心节点计算出错：{str(e)}")
        return
    
    # 7. 保存结果
    try:
        save_results(purchaser_df, centroid_nodes_df, OUTPUT_DIR)
    except Exception as e:
        print(f"结果保存出错：{str(e)}")
        return
    
    # 8. 输出示例
    print("\n购买者聚类结果示例：")
    print(purchaser_df[[TARGET_COL, 'Cluster']].head(10))
    print("\n每个簇的中心节点：")
    print(centroid_nodes_df.to_string(index=False))

if __name__ == "__main__":
    main()