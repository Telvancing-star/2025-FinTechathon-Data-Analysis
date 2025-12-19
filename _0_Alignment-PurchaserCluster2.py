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
# 关键参数设置
# --------------------------
# 读取簇ID与k值对应关系文件
CLUSTER_K_FILE = './data/Aligned Paired Data/cluster_centroid_nodes+与ego节点对应.xlsx'
# 读取待聚类数据文件
INPUT_DATA_FILE = './data/Bonds/merged_three_keys.xlsx'
SHEET_NAME = 'Sheet1'
# 聚类目标列
TARGET_COL = 'Name of the Purchaser'
# 待使用的特征列
FEATURE_COLS = [
    'Date of\nEncashment',
    'Name of the Political Party',
    'Prefix',
    'Denominations',
    'Journal Date',
    'Date of\nPurchase',
    'Date of Expiry'
]
# 结果保存目录
OUTPUT_DIR = 'data/Aligned Paired Data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------
# 步骤1：读取簇ID对应的k值
# --------------------------
def load_cluster_k_mapping(file_path):
    """读取簇ID与k值的对应关系"""
    df = pd.read_excel(file_path)
    # 检查必要列是否存在
    required_cols = ['簇ID', 'k']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"簇k值文件缺少必要列：{missing_cols}")
    # 构建簇ID到k值的映射字典
    cluster_k_mapping = dict(zip(df['簇ID'].astype(int), df['k'].astype(int)))
    print(f"成功读取簇ID与k值映射，共{len(cluster_k_mapping)}个簇")
    print("簇ID->k值映射：", cluster_k_mapping)
    return cluster_k_mapping

# --------------------------
# 步骤2：读取并预处理原始数据
# --------------------------
def load_and_preprocess_raw_data(file_path, sheet_name):
    """读取原始数据并进行基础预处理"""
    # 读取数据
    excel_file = pd.ExcelFile(file_path)
    df = excel_file.parse(sheet_name)
    
    # 检查必要列是否存在
    required_cols = FEATURE_COLS + [TARGET_COL, 'Cluster']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"原始数据缺少必要列：{missing_cols}")
    
    # 日期列转换为datetime格式
    date_cols = ['Date of\nEncashment', 'Date of\nPurchase', 'Journal Date', 'Date of Expiry']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # 处理缺失值（数值型填充中位数，日期型填充众数）
    for col in FEATURE_COLS:
        if col in date_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
        elif col == 'Denominations':
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna('未知', inplace=True)
    
    print(f"原始数据读取完成，共{len(df)}条记录，{df['Cluster'].nunique()}个簇")
    return df

# --------------------------
# 步骤3：按购买者聚合特征
# --------------------------
def aggregate_purchaser_features(cluster_df):
    """对单个簇内的数据按购买者聚合特征"""
    def aggregate_func(group):
        # 数值特征（金额相关）
        denom_stats = {
            '总投资金额': group['Denominations'].sum(),
            '平均投资金额': group['Denominations'].mean(),
            '投资次数': group['Denominations'].count()
        }
        
        # 日期特征（时间跨度）
        purchase_dates = group['Date of\nPurchase'].dropna()
        encash_dates = group['Date of\nEncashment'].dropna()
        journal_dates = group['Journal Date'].dropna()
        expiry_dates = group['Date of Expiry'].dropna()
        
        date_stats = {
            '投资时间跨度(天)': (purchase_dates.max() - purchase_dates.min()).days if not purchase_dates.empty else 0,
            '兑现时间跨度(天)': (encash_dates.max() - encash_dates.min()).days if not encash_dates.empty else 0,
            '日志时间跨度(天)': (journal_dates.max() - journal_dates.min()).days if not journal_dates.empty else 0,
            '到期时间跨度(天)': (expiry_dates.max() - expiry_dates.min()).days if not expiry_dates.empty else 0
        }
        
        # 分类特征（取出现频率最高的值）
        prefix_mode = group['Prefix'].mode()
        party_mode = group['Name of the Political Party'].mode()
        category_stats = {
            '主要前缀': prefix_mode[0] if not prefix_mode.empty else '未知',
            '主要投资政党': party_mode[0] if not party_mode.empty else '未知'
        }
        
        # 合并所有特征
        return pd.Series({**denom_stats, **date_stats, **category_stats})
    
    # 按购买者分组聚合
    purchaser_features = cluster_df.groupby(TARGET_COL, group_keys=False).apply(aggregate_func).reset_index()
    return purchaser_features

# --------------------------
# 步骤4：特征编码与标准化
# --------------------------
def preprocess_features_for_clustering(purchaser_df):
    """对聚合后的特征进行编码和标准化"""
    # 分类特征独热编码
    df_encoded = pd.get_dummies(
        purchaser_df,
        columns=['主要前缀', '主要投资政党'],
        prefix=['前缀', '政党'],
        dummy_na=False
    )
    
    # 选择用于聚类的数值特征列
    feature_cols = [
        '总投资金额', '平均投资金额', '投资次数',
        '投资时间跨度(天)', '兑现时间跨度(天)', '日志时间跨度(天)', '到期时间跨度(天)'
    ] + [col for col in df_encoded.columns if col.startswith('前缀_') or col.startswith('政党_')]
    
    # 提取特征矩阵并处理缺失值
    X = df_encoded[feature_cols].fillna(0)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return df_encoded, X_scaled, feature_cols, scaler

# --------------------------
# 步骤5：对单个簇执行聚类
# --------------------------
def cluster_single_group(cluster_id, cluster_data, k_value):
    """对单个簇执行K-Means聚类"""
    print(f"\n=== 开始处理簇ID：{cluster_id}，聚类k值：{k_value} ===")
    
    # 按购买者聚合特征
    purchaser_features = aggregate_purchaser_features(cluster_data)
    if len(purchaser_features) < k_value:
        print(f"警告：簇{cluster_id}仅包含{len(purchaser_features)}个购买者，小于k值{k_value}，调整k为{len(purchaser_features)}")
        k_value = len(purchaser_features) if len(purchaser_features) > 1 else 1
    
    if len(purchaser_features) <= 1:
        print(f"簇{cluster_id}购买者数量过少，无需聚类")
        purchaser_features['子簇标签'] = 0
        return purchaser_features, None, None
    
    # 特征预处理
    df_encoded, X_scaled, feature_cols, scaler = preprocess_features_for_clustering(purchaser_features)
    
    # 执行K-Means聚类
    kmeans = KMeans(n_clusters=k_value, random_state=2024, n_init='auto')
    cluster_labels = kmeans.fit_predict(X_scaled)
    df_encoded['子簇标签'] = cluster_labels
    
    # 计算轮廓系数
    silhouette_score_val = silhouette_score(X_scaled, cluster_labels) if k_value > 1 else np.nan
    print(f"簇{cluster_id}聚类完成，轮廓系数：{silhouette_score_val:.4f}")
    
    # 找到每个子簇的中心节点（距离中心最近的购买者）
    centroid_nodes = []
    for sub_cluster_id in range(k_value):
        sub_cluster_mask = (cluster_labels == sub_cluster_id)
        sub_cluster_X = X_scaled[sub_cluster_mask]
        if len(sub_cluster_X) == 0:
            continue
        centroid = kmeans.cluster_centers_[sub_cluster_id:sub_cluster_id+1]
        distances = euclidean_distances(sub_cluster_X, centroid).flatten()
        min_dist_idx = np.argmin(distances)
        centroid_purchaser = purchaser_features.iloc[np.where(sub_cluster_mask)[0][min_dist_idx]][TARGET_COL]
        centroid_nodes.append({
            '簇ID': cluster_id,
            '子簇ID': sub_cluster_id,
            '中心购买者': centroid_purchaser,
            '到质心距离(标准化)': round(distances[min_dist_idx], 4)
        })
    
    return df_encoded[[TARGET_COL, '子簇标签']], pd.DataFrame(centroid_nodes), silhouette_score_val

# --------------------------
# 步骤6：结果汇总与保存
# --------------------------
def save_results(all_cluster_results, all_centroid_nodes, cluster_scores):
    """汇总并保存所有聚类结果"""
    # 保存购买者-子簇对应关系
    final_purchaser_cluster = pd.concat(all_cluster_results, ignore_index=True)
    final_purchaser_cluster.to_csv(f"{OUTPUT_DIR}/purchaser_subcluster_mapping.csv", index=False, encoding='gbk')
    
    # 保存子簇中心节点
    final_centroid_nodes = pd.concat(all_centroid_nodes, ignore_index=True)
    final_centroid_nodes.to_csv(f"{OUTPUT_DIR}/subcluster_centroid_nodes.csv", index=False, encoding='gbk')
    
    # 保存每个簇的聚类评估分数
    cluster_score_df = pd.DataFrame([
        {'簇ID': cid, 'k值': k, '轮廓系数': score}
        for (cid, k), score in cluster_scores.items()
    ])
    cluster_score_df.to_csv(f"{OUTPUT_DIR}/cluster_evaluation_scores.csv", index=False, encoding='gbk')
    
    print(f"\n所有结果已保存至目录：{OUTPUT_DIR}")
    print("\n购买者-子簇映射示例：")
    print(final_purchaser_cluster.head(10))
    print("\n子簇中心节点示例：")
    print(final_centroid_nodes.head(10))

# --------------------------
# 主函数
# --------------------------
def main():
    try:
        # 1. 读取簇ID与k值映射
        cluster_k_mapping = load_cluster_k_mapping(CLUSTER_K_FILE)
        
        # 2. 读取并预处理原始数据
        raw_data = load_and_preprocess_raw_data(INPUT_DATA_FILE, SHEET_NAME)
        
        # 3. 按Cluster列分组，对每个簇执行聚类
        all_cluster_results = []
        all_centroid_nodes = []
        cluster_scores = {}
        
        # 遍历每个簇
        for cluster_id in raw_data['Cluster'].unique():
            cluster_id_int = int(cluster_id)
            # 检查当前簇是否有对应的k值
            if cluster_id_int not in cluster_k_mapping:
                print(f"警告：簇ID {cluster_id} 无对应的k值，跳过该簇")
                continue
            
            # 获取当前簇的k值和数据
            k_value = cluster_k_mapping[cluster_id_int]
            cluster_data = raw_data[raw_data['Cluster'] == cluster_id].copy()
            
            # 执行聚类
            cluster_result, centroid_nodes, score = cluster_single_group(cluster_id_int, cluster_data, k_value)
            
            # 保存结果
            if cluster_result is not None:
                cluster_result['原簇ID'] = cluster_id_int
                all_cluster_results.append(cluster_result)
            if centroid_nodes is not None and not centroid_nodes.empty:
                all_centroid_nodes.append(centroid_nodes)
            cluster_scores[(cluster_id_int, k_value)] = score
        
        # 4. 保存汇总结果
        save_results(all_cluster_results, all_centroid_nodes, cluster_scores)
        
        print("\n所有簇的聚类任务完成！")
    
    except Exception as e:
        print(f"程序执行出错：{str(e)}")

if __name__ == "__main__":
    main()