import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
import random

# 1. 读取文件（指定编码格式）
df = pd.read_csv('./data/Aligned Paired Data/compressed_features_expanded.csv', encoding='UTF-8')
centroid_nodes_df = pd.read_csv('./data/Aligned Paired Data/cluster_centroid_nodes.csv', encoding='UTF-8')

# 提取特征列和标识列（保留local_node_id用于定位实际点）
feature_cols = [col for col in df.columns if col not in ['global_index', 'ego_id', 'local_node_id']]
id_cols = ['global_index', 'local_node_id']  # 用于定位实际点的标识列

# 存储每个 ego_id 的关键信息（用于后续生成表格）
best_k_summary = []          # 最优k及轮廓系数汇总
sample_cluster_labels = []   # 每个样本的聚类标签汇总
centroid_points_summary = [] # 聚类中心对应点汇总
silhouette_scores = {}       # 存储每个ego_id不同k的轮廓系数

# 2. 对每个ego_id进行聚类（使用最优k）
k_range = range(2, 50)  # k值评估范围

for ego in df['ego_id'].unique():
    # 提取当前ego的所有样本（包含特征和标识列）
    ego_data = df[df['ego_id'] == ego].copy()
    ego_features = ego_data[feature_cols]
    sample_count = len(ego_features)
    
    # 存储当前ego不同k的轮廓系数
    ego_scores = {}
    best_score = -1
    best_k = None
    
    # 计算不同k的轮廓系数并输出
    print(f"\n=== ego_id {ego} 不同k值的轮廓系数 ===")
    if sample_count >= 2:  # 至少需要2个样本才能计算轮廓系数
        for k in k_range:
            if sample_count >= k:  # 样本数需≥聚类数
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(ego_features)
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(ego_features, labels)
                    ego_scores[k] = score
                    print(f"k={k}: 轮廓系数={score:.4f}")
                    if score > best_score:
                        best_score = score
                        best_k = k
            else:
                print(f"k={k}: 样本数={sample_count} < k={k}，无法计算轮廓系数")
    else:
        print(f"样本数={sample_count} < 2，无法计算轮廓系数")
    
    # 保存当前ego的轮廓系数
    silhouette_scores[ego] = ego_scores
    
    # 确定最终聚类k值：优先用最优k，否则用样本数（每个样本一类）
    if best_k is not None:
        final_k = best_k
        print(f"ego_id {ego} 最优k值：{final_k}，对应轮廓系数：{best_score:.4f}")
    else:
        final_k = sample_count
        best_score = np.nan  # 无法计算轮廓系数时设为NaN
        print(f"警告：ego_id {ego} 无法确定最优k，聚类数调整为样本数：{final_k}")
    
    # 保存最优k汇总信息
    best_k_summary.append({
        'ego_id': ego,
        'sample_count': sample_count,
        'best_k': final_k,
        'best_silhouette_score': best_score
    })
    
    # 执行最终聚类
    kmeans_final = KMeans(n_clusters=final_k, random_state=42)
    ego_data['cluster_label'] = kmeans_final.fit_predict(ego_features)
    
    # 保存每个样本的聚类标签（用于生成表格）
    for _, row in ego_data.iterrows():
        sample_cluster_labels.append({
            'ego_id': ego,
            'global_index': row['global_index'],
            'local_node_id': row['local_node_id'],
            'cluster_label': row['cluster_label']
        })
    
    # 找到每个簇中距离中心最近的实际点（中心对应点），并保存到汇总表
    for cluster_id in range(final_k):
        # 筛选当前簇的所有样本
        cluster_samples = ego_data[ego_data['cluster_label'] == cluster_id]
        cluster_features = cluster_samples[feature_cols]
        
        # 计算簇内每个样本到中心的距离
        centroid = kmeans_final.cluster_centers_[cluster_id:cluster_id+1]
        distances = euclidean_distances(cluster_features, centroid).flatten()
        
        # 找到距离最近的样本（中心对应点）
        min_dist_idx = np.argmin(distances)
        centroid_point = cluster_samples.iloc[min_dist_idx]
        
        # 构建中心对应点汇总数据
        centroid_point_dict = {
            'ego_id': ego,
            'cluster_id': cluster_id,
            'global_index': centroid_point['global_index'],
            'local_node_id': centroid_point['local_node_id'],
            'min_distance_to_centroid': distances[min_dist_idx]
        }
        # 添加特征列数据
        for feat in feature_cols:
            centroid_point_dict[feat] = centroid_point[feat]
        centroid_points_summary.append(centroid_point_dict)

# 3. 生成并保存表格数据
# 3.1 最优k及轮廓系数汇总表
best_k_df = pd.DataFrame(best_k_summary)
best_k_df.to_csv('./data/Aligned Paired Data/ego_best_k_summary.csv', index=False, encoding='UTF-8')
print("\n✅ 最优k汇总表已保存为：./data/Aligned Paired Data/ego_best_k_summary.csv")

# 3.2 样本聚类标签表
sample_cluster_df = pd.DataFrame(sample_cluster_labels)
sample_cluster_df.to_csv('./data/Aligned Paired Data/sample_cluster_labels.csv', index=False, encoding='UTF-8')
print("✅ 样本聚类标签表已保存为：./data/Aligned Paired Data/sample_cluster_labels.csv")

# 3.3 聚类中心对应点表
centroid_points_df = pd.DataFrame(centroid_points_summary)
centroid_points_df.to_csv('./data/Aligned Paired Data/cluster_centroid_points.csv', index=False, encoding='UTF-8')
print("✅ 聚类中心对应点表已保存为：./data/Aligned Paired Data/cluster_centroid_points.csv")

# 4. 10个ego节点与10个购买者节点随机对应
random.seed(42)
selected_egos = random.sample(list(df['ego_id'].unique()), 10)
selected_centroids = centroid_nodes_df['中心节点（购买者）'].tolist()[:10]
random.shuffle(selected_centroids)
ego_buyer_mapping = dict(zip(selected_egos, selected_centroids))

# 生成并保存ego与购买者对应关系表
ego_buyer_df = pd.DataFrame(list(ego_buyer_mapping.items()), columns=['ego_id', '中心节点（购买者）'])
ego_buyer_df.to_csv('./data/Aligned Paired Data/ego_buyer_mapping.csv', index=False, encoding='UTF-8')
print("✅ ego与购买者对应关系表已保存为：./data/Aligned Paired Data/ego_buyer_mapping.csv")

# 5. 打印关键汇总结果
print("\n=== 各ego_id最优k及得分汇总 ===")
print(best_k_df.to_string(index=False))

print("\n=== ego节点与购买者节点对应关系 ===")
print(ego_buyer_df.to_string(index=False))