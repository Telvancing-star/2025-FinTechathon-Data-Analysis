# 获取压缩特征矩阵
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import os


class EgoFeatureProcessor:
    def __init__(self, ego_nodes, data_path):
        """
        ego_nodes: ego node ID 列表
        data_path: network 数据所在目录
        """
        self.ego_nodes = ego_nodes
        self.data_path = data_path
        self.parent_categories = None
        self.feat_dim = None

    def _parse_featnames(self, ego_id):
        """解析.featnames文件，提取父类别"""
        featnames_file = os.path.join(self.data_path, f"{ego_id}.featnames")
        categories = set()

        with open(featnames_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) < 2:
                    continue
                feat_path = parts[1]
                # 提取父类别（去掉最后一个叶子节点）
                path_parts = feat_path.split(';')
                if len(path_parts) > 1:
                    parent = ';'.join(path_parts[:-1])
                    categories.add(parent)

        return sorted(list(categories))

    def _load_feat_file(self, ego_id):
        """加载.feat文件，返回节点特征矩阵"""
        feat_file = os.path.join(self.data_path, f"{ego_id}.feat")
        node_data = []

        with open(feat_file, 'r') as f:
            for line in f:
                parts = list(map(int, line.strip().split()))
                node_id = parts[0]
                feature_vector = parts[1:]
                node_data.append((node_id, feature_vector))

        # 按节点ID排序
        node_data.sort(key=lambda x: x[0])
        node_ids = [x[0] for x in node_data]
        feature_matrix = np.array([x[1] for x in node_data])

        return node_ids, feature_matrix

    def _compress_features(self, feature_matrix, featnames, parent_categories):
        """将原始特征压缩为父类别级别的特征"""
        compressed_features = []

        for node_features in feature_matrix:
            compressed_vec = []

            for parent in parent_categories:
                # 找到属于该父类的所有特征维度
                child_indices = []
                for i, feat_path in enumerate(featnames):
                    if feat_path.startswith(parent + ';'):
                        child_indices.append(i)

                # 统计该父类下有多少个特征为1
                if child_indices:
                    parent_sum = np.sum([node_features[i] for i in child_indices])
                else:
                    parent_sum = 0

                compressed_vec.append(parent_sum)

            # 添加常数项1作为第一维（对应论文中的常数特征）
            compressed_vec = [1] + compressed_vec
            compressed_features.append(compressed_vec)

        return np.array(compressed_features)

    def process_ego_network(self, ego_id):
        """处理单个ego-network的特征"""
        # 1. 解析特征名称
        featnames = []
        with open(os.path.join(self.data_path, f"{ego_id}.featnames"), 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) >= 2:
                    featnames.append(parts[1])

        # 2. 提取父类别（第一次运行时初始化）
        if self.parent_categories is None:
            self.parent_categories = self._parse_featnames(ego_id)
            self.feat_dim = len(self.parent_categories) + 1  # +1 for constant feature

        # 3. 加载特征数据
        node_ids, feature_matrix = self._load_feat_file(ego_id)

        # 4. 压缩特征
        X_compressed = self._compress_features(feature_matrix, featnames, self.parent_categories)

        return node_ids, X_compressed

    def get_all_features(self):
        """获取所有ego-network的压缩特征矩阵，并添加缺失的节点"""
        all_X = []
        all_node_info = []  # 保存节点信息 (ego_id, local_node_id, global_index)

        global_idx = 0

        for ego_id in self.ego_nodes:
            print(f"Processing ego network {ego_id}...")

            try:
                node_ids, X_ego = self.process_ego_network(ego_id)

                # 为当前ego-network的所有节点添加信息
                for local_idx, node_id in enumerate(node_ids):
                    all_node_info.append({
                        'ego_id': ego_id,
                        'local_node_id': node_id,
                        'global_index': global_idx
                    })
                    global_idx += 1

                all_X.append(X_ego)

            except FileNotFoundError as e:
                print(f"Warning: Files for ego {ego_id} not found: {e}")
                continue
            except Exception as e:
                print(f"Error processing ego {ego_id}: {e}")
                continue

        if not all_X:
            raise ValueError("No ego networks were successfully processed")

        # 合并所有特征矩阵
        X_combined = np.vstack(all_X)

        # 添加四个缺失的节点
        X_combined, all_node_info = self._add_missing_nodes(X_combined, all_node_info, global_idx)

        return X_combined, all_node_info

    def _add_missing_nodes(self, X_combined, all_node_info, start_global_idx):
        """
        添加四个缺失的节点到特征矩阵和节点信息中

        参数:
        X_combined: 当前的特征矩阵
        all_node_info: 当前的节点信息列表
        start_global_idx: 下一个可用的global_index

        返回:
        更新后的X_combined和all_node_info
        """
        print("Adding missing nodes...")

        # 缺失的节点信息
        missing_nodes = [
            {'local_node_id': 686, 'ego_id': -1},
            {'local_node_id': 1912, 'ego_id': -1},
            {'local_node_id': 3437, 'ego_id': -1},
            {'local_node_id': 3980, 'ego_id': -1}
        ]

        # 获取特征向量的维度
        num_features = X_combined.shape[1]

        # 创建缺失节点的特征向量 (feature_0=1, 其他特征=0)
        missing_features = np.zeros((4, num_features))
        missing_features[:, 0] = 1  # 设置feature_0为1

        # 添加节点信息
        for i, node_info in enumerate(missing_nodes):
            all_node_info.append({
                'ego_id': node_info['ego_id'],
                'local_node_id': node_info['local_node_id'],
                'global_index': start_global_idx + i
            })

        # 合并特征矩阵
        X_combined_with_missing = np.vstack([X_combined, missing_features])

        print(
            f"Added {len(missing_nodes)} missing nodes with global indices {start_global_idx}-{start_global_idx + len(missing_nodes) - 1}")

        return X_combined_with_missing, all_node_info

    def get_feature_description(self):
        """返回特征维度的描述"""
        if self.parent_categories is None:
            raise ValueError("Please process at least one ego network first")

        feature_desc = ["constant_feature_1"] + self.parent_categories
        return feature_desc

    def save_features_to_csv(self, X, node_info, feature_desc, output_path):
        """将特征矩阵保存为CSV文件"""
        import pandas as pd

        # 创建列名
        column_names = [f"feature_{i}" for i in range(X.shape[1])]

        # 创建DataFrame
        df = pd.DataFrame(X, columns=column_names)

        # 添加节点信息列
        df['ego_id'] = [info['ego_id'] for info in node_info]
        df['local_node_id'] = [info['local_node_id'] for info in node_info]
        df['global_index'] = [info['global_index'] for info in node_info]

        # 重新排列列顺序，让节点信息在前
        cols = ['global_index', 'ego_id', 'local_node_id'] + column_names
        df = df[cols]

        # 保存CSV
        df.to_csv(output_path, index=False)

        # 保存特征描述到单独文件
        desc_df = pd.DataFrame({
            'feature_index': range(len(feature_desc)),
            'feature_description': feature_desc
        })
        desc_path = output_path.replace('.csv', '_description.csv')
        desc_df.to_csv(desc_path, index=False)

        print(f"特征矩阵已保存至: {output_path}")
        print(f"特征描述已保存至: {desc_path}")


# 使用示例
def main():
    # 你的ego节点列表
    ego_nodes = [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]
    data_path = "./data/Social"  # 替换为实际路径

    # 创建处理器
    processor = EgoFeatureProcessor(ego_nodes, data_path)

    # 获取所有压缩特征
    X, node_info = processor.get_all_features()

    # 获取特征描述
    feature_desc = processor.get_feature_description()

    print(f"特征矩阵形状: {X.shape}")  # (总节点数, 特征维度)
    print(f"特征维度描述:")
    for i, desc in enumerate(feature_desc):
        print(f"  维度 {i}: {desc}")

    print(f"前5个节点的信息:")
    for i in range(min(5, len(node_info))):
        info = node_info[i]
        print(f"  全局索引 {info['global_index']}: ego={info['ego_id']}, 本地节点={info['local_node_id']}")

    csv_output_path = "data/Social/compressed_features.csv"
    processor.save_features_to_csv(X, node_info, feature_desc, csv_output_path)

    # 现在X可以用于PoRe-LSM模型
    # 例如: pop_model = Pop(N=len(X), beta=beta, delta=delta, C_min=C_min, C_max=C_max)
    # 然后将X赋值给相应的属性

    return X, node_info, feature_desc


if __name__ == "__main__":
    X, node_info, feature_desc = main()