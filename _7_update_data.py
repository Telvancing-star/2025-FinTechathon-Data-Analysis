import json
import pandas as pd
from collections import defaultdict
import numpy as np


class CustomJSONEncoder(json.JSONEncoder):
    """自定义JSON编码器，用于处理numpy.int64等类型"""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        return super().default(obj)


def update_csv_node_ids(csv_file, duplicates_file, node_id_column, output_file):
    """
    更新CSV文件中的节点ID：为重复节点分配新的节点ID

    参数:
    csv_file: 原始CSV文件路径
    duplicates_file: 重复节点信息JSON文件路径
    node_id_column: CSV文件中节点ID的列名
    output_file: 输出文件路径
    """

    # 读取重复节点信息
    with open(duplicates_file, 'r') as f:
        duplicates = json.load(f)

    print(f"找到 {len(duplicates)} 个需要更新的节点")

    # 读取原始CSV文件
    df = pd.read_csv(csv_file)
    print(f"原始CSV: {len(df)} 条记录")

    # 找到最大的节点ID
    max_node_id = df[node_id_column].max()
    print(f"最大节点ID: {max_node_id}")

    # 记录每个原始节点对应的新节点ID映射
    original_to_new_ids = defaultdict(list)
    next_new_id = max_node_id + 1

    # 创建节点ID映射字典
    node_id_mapping = {}

    for original_id, count in duplicates.items():
        original_id = int(original_id)

        # 为每个重复创建新节点ID (包括原始节点本身)
        new_ids = [original_id]  # 第一个使用原始ID
        for i in range(count - 1):  # 后续的创建新ID
            new_id = next_new_id
            new_ids.append(new_id)
            next_new_id += 1

        # 记录映射关系
        original_to_new_ids[original_id] = new_ids

    print(f"将使用 {next_new_id - max_node_id - 1} 个新节点ID")

    # 为每个重复的节点分配新的节点ID
    updated_records = []

    for original_id, new_ids in original_to_new_ids.items():
        # 找到该原始节点的所有记录
        original_records = df[df[node_id_column] == original_id]

        if len(original_records) == 0:
            print(f"警告: 节点 {original_id} 在CSV中不存在")
            continue

        # 检查记录数量是否匹配
        if len(original_records) != len(new_ids):
            print(f"警告: 节点 {original_id} 的记录数 ({len(original_records)}) 与分配ID数 ({len(new_ids)}) 不匹配")
            # 使用可用的ID数量
            use_ids = new_ids[:len(original_records)]
        else:
            use_ids = new_ids

        # 为每条记录分配新的节点ID
        for i, (idx, record) in enumerate(original_records.iterrows()):
            if i < len(use_ids):
                updated_record = record.copy()
                updated_record[node_id_column] = use_ids[i]
                updated_records.append(updated_record)

    # 获取不需要更新的记录（不在重复节点列表中的记录）
    non_duplicate_records = df[~df[node_id_column].isin(original_to_new_ids.keys())]

    # 合并更新后的记录和不需要更新的记录
    final_df = pd.concat([pd.DataFrame(updated_records), non_duplicate_records], ignore_index=True)

    print(f"更新后CSV: {len(final_df)} 条记录")

    # 保存更新后的CSV文件
    final_df.to_csv(output_file, index=False)

    # 保存节点映射信息（使用自定义编码器）
    mapping_file = output_file.replace('.csv', '_mapping.json')
    with open(mapping_file, 'w') as f:
        # 将numpy类型转换为Python原生类型
        serializable_mapping = {}
        for orig_id, new_ids in original_to_new_ids.items():
            serializable_mapping[int(orig_id)] = [int(id) for id in new_ids]

        json.dump(serializable_mapping, f, indent=2, cls=CustomJSONEncoder)

    print(f"更新后的CSV已保存到: {output_file}")
    print(f"节点映射信息已保存到: {mapping_file}")

    return final_df, original_to_new_ids


def verify_csv_update(original_csv, updated_csv, duplicates_file, node_id_column):
    """
    验证CSV更新结果
    """
    # 读取原始CSV
    original_df = pd.read_csv(original_csv)

    # 读取更新后的CSV
    updated_df = pd.read_csv(updated_csv)

    # 读取重复节点信息
    with open(duplicates_file, 'r') as f:
        duplicates = json.load(f)

    print("\n验证结果:")
    print(f"原始CSV: {len(original_df)} 条记录")
    print(f"更新后CSV: {len(updated_df)} 条记录")

    # 检查记录数量是否一致
    if len(original_df) == len(updated_df):
        print("✓ 记录数量一致")
    else:
        print("✗ 记录数量不一致")

    # 检查是否有重复的节点ID
    duplicate_node_ids = updated_df[node_id_column].duplicated().sum()
    if duplicate_node_ids == 0:
        print("✓ 无重复节点ID")
    else:
        print(f"✗ 发现 {duplicate_node_ids} 个重复节点ID")


if __name__ == "__main__":
    # 文件路径配置 - 请根据你的实际情况修改
    csv_file = "data/Social/compressed_features.csv"  # 替换为你的CSV文件路径
    duplicates_file = "data/Social/multiple_occurrences.json"
    output_file = "data/Social/compressed_features_expanded.csv"
    node_id_column = "local_node_id"  # 替换为你的CSV中节点ID的列名

    # 执行CSV节点ID更新
    updated_df, node_mapping = update_csv_node_ids(
        csv_file, duplicates_file, node_id_column, output_file
    )

    # 验证结果
    verify_csv_update(csv_file, output_file, duplicates_file, node_id_column)

    # 打印一些统计信息
    print("\n更新统计:")
    for original_id, new_ids in list(node_mapping.items())[:5]:  # 只显示前5个
        print(f"节点 {original_id} -> 分配ID {new_ids}")

    if len(node_mapping) > 5:
        print(f"... 还有 {len(node_mapping) - 5} 个节点的映射")
