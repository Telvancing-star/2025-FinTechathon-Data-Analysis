# 寻找出现多次的节点
import pandas as pd
import json
from collections import Counter


def count_local_node_ids(csv_file_path, output_json_path):
    """
    统计CSV文件中local_node_id的出现次数，并将出现多次的节点保存为JSON文件

    参数:
    csv_file_path: CSV文件路径
    output_json_path: 输出JSON文件路径
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file_path)

        # 检查local_node_id列是否存在
        if 'local_node_id' not in df.columns:
            print("错误：CSV文件中没有找到'local_node_id'列")
            return

        # 统计每个local_node_id的出现次数
        id_counts = Counter(df['local_node_id'])

        # 过滤出出现次数大于1的节点
        multiple_occurrences = {node_id: count for node_id, count in id_counts.items() if count > 1}

        # 按节点ID排序（可选）
        multiple_occurrences = dict(sorted(multiple_occurrences.items()))

        # 保存到JSON文件
        with open(output_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(multiple_occurrences, json_file, indent=4, ensure_ascii=False)

        print(f"成功处理完成！")
        print(f"总共处理了 {len(df)} 行数据")
        print(f"找到 {len(multiple_occurrences)} 个出现多次的local_node_id")
        print(f"结果已保存到: {output_json_path}")

        # 打印前几个结果作为示例
        if multiple_occurrences:
            print("\n前几个结果示例:")
            for i, (node_id, count) in enumerate(list(multiple_occurrences.items())[:5]):
                print(f"  {node_id}: {count}次")

    except FileNotFoundError:
        print(f"错误：找不到文件 {csv_file_path}")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")


# 使用示例
if __name__ == "__main__":
    # 替换为你的CSV文件路径
    csv_file = "data/Social/compressed_features.csv"  # 修改为你的CSV文件路径
    output_file = "data/Social/multiple_occurrences.json"  # 输出JSON文件路径

    count_local_node_ids(csv_file, output_file)