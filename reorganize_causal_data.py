"""
重组因果推断数据：基于Rubin潜在结果框架
提取意见领袖2630号节点及其邻居的投资记录，生成处理变量T_{i,j}和响应变量Y_{i,j}
- T_{i,j}: 处理变量，表示意见领袖是否投资产品j（0或1）
- Y_{i,j}: 响应变量，表示邻居节点对产品j的投资金额（Denominations），未投资时为0

支持按产品（Name of the Political Party）分别进行因果推断分析
"""
import pandas as pd
import numpy as np
from scipy.stats import norm

def reorganize_causal_data(csv_file_path, ego_node_id=2630, output_file='causal_inference_data.csv', 
                           by_product=False, product_col='Name of the Political Party'):
    """
    重组数据用于因果推断分析
    
    参数:
    csv_file_path: 原始CSV文件路径
    ego_node_id: 意见领袖节点ID（默认2630）
    output_file: 输出文件路径
    by_product: 是否按产品分组分析（默认False）
    product_col: 产品列名（默认'Name of the Political Party'）
    
    返回:
    reorganized_df: 重组后的数据框
    """
    # 读取原始数据
    print(f"正在读取数据文件: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    
    # 移除空行
    df = df.dropna(subset=['local_node_id', 'Round'])
    
    # 确保Round和local_node_id是数值类型
    df['Round'] = df['Round'].astype(int)
    df['local_node_id'] = df['local_node_id'].astype(int)
    
    # 确保Denominations是数值类型
    df['Denominations'] = pd.to_numeric(df['Denominations'], errors='coerce')
    
    print(f"原始数据形状: {df.shape}")
    print(f"轮次范围: {df['Round'].min()} - {df['Round'].max()}")
    print(f"唯一节点数: {df['local_node_id'].nunique()}")
    
    # 如果按产品分析，找出所有唯一的产品
    if by_product:
        products = sorted(df[product_col].dropna().unique())
        print(f"\n=== 按产品分析模式 ===")
        print(f"唯一产品数: {len(products)}")
        print(f"产品列表: {products}")
    else:
        products = [None]  # 不分产品，使用None作为占位符
        print(f"\n=== 总体分析模式 ===")
    
    # 找出所有唯一的邻居节点（不包括意见领袖本身）
    all_nodes = set(df['local_node_id'].unique())
    neighbor_nodes = sorted(list(all_nodes - {ego_node_id}))
    
    print(f"\n意见领袖节点ID: {ego_node_id}")
    print(f"邻居节点数量: {len(neighbor_nodes)}")
    print(f"邻居节点列表（前10个）: {neighbor_nodes[:10]}")
    
    # 找出所有唯一的轮次
    all_rounds = sorted(df['Round'].unique())
    print(f"\n总轮次数: {len(all_rounds)}")
    print(f"轮次列表: {all_rounds}")
    
    # 构建重组数据
    reorganized_data = []
    
    # 对每个产品分别处理
    for product in products:
        if by_product:
            product_df = df[df[product_col] == product].copy()
            print(f"\n--- 产品: {product} ---")
        else:
            product_df = df.copy()
        
        for round_num in all_rounds:
            # 筛选当前轮次和当前产品的数据
            round_data = product_df[product_df['Round'] == round_num]
            
            if len(round_data) == 0:
                continue  # 如果该轮次没有该产品的数据，跳过
            
            # 判断意见领袖是否投资了该产品（检查是否有ego_node_id的记录）
            ego_invested = (round_data['local_node_id'] == ego_node_id).any()
            T_ij = 1 if ego_invested else 0
            
            # 找出当前轮次投资该产品的邻居节点及其投资金额
            neighbor_investments = round_data[
                (round_data['local_node_id'] != ego_node_id) & 
                (round_data['local_node_id'].isin(neighbor_nodes))
            ]
            
            # 计算每个邻居节点对该产品的总投资金额（同一轮次可能有多次投资）
            neighbor_amounts = neighbor_investments.groupby('local_node_id')['Denominations'].sum().to_dict()
            
            # 为每个邻居节点创建记录
            for neighbor_id in neighbor_nodes:
                # Y_{i,j}: 邻居节点在当前轮次对该产品的总投资金额，如果未投资则为0
                Y_ij = neighbor_amounts.get(neighbor_id, 0)
                
                record = {
                    'Round': round_num,
                    'ego_node_id': ego_node_id,
                    'neighbor_node_id': neighbor_id,
                    'T_i_j': T_ij,  # 处理变量：意见领袖是否投资该产品
                    'Y_i_j': Y_ij   # 响应变量：邻居节点对该产品的投资金额（Denominations）
                }
                
                if by_product:
                    record['Product'] = product  # 添加产品列
                
                reorganized_data.append(record)
            
            invested_neighbor_count = len(neighbor_amounts)
            total_investment = sum(neighbor_amounts.values())
            product_label = f"[产品: {product}] " if by_product else ""
            print(f"  {product_label}轮次 {round_num}: 意见领袖投资={ego_invested} (T={T_ij}), "
                  f"投资邻居数={invested_neighbor_count}/{len(neighbor_nodes)}, "
                  f"总投资金额={total_investment:,.0f}")
    
    # 创建重组后的数据框
    reorganized_df = pd.DataFrame(reorganized_data)
    
    # 保存结果
    reorganized_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n重组数据已保存到: {output_file}")
    print(f"重组后数据形状: {reorganized_df.shape}")
    print(f"\n数据预览:")
    print(reorganized_df.head(20))
    
    # 统计信息
    if by_product and 'Product' in reorganized_df.columns:
        # 按产品分别统计
        print(f"\n{'='*80}")
        print(f"=== 按产品分别统计 ===")
        print(f"{'='*80}")
        
        product_results = []
        
        for product in sorted(reorganized_df['Product'].unique()):
            product_df = reorganized_df[reorganized_df['Product'] == product]
            
            print(f"\n{'='*80}")
            print(f"产品: {product}")
            print(f"{'='*80}")
            print(f"总记录数: {len(product_df)}")
            print(f"处理组 (T=1) 记录数: {(product_df['T_i_j'] == 1).sum()}")
            print(f"控制组 (T=0) 记录数: {(product_df['T_i_j'] == 0).sum()}")
            print(f"有投资的邻居节点记录数: {(product_df['Y_i_j'] > 0).sum()}")
            print(f"未投资的邻居节点记录数: {(product_df['Y_i_j'] == 0).sum()}")
            print(f"响应变量Y统计:")
            print(f"  平均值: {product_df['Y_i_j'].mean():,.2f}")
            print(f"  中位数: {product_df['Y_i_j'].median():,.2f}")
            print(f"  最大值: {product_df['Y_i_j'].max():,.2f}")
            print(f"  最小值: {product_df['Y_i_j'].min():,.2f}")
            print(f"  标准差: {product_df['Y_i_j'].std():,.2f}")
            
            # 计算平均因果效应估计量
            treatment_group = product_df[product_df['T_i_j'] == 1]
            control_group = product_df[product_df['T_i_j'] == 0]
            
            if len(treatment_group) > 0 and len(control_group) > 0:
                Y_bar_1 = treatment_group['Y_i_j'].mean()
                Y_bar_0 = control_group['Y_i_j'].mean()
                tau_hat = Y_bar_1 - Y_bar_0
                
                # 计算标准误（Neyman方差估计）
                n1 = len(treatment_group)
                n0 = len(control_group)
                
                var_Y1 = treatment_group['Y_i_j'].var(ddof=0) if n1 > 0 else 0
                var_Y0 = control_group['Y_i_j'].var(ddof=0) if n0 > 0 else 0
                
                se_tau = np.sqrt(var_Y1 / n1 + var_Y0 / n0) if n1 > 0 and n0 > 0 else 0
                
                print(f"\n=== Neyman平均因果效应估计 ===")
                print(f"处理组 (T=1) 样本数: {n1}")
                print(f"控制组 (T=0) 样本数: {n0}")
                print(f"Ȳ(1) = {Y_bar_1:,.2f}")
                print(f"Ȳ(0) = {Y_bar_0:,.2f}")
                print(f"τ̂ = Ȳ(1) - Ȳ(0) = {tau_hat:,.2f}")
                print(f"标准误 SE(τ̂) = {se_tau:,.2f}")
                z_score = np.nan
                p_value = np.nan
                
                if se_tau > 0:
                    z_score = tau_hat / se_tau
                    print(f"Z统计量 = {z_score:.4f}")
                    if abs(z_score) < 10:
                        p_value = 2 * (1 - norm.cdf(abs(z_score)))
                        print(f"P值 (双尾) ≈ {p_value:.6f}")
                    else:
                        p_value = 0.000001
                        print(f"P值 (双尾) < 0.000001")
                
                # 保存结果用于汇总
                product_results.append({
                    'Product': product,
                    'n1': n1,
                    'n0': n0,
                    'Y_bar_1': Y_bar_1,
                    'Y_bar_0': Y_bar_0,
                    'tau_hat': tau_hat,
                    'se_tau': se_tau,
                    'z_score': z_score,
                    'p_value': p_value
                })
        
        # 汇总表格
        if product_results:
            print(f"\n{'='*80}")
            print(f"=== 所有产品因果效应汇总 ===")
            print(f"{'='*80}")
            summary_df = pd.DataFrame(product_results)
            print(summary_df.to_string(index=False))
            
            # 保存汇总结果
            summary_file = output_file.replace('.csv', '_summary_by_product.csv')
            summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
            print(f"\n汇总结果已保存到: {summary_file}")
    
    else:
        # 总体统计（不分产品）
        print(f"\n=== 数据统计 ===")
        print(f"总记录数: {len(reorganized_df)}")
        print(f"处理组 (T=1) 记录数: {(reorganized_df['T_i_j'] == 1).sum()}")
        print(f"控制组 (T=0) 记录数: {(reorganized_df['T_i_j'] == 0).sum()}")
        print(f"有投资的邻居节点记录数: {(reorganized_df['Y_i_j'] > 0).sum()}")
        print(f"未投资的邻居节点记录数: {(reorganized_df['Y_i_j'] == 0).sum()}")
        print(f"响应变量Y统计:")
        print(f"  平均值: {reorganized_df['Y_i_j'].mean():,.2f}")
        print(f"  中位数: {reorganized_df['Y_i_j'].median():,.2f}")
        print(f"  最大值: {reorganized_df['Y_i_j'].max():,.2f}")
        print(f"  最小值: {reorganized_df['Y_i_j'].min():,.2f}")
        print(f"  标准差: {reorganized_df['Y_i_j'].std():,.2f}")
        
        # 计算平均因果效应估计量的基本统计
        treatment_group = reorganized_df[reorganized_df['T_i_j'] == 1]
        control_group = reorganized_df[reorganized_df['T_i_j'] == 0]
        
        if len(treatment_group) > 0 and len(control_group) > 0:
            Y_bar_1 = treatment_group['Y_i_j'].mean()
            Y_bar_0 = control_group['Y_i_j'].mean()
            tau_hat = Y_bar_1 - Y_bar_0
            
            # 计算标准误（Neyman方差估计）
            n1 = len(treatment_group)
            n0 = len(control_group)
            
            var_Y1 = treatment_group['Y_i_j'].var(ddof=0) if n1 > 0 else 0
            var_Y0 = control_group['Y_i_j'].var(ddof=0) if n0 > 0 else 0
            
            se_tau = np.sqrt(var_Y1 / n1 + var_Y0 / n0) if n1 > 0 and n0 > 0 else 0
            
            print(f"\n=== Neyman平均因果效应估计 ===")
            print(f"处理组 (T=1) 样本数: {n1}")
            print(f"控制组 (T=0) 样本数: {n0}")
            print(f"Ȳ(1) = {Y_bar_1:,.2f}")
            print(f"Ȳ(0) = {Y_bar_0:,.2f}")
            print(f"τ̂ = Ȳ(1) - Ȳ(0) = {tau_hat:,.2f}")
            print(f"标准误 SE(τ̂) = {se_tau:,.2f}")
            if se_tau > 0:
                z_score = tau_hat / se_tau
                print(f"Z统计量 = {z_score:.4f}")
    
    return reorganized_df


if __name__ == "__main__":
    # 执行数据重组 - 按产品分析
    input_file = "ego_network_records_node_2630.0.csv"
    output_file = "causal_inference_data_by_product.csv"
    
    # 按产品分别进行因果推断分析
    reorganized_df = reorganize_causal_data(
        input_file, 
        ego_node_id=2630, 
        output_file=output_file,
        by_product=True,  # 按产品分析
        product_col='Name of the Political Party'
    )
    
    print("\n数据重组完成！")

