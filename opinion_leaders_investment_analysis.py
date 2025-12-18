import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import traceback

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 文件路径
file_path = 'merged_three_keys_with_node_id.xlsx'

# 意见领袖节点ID
opinion_leaders = [2630, 786, 734, 1930, 3987]

try:
    # 读取Excel文件
    print(f"正在读取文件: {file_path}")
    df = pd.read_excel(file_path)
    
    print("\n文件读取成功！")
    print(f"数据总行数: {len(df)}")
    print(f"\n数据列名: {list(df.columns)}")
    
    # 过滤意见领袖的数据
    leaders_data = df[df['local_node_id'].isin(opinion_leaders)]
    print(f"\n意见领袖数据总行数: {len(leaders_data)}")
    
    if len(leaders_data) == 0:
        print("警告：未找到任何意见领袖的数据！")
        exit()
    
    # 创建结果目录
    output_dir = 'opinion_leaders_analysis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. 总体投资概况
    print("\n=== 1. 总体投资概况 ===")
    
    # 确保金额列存在并正确处理
    amount_column = None
    for col in df.columns:
        if 'Denominations' in col or '金额' in col:
            amount_column = col
            break
    
    if amount_column is None:
        print("警告：未找到金额相关列！")
        amount_column = 'Denominations'  # 默认使用Denominations
    
    print(f"使用金额列: {amount_column}")
    
    # 计算每个意见领袖的投资总额和次数
    leader_summary = {}
    for leader in opinion_leaders:
        leader_df = leaders_data[leaders_data['local_node_id'] == leader]
        if len(leader_df) > 0:
            # 尝试转换金额列
            try:
                total_amount = leader_df[amount_column].astype(float).sum()
            except (ValueError, TypeError):
                print(f"警告：节点 {leader} 的金额列无法转换为数字，使用0代替")
                total_amount = 0
            
            leader_summary[leader] = {
                'total_amount': total_amount,
                'transaction_count': len(leader_df),
                'purchaser_names': leader_df['Name of the Purchaser'].unique().tolist() if 'Name of the Purchaser' in leader_df.columns else []
            }
            print(f"节点 {leader}: 投资总额={total_amount}, 交易次数={len(leader_df)}, 购买者={leader_df['Name of the Purchaser'].unique() if 'Name of the Purchaser' in leader_df.columns else 'N/A'}")
    
    # 2. 按政治党派分析投资分布
    print("\n=== 2. 按政治党派分析 ===")
    if 'Name of the Political Party' in leaders_data.columns:
        party_analysis = {}
        for leader in opinion_leaders:
            leader_df = leaders_data[leaders_data['local_node_id'] == leader]
            if len(leader_df) > 0:
                party_groups = leader_df.groupby('Name of the Political Party')
                party_info = {}
                for party, group in party_groups:
                    try:
                        party_amount = group[amount_column].astype(float).sum()
                    except (ValueError, TypeError):
                        party_amount = 0
                    party_info[party] = {
                        'amount': party_amount,
                        'count': len(group)
                    }
                party_analysis[leader] = party_info
                print(f"\n节点 {leader} 的党派投资分布:")
                for party, info in party_info.items():
                    print(f"  {party}: 金额={info['amount']}, 次数={info['count']}")
    else:
        print("警告：未找到政党名称列！")
    
    # 3. 按时间分析投资趋势
    print("\n=== 3. 按时间分析投资趋势 ===")
    if 'Date of Purchase' in leaders_data.columns:
        try:
            # 尝试转换日期列
            leaders_data['Date of Purchase'] = pd.to_datetime(leaders_data['Date of Purchase'], errors='coerce')
            time_analysis = {}
            
            for leader in opinion_leaders:
                leader_df = leaders_data[leaders_data['local_node_id'] == leader]
                if len(leader_df) > 0 and not leader_df['Date of Purchase'].isnull().all():
                    # 按年份和月份分组
                    leader_df['Year_Month'] = leader_df['Date of Purchase'].dt.to_period('M')
                    time_groups = leader_df.groupby('Year_Month')
                    time_info = {}
                    for period, group in time_groups:
                        try:
                            period_amount = group[amount_column].astype(float).sum()
                        except (ValueError, TypeError):
                            period_amount = 0
                        time_info[str(period)] = {
                            'amount': period_amount,
                            'count': len(group)
                        }
                    time_analysis[leader] = time_info
                    print(f"\n节点 {leader} 的时间投资分布:")
                    for period, info in sorted(time_info.items()):
                        print(f"  {period}: 金额={info['amount']}, 次数={info['count']}")
        except Exception as e:
            print(f"时间分析出错: {e}")
    else:
        print("警告：未找到购买日期列！")
    
    # 4. 金额分布分析
    print("\n=== 4. 金额分布分析 ===")
    try:
        leaders_data[amount_column] = pd.to_numeric(leaders_data[amount_column], errors='coerce')
        amount_analysis = {}
        
        for leader in opinion_leaders:
            leader_df = leaders_data[leaders_data['local_node_id'] == leader]
            if len(leader_df) > 0:
                amounts = leader_df[amount_column].dropna()
                if len(amounts) > 0:
                    amount_analysis[leader] = {
                        'mean': amounts.mean(),
                        'median': amounts.median(),
                        'std': amounts.std(),
                        'min': amounts.min(),
                        'max': amounts.max()
                    }
                    print(f"\n节点 {leader} 的金额统计:")
                    print(f"  平均值: {amounts.mean()}")
                    print(f"  中位数: {amounts.median()}")
                    print(f"  标准差: {amounts.std()}")
                    print(f"  最小值: {amounts.min()}")
                    print(f"  最大值: {amounts.max()}")
    except Exception as e:
        print(f"金额分布分析出错: {e}")
    
    # 5. 生成可视化图表
    print("\n=== 5. 生成可视化图表 ===")
    
    # 5.1 各意见领袖投资总额对比
    try:
        leaders = list(leader_summary.keys())
        amounts = [leader_summary[l]['total_amount'] for l in leaders]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar([f'节点 {l}' for l in leaders], amounts)
        plt.title('各意见领袖投资总额对比')
        plt.xlabel('意见领袖节点ID')
        plt.ylabel('投资总额')
        plt.xticks(rotation=45)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1*max(amounts),
                    f'{height:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'total_investment_comparison.png'), dpi=300)
        plt.close()
        print("生成总投资对比图成功")
    except Exception as e:
        print(f"生成总投资对比图失败: {e}")
    
    # 5.2 各意见领袖投资次数对比
    try:
        counts = [leader_summary[l]['transaction_count'] for l in leaders]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar([f'节点 {l}' for l in leaders], counts)
        plt.title('各意见领袖投资次数对比')
        plt.xlabel('意见领袖节点ID')
        plt.ylabel('投资次数')
        plt.xticks(rotation=45)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1*max(counts),
                    f'{height}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'transaction_count_comparison.png'), dpi=300)
        plt.close()
        print("生成投资次数对比图成功")
    except Exception as e:
        print(f"生成投资次数对比图失败: {e}")
    
    # 5.3 政党投资分布饼图（对每个意见领袖）
    if 'Name of the Political Party' in leaders_data.columns:
        for leader in opinion_leaders:
            try:
                leader_df = leaders_data[leaders_data['local_node_id'] == leader]
                if len(leader_df) > 0:
                    party_groups = leader_df.groupby('Name of the Political Party')
                    party_amounts = []
                    party_names = []
                    
                    for party, group in party_groups:
                        try:
                            amount = group[amount_column].astype(float).sum()
                            if amount > 0:
                                party_amounts.append(amount)
                                party_names.append(party)
                        except:
                            continue
                    
                    if len(party_amounts) > 0:
                        plt.figure(figsize=(10, 8))
                        plt.pie(party_amounts, labels=party_names, autopct='%1.1f%%', startangle=90)
                        plt.title(f'节点 {leader} 的政党投资分布')
                        plt.axis('equal')
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, f'party_distribution_node_{leader}.png'), dpi=300)
                        plt.close()
                        print(f"生成节点 {leader} 政党分布饼图成功")
            except Exception as e:
                print(f"生成节点 {leader} 政党分布饼图失败: {e}")
    
    # 6. 保存分析结果到Excel
    print("\n=== 6. 保存分析结果 ===")
    try:
        summary_data = []
        for leader, info in leader_summary.items():
            summary_data.append({
                '节点ID': leader,
                '投资总额': info['total_amount'],
                '投资次数': info['transaction_count'],
                '购买者数量': len(info['purchaser_names']),
                '购买者名称': ', '.join(info['purchaser_names']) if len(info['purchaser_names']) <= 5 else ', '.join(info['purchaser_names'][:5]) + '...'
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(output_dir, 'opinion_leaders_investment_summary.xlsx')
        summary_df.to_excel(summary_file, index=False)
        print(f"分析结果已保存到: {summary_file}")
        
        # 保存详细数据
        detail_file = os.path.join(output_dir, 'opinion_leaders_detail_data.xlsx')
        leaders_data.to_excel(detail_file, index=False)
        print(f"详细数据已保存到: {detail_file}")
    except Exception as e:
        print(f"保存分析结果失败: {e}")
    
    print("\n分析完成！所有图表和数据已保存到 opinion_leaders_analysis 目录。")
    
    # 7. 生成简要的文字分析报告
    report = """
# 意见领袖投资情况分析报告

## 总体概览
本次分析了 {total_leaders} 个意见领袖节点的投资情况，共涉及 {total_transactions} 笔交易。

## 主要发现

### 投资规模排名
"""
    
    # 添加排名信息
    sorted_leaders = sorted(leader_summary.items(), key=lambda x: x[1]['total_amount'], reverse=True)
    for i, (leader, info) in enumerate(sorted_leaders, 1):
        report += f"{i}. 节点 {leader}: 投资总额 {info['total_amount']}，交易次数 {info['transaction_count']}\n"
    
    # 添加政党偏好分析
    if 'Name of the Political Party' in leaders_data.columns:
        report += "\n### 政党偏好分析\n"
        for leader in opinion_leaders:
            if leader in party_analysis and party_analysis[leader]:
                top_party = max(party_analysis[leader].items(), key=lambda x: x[1]['amount'])
                report += f"- 节点 {leader} 主要投资于 {top_party[0]}，占总投资额的 {top_party[1]['amount']/leader_summary[leader]['total_amount']*100:.1f}%\n"
    
    report += "\n## 投资模式特征\n"
    
    # 添加平均交易金额分析
    if amount_analysis:
        avg_amounts = [(l, amount_analysis[l]['mean']) for l in amount_analysis]
        avg_amounts.sort(key=lambda x: x[1], reverse=True)
        report += f"- 单笔交易金额最高的是节点 {avg_amounts[0][0]}，平均每笔 {avg_amounts[0][1]:.2f}\n"
        
    report += "\n## 结论与建议\n"
    report += "1. 意见领袖在政治献金中扮演重要角色，投资金额和频率存在显著差异\n"
    report += "2. 不同意见领袖可能存在不同的政党偏好，反映了其政治倾向\n"
    report += "3. 建议进一步分析这些意见领袖的社交网络关系，探索其影响力传播路径\n"
    report += "4. 可以结合时间维度，分析投资行为与重大政治事件的关联性\n"
    
    # 保存报告
    report_file = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"分析报告已保存到: {report_file}")
    
    # 8. 绘制最近五个轮次的投资动态图
    print("\n=== 8. 绘制最近五个轮次投资动态图 ===")
    try:
        # 确保日期列存在并正确处理
        if 'Date of Purchase' in leaders_data.columns:
            # 转换日期列
            leaders_data['Date of Purchase'] = pd.to_datetime(leaders_data['Date of Purchase'], errors='coerce')
            
            # 按年份和月份分组（一个月视为一个轮次）
            leaders_data['Year_Month'] = leaders_data['Date of Purchase'].dt.to_period('M')
            
            # 获取所有有效轮次（有记录的月份）
            valid_periods = sorted(leaders_data['Year_Month'].dropna().unique())
            
            if len(valid_periods) > 0:
                # 取最近的5个轮次
                recent_periods = valid_periods[-5:] if len(valid_periods) >=5 else valid_periods
                
                # 过滤出最近5个轮次的数据
                recent_data = leaders_data[leaders_data['Year_Month'].isin(recent_periods)]
                
                # 准备数据用于绘图
                period_labels = [str(p) for p in recent_periods]
                
                # 创建图表 - 投资金额动态图
                plt.figure(figsize=(14, 8))
                
                # 为每个意见领袖绘制折线图
                for leader in opinion_leaders:
                    leader_recent_data = recent_data[recent_data['local_node_id'] == leader]
                    if len(leader_recent_data) > 0:
                        # 按轮次分组计算投资总额
                        leader_period_amounts = []
                        for period in recent_periods:
                            period_data = leader_recent_data[leader_recent_data['Year_Month'] == period]
                            if len(period_data) > 0:
                                try:
                                    amount = period_data[amount_column].astype(float).sum()
                                except:
                                    amount = 0
                            else:
                                amount = 0
                            leader_period_amounts.append(amount)
                        
                        # 绘制折线图
                        plt.plot(period_labels, leader_period_amounts, marker='o', linewidth=2, markersize=8, label=f'节点 {leader}')
                        
                        # 在每个点上添加数值标签
                        for i, value in enumerate(leader_period_amounts):
                            if value > 0:  # 只在有投资的点上显示标签
                                plt.text(i, value + 0.05 * max(leader_period_amounts), f'{value:.0f}', 
                                        ha='center', va='bottom', fontsize=10)
                
                plt.title('意见领袖最近五个轮次投资动态', fontsize=16)
                plt.xlabel('投资轮次（年月）', fontsize=12)
                plt.ylabel('投资金额', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(fontsize=10)
                plt.tight_layout()
                
                # 保存图表
                dynamic_chart_path = os.path.join(output_dir, 'recent_5_rounds_investment_dynamics.png')
                plt.savefig(dynamic_chart_path, dpi=300)
                plt.close()
                print(f"最近五个轮次投资动态图已保存到: {dynamic_chart_path}")
                
                # 创建堆叠柱状图 - 展示各轮次各意见领袖的投资占比
                plt.figure(figsize=(14, 8))
                
                # 准备堆叠柱状图数据
                stack_data = {}
                for leader in opinion_leaders:
                    leader_recent_data = recent_data[recent_data['local_node_id'] == leader]
                    leader_amounts = []
                    for period in recent_periods:
                        period_data = leader_recent_data[leader_recent_data['Year_Month'] == period]
                        if len(period_data) > 0:
                            try:
                                amount = period_data[amount_column].astype(float).sum()
                            except:
                                amount = 0
                        else:
                            amount = 0
                        leader_amounts.append(amount)
                    stack_data[leader] = leader_amounts
                
                # 绘制堆叠柱状图
                bottom_values = [0] * len(recent_periods)
                for leader, amounts in stack_data.items():
                    if any(amount > 0 for amount in amounts):  # 只绘制有投资的意见领袖
                        bars = plt.bar(period_labels, amounts, bottom=bottom_values, label=f'节点 {leader}', alpha=0.8)
                        
                        # 更新底部值
                        bottom_values = [bottom + amount for bottom, amount in zip(bottom_values, amounts)]
                        
                        # 添加数值标签（只在金额较大时显示）
                        for i, (bar, amount) in enumerate(zip(bars, amounts)):
                            if amount > 0:  # 只在有投资的柱子上显示标签
                                height = bar.get_height()
                                y_pos = bar.get_y() + height / 2
                                # 计算是否显示标签的阈值
                                if height > max(bottom_values) * 0.05:  # 只在高度超过总高度5%时显示
                                    plt.text(bar.get_x() + bar.get_width()/2., y_pos, f'{amount:.0f}',
                                            ha='center', va='center', fontsize=9, color='white')
                
                plt.title('意见领袖最近五个轮次投资分布堆叠图', fontsize=16)
                plt.xlabel('投资轮次（年月）', fontsize=12)
                plt.ylabel('投资金额', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7, axis='y')
                plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                
                # 保存堆叠柱状图
                stack_chart_path = os.path.join(output_dir, 'recent_5_rounds_investment_stack.png')
                plt.savefig(stack_chart_path, dpi=300)
                plt.close()
                print(f"最近五个轮次投资分布堆叠图已保存到: {stack_chart_path}")
                
                # 创建投资次数动态图
                plt.figure(figsize=(14, 8))
                
                # 为每个意见领袖绘制交易次数折线图
                for leader in opinion_leaders:
                    leader_recent_data = recent_data[recent_data['local_node_id'] == leader]
                    if len(leader_recent_data) > 0:
                        # 按轮次分组计算交易次数
                        leader_period_counts = []
                        for period in recent_periods:
                            period_count = len(leader_recent_data[leader_recent_data['Year_Month'] == period])
                            leader_period_counts.append(period_count)
                        
                        # 绘制折线图
                        plt.plot(period_labels, leader_period_counts, marker='s', linewidth=2, markersize=8, label=f'节点 {leader}')
                        
                        # 在每个点上添加数值标签
                        for i, value in enumerate(leader_period_counts):
                            if value > 0:  # 只在有交易的点上显示标签
                                plt.text(i, value + 0.1, f'{value}', 
                                        ha='center', va='bottom', fontsize=10)
                
                plt.title('意见领袖最近五个轮次交易次数动态', fontsize=16)
                plt.xlabel('投资轮次（年月）', fontsize=12)
                plt.ylabel('交易次数', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(fontsize=10)
                plt.tight_layout()
                
                # 保存交易次数图表
                count_chart_path = os.path.join(output_dir, 'recent_5_rounds_transaction_counts.png')
                plt.savefig(count_chart_path, dpi=300)
                plt.close()
                print(f"最近五个轮次交易次数动态图已保存到: {count_chart_path}")
            else:
                print("警告：未找到有效的投资轮次数据！")
        else:
            print("警告：未找到购买日期列，无法绘制投资动态图！")
    except Exception as e:
        print(f"绘制最近五个轮次投资动态图失败: {e}")
        traceback.print_exc()
    
    print("\n" + "="*50)
    print("意见领袖投资分析完成！")
    print("="*50)
    
except Exception as e:
    print(f"分析过程中出错: {e}")
    traceback.print_exc()