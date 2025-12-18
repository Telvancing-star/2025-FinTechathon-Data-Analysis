import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import re

# 设置中文字体，确保中文正常显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 读取Excel文件
try:
    df_sheet1 = pd.read_excel('merged_three_keys.xlsx', sheet_name='Sheet1')
    df_party_stats = pd.read_excel('merged_three_keys.xlsx', sheet_name='Sheet3')
    print("文件读取成功！")
except Exception as e:
    print(f"文件读取失败：{e}")
    exit()

# 2. 统计投资金额的时间分布（按年月）并绘制图表
try:
    # 确保金额列是数值类型
    df_sheet1['Denominations'] = pd.to_numeric(df_sheet1['Denominations'], errors='coerce')
    
    # 检查日期列名称（可能是'Journal Date'或其他）
    date_column = None
    for col in ['Journal Date', 'Date', '日期', '交易日期']:
        if col in df_sheet1.columns:
            date_column = col
            break
    
    if date_column is None:
        print("警告：未找到日期列，尝试使用第一列作为日期")
        date_column = df_sheet1.columns[0]
    
    # 将日期列转换为datetime类型
    df_sheet1[date_column] = pd.to_datetime(df_sheet1[date_column], errors='coerce')
    
    # 移除日期为空或金额为空的记录
    df_sheet1 = df_sheet1.dropna(subset=[date_column, 'Denominations'])
    
    # 提取年月信息（格式：YYYY-MM）
    df_sheet1['年月'] = df_sheet1[date_column].dt.to_period('M')
    
    # 检查投资者ID列名称
    investor_column = None
    for col in ['Name of the Purchaser', 'Node ID', '投资者ID', '节点ID', 'Investor ID']:
        if col in df_sheet1.columns:
            investor_column = col
            break
    
    if investor_column is None:
        print("警告：未找到投资者ID列，将无法统计投资者数量")
        investor_column = None
    
    # 按年月分组并计算投资金额总和和投资者数量
    if investor_column:
        monthly_stats = df_sheet1.groupby('年月').agg({
            'Denominations': 'sum',
            investor_column: 'nunique'  # 统计唯一投资者数量
        }).sort_index()
        monthly_stats.columns = ['投资金额', '投资者数量']
    else:
        monthly_stats = df_sheet1.groupby('年月')['Denominations'].sum().to_frame('投资金额').sort_index()
        monthly_stats['投资者数量'] = 0
    
    # 转换为字符串格式用于显示
    monthly_stats.index = monthly_stats.index.astype(str)
    
    print(f"时间范围: {monthly_stats.index.min()} 至 {monthly_stats.index.max()}")
    print(f"总月份数: {len(monthly_stats)}")
    
    # 创建双y轴折线图
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # 左y轴：投资金额
    color1 = 'tab:blue'
    ax1.set_xlabel('年月', fontsize=28, labelpad=10)
    ax1.set_ylabel('投资金额', fontsize=28, color=color1, labelpad=10)
    line1 = ax1.plot(range(len(monthly_stats)), monthly_stats['投资金额'].values, 
                     color=color1, marker='o', linewidth=4, markersize=12, label='投资金额')
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=20)
    ax1.tick_params(axis='x', rotation=45, labelsize=20)
    
    # 格式化左y轴标签
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K' if x >= 1e3 else f'{x:.0f}'
    ))
    
    # 右y轴：投资者数量
    if investor_column and monthly_stats['投资者数量'].sum() > 0:
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('投资者数量', fontsize=28, color=color2, labelpad=10)
        line2 = ax2.plot(range(len(monthly_stats)), monthly_stats['投资者数量'].values, 
                         color=color2, marker='s', linewidth=4, markersize=12, 
                         linestyle='--', label='投资者数量')
        ax2.tick_params(axis='y', labelcolor=color2, labelsize=20)
    
    # 设置x轴标签
    ax1.set_xticks(range(len(monthly_stats)))
    ax1.set_xticklabels(monthly_stats.index, rotation=45, ha='right', fontsize=20)
    
    # 添加标题
    plt.title('投资金额和投资者数量时间分布（按年月）', fontsize=36, pad=20)
    
    # 添加图例
    lines = line1 + (line2 if investor_column and monthly_stats['投资者数量'].sum() > 0 else [])
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=24)
    
    # 添加网格
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    plt.savefig('投资金额和投资者数量时间分布图.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印统计信息
    print(f"\n投资金额统计:")
    print(f"  总金额: {monthly_stats['投资金额'].sum():,.0f}")
    print(f"  平均每月: {monthly_stats['投资金额'].mean():,.0f}")
    print(f"  最高月份: {monthly_stats['投资金额'].idxmax()} ({monthly_stats['投资金额'].max():,.0f})")
    print(f"  最低月份: {monthly_stats['投资金额'].idxmin()} ({monthly_stats['投资金额'].min():,.0f})")
    
    if investor_column and monthly_stats['投资者数量'].sum() > 0:
        # 计算总唯一投资者数（所有月份的去重）
        total_unique_investors = df_sheet1[investor_column].nunique()
        print(f"\n投资者数量统计:")
        print(f"  总唯一投资者数（所有月份去重）: {total_unique_investors:,.0f}")
        print(f"  平均每月投资者数: {monthly_stats['投资者数量'].mean():.1f}")
        print(f"  最高月份: {monthly_stats['投资者数量'].idxmax()} ({monthly_stats['投资者数量'].max():.0f}人)")
        print(f"  最低月份: {monthly_stats['投资者数量'].idxmin()} ({monthly_stats['投资者数量'].min():.0f}人)")
    
except Exception as e:
    print(f"时间分布图绘制失败：{e}")
    import traceback
    traceback.print_exc()

# 3. 读取"Sheet3"工作表的支持者数量并绘制饼状图
def get_party_abbreviation(party_name):
    """
    将政党名称转换为首字母缩写
    例如: "BHARATIYA JANATA PARTY" -> "BJP"
    """
    if pd.isna(party_name) or party_name == '其他':
        return str(party_name)
    
    # 去除标点符号，只保留字母和空格
    cleaned = re.sub(r'[^\w\s]', ' ', str(party_name))
    
    # 按空格分割成单词
    words = cleaned.split()
    
    # 取每个单词的首字母（大写）
    abbreviation = ''.join([word[0].upper() for word in words if len(word) > 0])
    
    return abbreviation

try:
    # 清理数据：去除空值并确保支持者数量为数值
    df_party_stats = df_party_stats.dropna(subset=['Name of the Political Party', '支持者数量'])
    df_party_stats['支持者数量'] = pd.to_numeric(df_party_stats['支持者数量'], errors='coerce')
    df_party_stats = df_party_stats.dropna(subset=['支持者数量'])
    
    # 筛选出支持者数量不为零的政党
    df_party_stats = df_party_stats[df_party_stats['支持者数量'] > 0]
    
    # 绘制饼状图
    plt.figure(figsize=(10, 10))
    
    # 处理可能的大量政党，合并小比例政党为"其他"
    total = df_party_stats['支持者数量'].sum()
    threshold = total * 0.03  # 3%以下的合并为其他
    small_parties = df_party_stats[df_party_stats['支持者数量'] < threshold]
    large_parties = df_party_stats[df_party_stats['支持者数量'] >= threshold]
    
    if not small_parties.empty:
        other = pd.DataFrame([['其他', small_parties['支持者数量'].sum()]],
                            columns=['Name of the Political Party', '支持者数量'])
        df_pie = pd.concat([large_parties, other], ignore_index=True)
    else:
        df_pie = large_parties
    
    # 将政党名称转换为首字母缩写
    df_pie['缩写'] = df_pie['Name of the Political Party'].apply(get_party_abbreviation)
    
    # 绘制饼图（使用缩写作为标签）
    wedges, texts, autotexts = plt.pie(
        df_pie['支持者数量'],
        labels=df_pie['缩写'],
        autopct='%1.1f%%',
        startangle=140,
        colors=plt.cm.Pastel1(np.linspace(0, 1, len(df_pie))),
        wedgeprops=dict(width=0.3, edgecolor='w')  # 环形饼图效果
    )
    
    # 美化文本
    plt.setp(texts, size=24)
    plt.setp(autotexts, size=20, color='black', weight='bold')
    
    plt.title('各政党支持者数量占比', fontsize=36, pad=20)
    plt.axis('equal')  # 保证饼图是正圆形
    plt.tight_layout()
    plt.savefig('政党支持者数量饼状图.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印政党名称和缩写的对应关系
    print("\n政党名称缩写对照表:")
    for _, row in df_pie.iterrows():
        print(f"  {row['缩写']}: {row['Name of the Political Party']}")
    
except Exception as e:
    print(f"饼状图绘制失败：{e}")
    import traceback
    traceback.print_exc()