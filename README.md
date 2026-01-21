# 2025 Shenzhen Fintechathon

## 项目概述

本项目旨在通过社交网络分析和投资数据挖掘，深入研究政治投资行为和网络影响力传播。我们结合了图算法、机器学习、因果推断等多种技术，构建了一个完整的社交网络投资分析框架，为政治投资决策提供数据支持。

## 项目结构

```
2025-FinTechathon-Data-Analysis/
├── _0_get_X.py               # 获取压缩特征矩阵
├── _0_Party_Visualization.py # 政党信息可视化
├── _1_cnt_local_id.py        # 统计重复节点
├── _2_update_data.py         # 更新节点ID
├── _3_update_edge.py         # 更新边数据
├── _4_get_adj.py             # 生成邻接矩阵
├── _5_data_generate.py       # 参数估计
├── _6_calculate.py           # 计算边概率矩阵
├── _7_get_neighbor.py        # 计算最短路径
├── _8_get_rounds.py          # 日期映射到轮次
├── _9_Party_Evaluation.py    # 政党评估(AHP)
├── _9Pre_Party_Investigation.py # 政党调查(AI)
├── _10_Simulation.py         # 扩散模拟
├── _11_Calculate_Authority.py # 计算权威性
├── _12_Opinion_Leaders_Investment.py # 意见领袖分析
└── README.md                 # 项目说明文档
```

## 技术栈

* **Python 3.x**：项目主要开发语言
* **数据处理**：Pandas, NumPy
* **可视化**：Matplotlib
* **图算法**：NetworkX
* **机器学习**：Scikit-learn (标准化)
* **AI分析**：OpenAI API (Deepseek)
* **决策分析**：AHP (层次分析法)
* **因果推断**：Rubin潜在结果框架
* **社交网络模型**：PoRe-LSM模型

## 项目流程

### 1. 数据预处理阶段

1. **特征提取与压缩**：从多个ego-network中提取节点特征并压缩，生成压缩特征矩阵
2. **重复节点处理**：识别并统计出现多次的local_node_id
3. **节点ID更新**：为重复节点分配新的节点ID，确保唯一性
4. **边数据更新**：扩展图结构，为新节点复制相关边
5. **邻接矩阵生成**：从边文件生成邻接矩阵

### 2. 模型构建和参数估计阶段

1. **数据格式转换**：构建估计器可用的数据格式
2. **参数估计**：完成beta_hat参数估计，生成结果文件
3. **边概率计算**：计算网络中边存在的概率矩阵
4. **最短路径计算**：使用BFS计算所有点对之间的最短路径
5. **时间轮次映射**：将交易日期映射到26个轮次

### 3. 数据分析和可视化阶段

1. **政党可视化**：投资金额时间分布、政党支持者占比
2. **政党评估**：基于AHP方法进行多维度政党评估
3. **AI辅助调查**：使用Deepseek API进行政党相关分析

### 4. 模拟和预测阶段

1. **投资扩散模拟**：模拟投资在社交网络中的扩散过程
2. **权威性计算**：基于因果推断框架计算节点权威性
3. **意见领袖分析**：分析意见领袖的投资行为和影响力

## 功能模块详细说明

### 数据处理模块

* **_0_get_X.py**：从多个ego-network中提取节点特征，进行特征压缩，生成压缩特征矩阵
* **_1_cnt_local_id.py**：统计CSV文件中local_node_id的出现次数，识别重复节点
* **_2_update_data.py**：更新节点ID，为重复节点分配新ID，确保唯一性
* **_3_update_edge.py**：扩展图结构，为新节点复制相关边，更新边数据
* **_4_get_adj.py**：从边文件生成邻接矩阵
* **_7_get_neighbor.py**：使用BFS计算所有点对之间的最短路径
* **_8_get_rounds.py**：将交易日期映射到26个轮次

### 模型构建模块

* **_5_data_generate.py**：构建Pop类实例，完成beta_hat参数估计，生成结果文件
* **_6_calculate.py**：计算边概率矩阵，运行完整的概率分析pipeline

### 政党分析模块

* **_0_Party_Visualization.py**：政党信息可视化，包括投资金额时间分布和支持者占比
* **_9_Party_Evaluation.py**：基于AHP方法进行多维度政党评估
* **_9Pre_Party_Investigation.py**：使用Deepseek API进行政党相关调查和分析

### 社交网络分析模块

* **_10_Simulation.py**：投资扩散模拟，基于社交网络传播模型
* **_11_Calculate_Authority.py**：基于Rubin潜在结果框架，计算节点权威性
* **_12_Opinion_Leaders_Investment.py**：意见领袖投资分析，包括投资概况、党派分布、时间趋势等

## 使用说明

### 运行顺序建议

1. **数据预处理阶段**：
   ```
   _0_get_X.py → _1_cnt_local_id.py → _2_update_data.py → _3_update_edge.py → _4_get_adj.py
   ```

2. **模型构建阶段**：
   ```
   _5_data_generate.py → _6_calculate.py → _7_get_neighbor.py → _8_get_rounds.py
   ```

3. **政党分析阶段**：
   ```
   _0_Party_Visualization.py → _9_Party_Evaluation.py → _9Pre_Party_Investigation.py
   ```

4. **模拟和预测阶段**：
   ```
   _10_Simulation.py → _11_Calculate_Authority.py → _12_Opinion_Leaders_Investment.py
   ```

### 依赖安装

```bash
pip install pandas numpy matplotlib networkx scikit-learn openai
```

## 结果展示

1. **投资金额和投资者数量时间分布图**：展示不同时间段的投资金额和投资者数量变化
2. **政党支持者数量饼状图**：展示各政党支持者数量占比
3. **社交网络扩散模拟动画**：展示投资在社交网络中的扩散过程
4. **意见领袖投资分析报告**：包括投资概况、党派分布、时间趋势等
5. **政党评估结果**：基于AHP方法的多维度政党评估
6. **权威性计算结果**：基于因果推断的节点权威性分析

## 总结和展望

### 总结

本项目成功构建了一个完整的社交网络投资分析框架，结合了图算法、机器学习、因果推断等多种技术，实现了从数据预处理到模型构建，再到数据分析和模拟预测的全流程。通过对政治投资数据和社交网络的深入分析，我们能够更好地理解投资行为的传播机制和意见领袖的影响力。

### 创新点

1. **多源数据整合**：整合了社交网络数据和投资数据，实现了跨领域数据的融合分析
2. **先进模型应用**：应用了PoRe-LSM等先进的社交网络模型，提高了分析的准确性
3. **因果推断方法**：采用了Rubin潜在结果框架进行因果推断，确保了分析结果的可靠性
4. **AI辅助分析**：结合了AI技术进行政党相关分析，增强了分析的深度和广度
5. **可视化展示**：丰富的可视化结果，直观展示分析结论

### 展望

1. **模型优化**：进一步优化社交网络模型，提高预测准确性
2. **实时分析**：实现实时数据处理和分析，支持动态决策
3. **多维度扩展**：扩展到更多领域的社交网络分析，如金融、医疗等
4. **交互界面**：开发用户友好的交互界面，方便非技术人员使用
5. **深度学习**：引入深度学习技术，提高特征提取和预测能力


## 联系方式

- 项目邮箱：yuxianglin@stu.xjtu.edu.cn
- GitHub仓库：https://github.com/Telvancing-star/2025-FinTechathon-Data-Analysis

---
