#!/usr/bin/env python
# coding: utf-8

# In[7]:

import numpy as np
from scipy.stats import multivariate_normal
from scipy.sparse import csr_matrix, vstack
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体和数学符号
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 添加备用字体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'  # 使用STIX数学字体


class Pop:
    def __init__(self, N, beta, delta, C_min, C_max, seed=0, external_X=None, adjacency_csv_file=None):
        """
        直接要求传入特征矩阵X
        """
        self.N = N
        self.C_min = C_min
        self.C_max = C_max
        self.beta = np.array(beta).reshape(-1)
        self.delta = delta
        self.p = len(beta)
        self.seed = seed

        # 必须传入特征矩阵X
        if external_X is None:
            raise ValueError("必须提供特征矩阵X")
        if external_X.shape != (N, self.p):
            raise ValueError(f"特征矩阵X的维度应为({N}, {self.p})，但得到{external_X.shape}")
        self.X = external_X

        self._generate_parameter_alpha()
        self._generate_popularity()
        self._generate_adjacency_matrix(adjacency_csv_file, self.N)
        self._gen_in_degrees()

    def _generate_popularity(self):
        self.gamma = np.exp(self.X @ self.beta + self.alpha)
        return

    def _generate_alpha_MLE(self):
        """
        使用最大似然估计方法计算alpha参数
        基于真实网络数据和特征矩阵
        """
        # 初始alpha估计
        current_alpha = 0.0
        max_iter = 20
        tolerance = 1e-6
        learning_rate = 0.1

        print("开始alpha的最大似然估计...")

        for iteration in range(max_iter):
            # 计算当前alpha下的流行度
            gamma = np.exp(self.X @ self.beta + current_alpha)

            # 计算梯度
            grad = 0.0
            total_pairs = 0

            # 使用随机采样来加速计算（对于大网络）
            sample_size = min(10000, self.N * (self.N - 1))
            sampled_pairs = 0

            while sampled_pairs < sample_size:
                # 随机选择一对节点
                i, j = np.random.randint(0, self.N, 2)
                if i == j:
                    continue

                # 计算连接概率（基于模型的简化版本）
                # 注意：这里使用了模型的概率形式，需要与你的具体模型匹配
                Z_diff_sq = (np.random.randn() - np.random.randn()) ** 2  # 简化：随机潜变量差异
                prob_ij = np.exp(-Z_diff_sq / (2 * gamma[i] * gamma[j]))

                # 观测到的连接
                observed = self.A[i, j] if hasattr(self.A, 'shape') else self.A[i, j]

                # 梯度贡献
                grad += (observed - prob_ij) * prob_ij
                sampled_pairs += 1
                total_pairs += 1

            # 平均梯度
            if total_pairs > 0:
                avg_grad = grad / total_pairs
            else:
                avg_grad = 0

            # 更新alpha
            alpha_update = learning_rate * avg_grad
            current_alpha += alpha_update

            print(
                f"迭代 {iteration + 1}: alpha = {current_alpha:.6f}, 梯度 = {avg_grad:.6f}, 更新 = {alpha_update:.6f}")

            # 检查收敛
            if abs(alpha_update) < tolerance:
                print(f"Alpha估计在迭代 {iteration + 1} 收敛")
                break

        # 根据估计的alpha计算C_alpha
        C_alpha = np.exp(current_alpha + (1 - self.delta) * np.log(self.N))

        # 确保C_alpha在合理范围内
        C_alpha = np.clip(C_alpha, self.C_min, self.C_max)

        print(f"最终估计: alpha = {current_alpha:.6f}, C_alpha = {C_alpha:.2f}")
        return current_alpha, C_alpha

    def _generate_alpha(self):
        """
        基于网络统计的alpha估计方法
        """
        print("使用基于网络统计的alpha估计...")

        # 计算网络的基本统计量
        if hasattr(self, 'A') and self.A is not None:
            # 使用真实网络数据
            total_edges = np.sum(self.A)
            network_density = total_edges / (self.N * (self.N - 1))
            avg_degree = total_edges / self.N
        else:
            # 如果没有网络数据，使用期望值
            avg_degree = (self.C_min + self.C_max) / 2
            network_density = avg_degree / (self.N - 1)

        print(f"网络统计: 平均度 = {avg_degree:.2f}, 密度 = {network_density:.6f}")

        # 基于模型关系估计C_alpha
        # 模型: E[degree] ≈ C_alpha * C_beta * N^(delta-1)
        # expected_degree = self.C_alpha * self.C_beta * (self.N ** (self.delta - 1))

        # 使用网格搜索找到最优的C_alpha
        best_C_alpha = self.C_min
        best_error = float('inf')

        C_alpha_candidates = np.linspace(self.C_min, self.C_max, 100)

        for candidate in C_alpha_candidates:
            # 计算对应的alpha
            candidate_alpha = np.log(candidate) - (1 - self.delta) * np.log(self.N)

            # 计算期望度
            expected_deg = candidate * self.C_beta * (self.N ** (self.delta - 1))

            # 计算误差（与观测度或目标度的差异）
            error = abs(expected_deg - avg_degree)

            if error < best_error:
                best_error = error
                best_C_alpha = candidate

        # 计算对应的alpha
        best_alpha = np.log(best_C_alpha) - (1 - self.delta) * np.log(self.N)

        print(f"估计结果: C_alpha = {best_C_alpha:.2f}, alpha = {best_alpha:.6f}")
        print(f"期望平均度: {best_C_alpha * self.C_beta * (self.N ** (self.delta - 1)):.2f}")

        return best_alpha, best_C_alpha

    def _generate_parameter_alpha(self):
        """
        使用外部输入的特征矩阵计算C_beta和alpha
        """
        # 使用外部特征矩阵计算C_beta
        if hasattr(self, 'X') and self.X is not None:
            print("使用外部特征矩阵计算C_beta...")
            test_exp_X_beta = np.exp(self.X @ self.beta)
            self.C_beta = np.mean(test_exp_X_beta)
            print(f"C_beta = {self.C_beta:.6f} (基于{self.X.shape[0]}个节点)")

        # 估计alpha
        self.alpha, self.C_alpha = self._generate_alpha()
        return

    def _generate_adjacency_matrix(self, adjacency_csv_file, total_nodes=None):
        """
        从CSV格式的邻接矩阵文件构建稀疏邻接矩阵

        参数:
        adjacency_csv_file: 邻接矩阵CSV文件路径
        total_nodes: 总节点数

        返回:
        A: CSR格式的稀疏邻接矩阵
        """
        try:
            print(f"Reading adjacency matrix from {adjacency_csv_file}...")

            # 跳过第一列（索引列）
            adj_df = pd.read_csv(adjacency_csv_file, index_col=0)
            adj_matrix = adj_df.values

            print(f"邻接矩阵形状: {adj_matrix.shape}")
            print(f"矩阵数据类型: {adj_matrix.dtype}")

            # 使用矩阵的实际大小
            if total_nodes is None:
                total_nodes = adj_matrix.shape[0]

            # 转换为稀疏矩阵
            A = csr_matrix(adj_matrix)
            A.setdiag(0)  # 移除自环
            A.eliminate_zeros()

            print(f"最终网络: 节点数={A.shape[0]}, 边数={A.nnz}")
            print(f"网络密度: {A.nnz / (A.shape[0] * (A.shape[0] - 1)):.8f}")

            self.A = A
            return

        except Exception as e:
            print(f"Error building adjacency matrix: {e}")
            raise

        except Exception as e:
            print(f"Error building adjacency matrix: {e}")
            raise

    def _gen_in_degrees(self):
        self.in_degrees = np.array(np.sum(self.A, axis=0))[0]
        return


# In[8]:


class Pop_NR:
    def __init__(self, data):
        self.X = data.X
        self.N, self.p = self.X.shape
        self.A = data.A
        self.d_in = data.in_degrees
        self.col_prod = self.A.T @ self.A

        # 添加维度检查
        print(f"Pop_NR初始化检查:")
        print(f"  X形状: {self.X.shape}")
        print(f"  A形状: {self.A.shape}")
        print(f"  d_in形状: {self.d_in.shape}")
        print(f"  col_prod形状: {self.col_prod.shape}")

        # 确保所有维度一致
        assert self.A.shape[0] == self.A.shape[1] == self.N, f"邻接矩阵A形状{self.A.shape}与节点数{self.N}不匹配"
        assert len(self.d_in) == self.N, f"入度向量d_in长度{len(self.d_in)}与节点数{self.N}不匹配"

    def generate_pi_matrix(self, beta):
        beta = np.array(beta).reshape(-1, 1)
        vec = np.exp(2 * (self.X @ beta)).reshape(-1, 1)

        # 添加维度调试信息
        print(f"generate_pi_matrix调试:")
        print(f"  beta形状: {beta.shape}")
        print(f"  vec形状: {vec.shape}")
        print(f"  A形状: {self.A.shape}")

        len_slice = min(1000, self.N)  # 确保切片不超过矩阵大小

        # 检查切片范围
        end_slice = min(len_slice, self.N)
        print(f"  第一个切片: 0:{end_slice}")

        # 第一个切片
        slice_A = self.A[0:end_slice, :]
        slice_vec = vec[0:end_slice]

        # 计算概率项
        prob_denom = slice_vec + 2 * vec.reshape(1, -1)
        prob_matrix = np.sqrt(vec.reshape(1, -1) / prob_denom)

        # 检查概率矩阵维度
        print(f"  prob_matrix形状: {prob_matrix.shape}")
        print(f"  slice_A形状: {slice_A.shape}")

        pi = slice_A.multiply(prob_matrix)
        log_pi = slice_A.multiply(np.log(prob_matrix))
        log_pi_minus = slice_A.multiply(np.log(1 - prob_matrix))

        # 转换为稠密矩阵进行计算
        pi_dense = pi.toarray()
        const_grad1 = slice_A.multiply((1 - 2 * pi_dense ** 2) / (1 - pi_dense))
        const_grad2 = -pi.multiply(const_grad1)

        const_hes1 = slice_A.multiply(
            (4 * pi_dense - 2 * pi_dense ** 2 - 1) / (1 - pi_dense) ** 2)
        const_hes2 = slice_A.multiply(
            (4 * pi_dense ** 3 - 6 * pi_dense ** 2 + 1) / (1 - pi_dense) ** 2)

        # 处理剩余切片
        total_slices = int(np.ceil(self.N / len_slice))
        print(f"  总切片数: {total_slices}")

        for r in range(1, total_slices):
            start_idx = int(len_slice * r)
            end_idx = min(int(len_slice * (r + 1)), self.N)

            if start_idx >= self.N:
                break

            print(f"  处理切片 {r}: {start_idx}:{end_idx}")

            slice_A = self.A[start_idx:end_idx, :]
            slice_vec = vec[start_idx:end_idx]

            # 计算当前切片的概率矩阵
            prob_denom = slice_vec + 2 * vec.reshape(1, -1)
            prob_matrix = np.sqrt(vec.reshape(1, -1) / prob_denom)

            row_pi = slice_A.multiply(prob_matrix)
            row_log_pi = slice_A.multiply(np.log(prob_matrix))
            row_log_pi_minus = slice_A.multiply(np.log(1 - prob_matrix))

            # 转换为稠密矩阵进行计算
            row_pi_dense = row_pi.toarray()
            row_const_grad1 = slice_A.multiply(
                (1 - 2 * row_pi_dense ** 2) / (1 - row_pi_dense))
            row_const_grad2 = -row_pi.multiply(row_const_grad1)

            row_const_hes1 = slice_A.multiply(
                (4 * row_pi_dense - 2 * row_pi_dense ** 2 - 1) / (1 - row_pi_dense) ** 2)
            row_const_hes2 = slice_A.multiply(
                (4 * row_pi_dense ** 3 - 6 * row_pi_dense ** 2 + 1) / (1 - row_pi_dense) ** 2)

            # 垂直堆叠
            pi = vstack([pi, row_pi])
            log_pi = vstack([log_pi, row_log_pi])
            log_pi_minus = vstack([log_pi_minus, row_log_pi_minus])
            const_grad1 = vstack([const_grad1, row_const_grad1])
            const_grad2 = vstack([const_grad2, row_const_grad2])
            const_hes1 = vstack([const_hes1, row_const_hes1])
            const_hes2 = vstack([const_hes2, row_const_hes2])

        # 最终维度检查
        print(f"生成矩阵最终形状:")
        print(f"  pi: {pi.shape}, log_pi_minus: {log_pi_minus.shape}")
        print(f"  A: {self.A.shape}, d_in: {self.d_in.shape}")

        return pi, log_pi, log_pi_minus, const_grad1, const_grad2, const_hes1, const_hes2

    def loss_grad_hessian(self, parameter):
        print("开始计算loss_grad_hessian...")
        parameter = np.array(parameter).reshape(-1, 1)
        pi, log_pi, log_pi_minus, const_grad1, const_grad2, const_hes1, const_hes2 = self.generate_pi_matrix(parameter)

        # loss function
        print("计算损失函数...")
        l2 = self.A.multiply(log_pi_minus).multiply(self.d_in.reshape(-1, 1))
        l3 = -self.A.multiply(log_pi_minus)
        l1 = self.col_prod.multiply(self.A).multiply(log_pi) - self.col_prod.multiply(self.A).multiply(log_pi_minus)
        loss = -(np.sum(l1) + np.sum(l2) + np.sum(l3)) / (self.N * (self.N - 1) * (self.N - 2))

        # gradient
        print("计算梯度...")
        weight_g1 = self.col_prod.multiply(self.A).multiply(const_grad1)
        weight_g2 = self.A.multiply(self.d_in.reshape(-1, 1)).multiply(const_grad2)
        weight_g3 = -self.A.T.multiply(self.A.multiply(const_grad2))

        weight_g = -(weight_g1 + weight_g2 + weight_g3)
        g = np.zeros((self.p, 1))

        splits = min(1000, weight_g.nnz)
        if weight_g.nnz > 0:
            zipped_indices = zip(
                *(np.array_split(weight_g.nonzero()[0], splits),
                  np.array_split(weight_g.nonzero()[1], splits)))
            for i_s, j_s in zipped_indices:
                g += np.sum(weight_g[i_s, j_s] * (self.X[j_s,] - self.X[i_s,]), axis=0).reshape(-1, 1)

        # Hessian - 改进的稳健计算
        print("计算Hessian...")
        weight_h1 = self.col_prod.multiply(self.A).multiply(const_hes1)
        weight_h2 = self.A.multiply(self.d_in.reshape(-1, 1)).multiply(const_hes2)
        weight_h3 = -self.A.T.multiply(self.A.multiply(const_hes2))

        weight_h = weight_h1 + weight_h2 + weight_h3
        h = np.zeros((self.p, self.p))

        if weight_h.nnz > 0:
            zipped_indices = zip(
                *(np.array_split(weight_h.nonzero()[0], splits),
                  np.array_split(weight_h.nonzero()[1], splits)))
            for i_s, j_s in zipped_indices:
                # 稳健的Hessian计算
                X_diff = self.X[i_s,] - self.X[j_s,]
                weights = np.array(weight_h[i_s, j_s]).reshape(-1, 1)

                # 避免数值不稳定
                weights = np.clip(weights, -1e10, 1e10)  # 限制权重范围
                weights = np.nan_to_num(weights, nan=0.0, posinf=1e10, neginf=-1e10)

                # 稳健的外积计算
                outer_prod = X_diff.T @ (X_diff * weights)
                outer_prod = np.nan_to_num(outer_prod, nan=0.0, posinf=1e10, neginf=-1e10)

                h += outer_prod

        # 确保Hessian对称
        h = 0.5 * (h + h.T)

        # 添加基础正则化防止奇异
        h_reg = h + 1e-8 * np.eye(self.p)

        # 检查Hessian性质
        try:
            eigvals = np.linalg.eigvalsh(h_reg)
            min_eigval = np.min(eigvals)
            max_eigval = np.max(eigvals)
            cond_number = max_eigval / (min_eigval + 1e-12)

            print(f"Hessian诊断: 最小特征值={min_eigval:.2e}, 最大特征值={max_eigval:.2e}, 条件数={cond_number:.2e}")

            # 如果条件数过大，增强正则化
            if cond_number > 1e12:
                print("Hessian条件数过大，增强正则化")
                reg_strength = max(1e-6, 1e-6 * max_eigval) * np.eye(self.p)
                h_reg = h + reg_strength

        except np.linalg.LinAlgError:
            print("Hessian特征值计算失败，使用强正则化")
            h_reg = h + 1e-4 * np.eye(self.p)

        print(f"计算完成: loss={loss:.6f}, 梯度范数={np.linalg.norm(g):.6f}")
        return loss, g.reshape(-1, 1), h_reg

    def run(self, running_parameter, alpha=0.01, max_iter=200, epsilon=1e-8):
        print(f"开始优化，初始参数形状: {running_parameter.shape}")
        running_parameter = np.array(running_parameter).reshape(-1, 1)
        it = 0
        L_old = None

        # 记录收敛历史
        loss_history = []
        grad_norm_history = []
        param_norm_history = [np.linalg.norm(running_parameter)]
        update_norm_history = []

        while it < max_iter:
            print(f"\n=== 迭代 {it + 1}/{max_iter} ===")
            L, g, h = self.loss_grad_hessian(running_parameter)
            loss_history.append(L)
            grad_norm = np.linalg.norm(g)
            grad_norm_history.append(grad_norm)

            # 详细诊断信息
            print(f"损失: {L:.8f}")
            print(f"梯度范数: {grad_norm:.6f}")
            print(f"参数范数: {param_norm_history[-1]:.6f}")
            print(f"梯度范围: [{g.min():.6f}, {g.max():.6f}]")

            if grad_norm > 80000:
                alpha = 0.8
            elif grad_norm > 10000:
                alpha = 0.5
            elif grad_norm > 1000:
                alpha = 0.1
            else:
                alpha = 0.01

            # 收敛检查
            if L_old is not None:
                rel_change = abs(L_old - L) / (abs(L_old) + 1e-12)
                print(f"相对变化: {rel_change:.2e}")

                if rel_change < epsilon:
                    print(f"在迭代 {it + 1} 收敛!")
                    break

                # 检查损失是否发散
                if L > L_old * 10 and it > 5:  # 损失显著增加
                    print("警告: 损失显著增加，可能发散")
                    alpha *= 0.5  # 减小学习率
                    print(f"调整学习率为: {alpha}")

            L_old = L

            # 稳健的更新计算
            try:
                # 计算Hessian条件数
                cond_number = np.linalg.cond(h)
                print(f"Hessian条件数: {cond_number:.2e}")

                # 自适应正则化
                if cond_number > 1e10:
                    print("使用强正则化")
                    reg_strength = 1e-4 * np.trace(h) / h.shape[0] * np.eye(h.shape[0])
                    h_safe = h + reg_strength
                elif cond_number > 1e6:
                    print("使用中等正则化")
                    reg_strength = 1e-6 * np.trace(h) / h.shape[0] * np.eye(h.shape[0])
                    h_safe = h + reg_strength
                else:
                    h_safe = h

                # 稳健的矩阵求逆
                try:
                    h_inv = np.linalg.inv(h_safe)
                except np.linalg.LinAlgError:
                    print("矩阵求逆失败，使用伪逆")
                    h_inv = np.linalg.pinv(h_safe)

                update = alpha * h_inv @ g

            except Exception as e:
                print(f"Hessian处理失败: {e}，使用梯度下降")
                # 梯度下降作为备选
                update = alpha * 0.01 * g / (grad_norm + 1e-12)

            # 自适应步长控制
            update_norm = np.linalg.norm(update)
            update_norm_history.append(update_norm)
            print(f"更新步长: {update_norm:.6f}")

            # 动态步长调整
            if update_norm > 1.0:
                print(f"步长过大 ({update_norm:.2f})，进行裁剪")
                update = update / update_norm * 0.5
                alpha *= 0.8  # 减小学习率
            elif update_norm < 1e-8 and it > 3:
                print("步长过小，可能已收敛")
                break
            elif update_norm < 1e-10:
                print("步长接近零，停止优化")
                break

            # 应用更新
            running_parameter = running_parameter - update
            param_norm = np.linalg.norm(running_parameter)
            param_norm_history.append(param_norm)

            # 参数边界检查
            if np.any(np.isnan(running_parameter)) or np.any(np.isinf(running_parameter)):
                print("警告: 参数包含NaN或Inf值，恢复上一步参数")
                running_parameter = running_parameter + update  # 恢复
                alpha *= 0.5  # 减小学习率
                print(f"调整学习率为: {alpha}")

            it += 1

        # 绘制收敛历史
        self._plot_detailed_convergence(loss_history, grad_norm_history, param_norm_history, update_norm_history)

        print(f"\n优化完成，共进行 {it} 次迭代")
        print(f"最终损失: {loss_history[-1]:.8f}")
        print(f"最终梯度范数: {grad_norm_history[-1]:.6f}")
        print(f"最终参数范数: {param_norm_history[-1]:.6f}")

        return running_parameter

    def _plot_detailed_convergence(self, loss_history, grad_norm_history, param_norm_history, update_norm_history):
        """绘制详细的收敛历史"""
        try:
            import matplotlib.pyplot as plt

            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['mathtext.fontset'] = 'stix'

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            # 损失函数
            ax1.plot(loss_history, 'b-o', linewidth=2, markersize=4)
            ax1.set_xlabel('迭代次数')
            ax1.set_ylabel('损失函数值')
            ax1.set_title('损失函数收敛历史')
            ax1.grid(True, alpha=0.3)

            # 梯度范数
            ax2.plot(grad_norm_history, 'r-o', linewidth=2, markersize=4)
            ax2.set_xlabel('迭代次数')
            ax2.set_ylabel('梯度范数')
            ax2.set_title('梯度范数收敛历史')
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')

            # 参数范数
            ax3.plot(param_norm_history, 'g-o', linewidth=2, markersize=4)
            ax3.set_xlabel('迭代次数')
            ax3.set_ylabel('参数范数')
            ax3.set_title('参数范数变化历史')
            ax3.grid(True, alpha=0.3)

            # 更新步长
            ax4.plot(update_norm_history, 'm-o', linewidth=2, markersize=4)
            ax4.set_xlabel('迭代次数')
            ax4.set_ylabel('更新步长')
            ax4.set_title('更新步长变化历史')
            ax4.grid(True, alpha=0.3)
            ax4.set_yscale('log')

            plt.tight_layout()
            plt.savefig('pop_nr_detailed_convergence.png', dpi=300, bbox_inches='tight')
            plt.close()

            print("详细收敛历史图已保存为 'pop_nr_detailed_convergence.png'")

        except ImportError:
            print("无法绘制收敛历史 (matplotlib 未安装)")


# In[9]:


import itertools


def compute_M1_block(exp_block, X_block, p=None):
    if p is None:
        p = X_block.shape[1]

    exp_block = exp_block.reshape(-1, 1)
    permutations = list(itertools.permutations([0, 1, 2, 3]))
    fm = np.zeros((p, p))
    w = np.zeros(12)
    mtrcs = np.zeros((12, p, p))

    for perm in permutations:
        exp1 = exp_block[list(perm)]
        X = X_block[list(perm), :]
        vec = exp1 ** 2
        pi = np.sqrt(vec.T / (vec + 2 * vec.T))
        weight_xi = -(1 - 2 * pi ** 2) / (1 - pi)

        v12 = exp1[0] ** 2 + 2 * exp1[1] ** 2
        v21 = exp1[1] ** 2 + 2 * exp1[0] ** 2
        v13 = exp1[0] ** 2 + 2 * exp1[2] ** 2
        v31 = exp1[2] ** 2 + 2 * exp1[0] ** 2
        v14 = exp1[0] ** 2 + 2 * exp1[3] ** 2
        v41 = exp1[3] ** 2 + 2 * exp1[0] ** 2
        v23 = exp1[1] ** 2 + 2 * exp1[2] ** 2
        v32 = exp1[2] ** 2 + 2 * exp1[1] ** 2
        v24 = exp1[1] ** 2 + 2 * exp1[3] ** 2
        v42 = exp1[3] ** 2 + 2 * exp1[1] ** 2

        # 1,1 2,2
        w1122 = 0.5 * exp1[1] * exp1[2] ** 2 * exp1[3] ** 2 * (
                (v23 * v24 - exp1[1] ** 4) ** (-0.5) - (v23 * v24) ** (-0.5))
        w1122 = w1122 * weight_xi[1, 2] * weight_xi[1, 3]
        m1122 = (X[2, :] - X[1, :]).reshape(-1, 1) @ (X[3, :] - X[1, :]).reshape(1, -1)
        m1122 = 0.5 * w1122 * (m1122 + m1122.T)

        # 1,1 3,3
        w1133 = 0.5 * (1 - pi[2, 1] - pi[3, 1]) * exp1[1] ** 3 * exp1[2] * exp1[3] * (v32 * v42 - exp1[1] ** 4) ** (
            -0.5) + 0.5 * pi[2, 1] * pi[3, 1] * exp1[1] ** 2 * exp1[2] * exp1[3] * (
                        2 * exp1[1] ** 2 + exp1[2] ** 2 + exp1[3] ** 2) ** (-0.5)
        w1133 = w1133 * weight_xi[2, 1] * weight_xi[3, 1]
        m1133 = (X[1, :] - X[2, :]).reshape(-1, 1) @ (X[1, :] - X[3, :]).reshape(1, -1)
        m1133 = 0.5 * w1133 * (m1133 + m1133.T)

        # 2,2 3,3
        w2233 = 0.5 * exp1[0] ** 2 * exp1[1] ** 3 * ((v12 ** 2 - exp1[1] ** 4) ** (-0.5) - (v12) ** (-1))
        w2233 = w2233 * weight_xi[0, 1] * weight_xi[0, 1]
        m2233 = (X[1, :] - X[0, :]).reshape(-1, 1) @ (X[1, :] - X[0, :]).reshape(1, -1)
        m2233 = 0.5 * w2233 * (m2233 + m2233.T)

        # 1,1 2,3
        w1123 = 0.5 * (1 - pi[3, 1]) * exp1[1] ** 2 * exp1[2] ** 2 * exp1[3] * (
                (v23 * v42 - exp1[1] ** 4) ** (-0.5) - (v23 * v42) ** (-0.5))
        w1123 = w1123 * weight_xi[1, 2] * weight_xi[3, 1]
        m1123 = (X[2, :] - X[1, :]).reshape(-1, 1) @ (X[1, :] - X[3, :]).reshape(1, -1)
        m1123 = 0.5 * w1123 * (m1123 + m1123.T)

        # 2,2 1,3
        w2213 = 0.5 * exp1[0] ** 2 * exp1[1] ** 2 * exp1[2] ** 2
        lv2213 = (exp1[0] ** 2 + exp1[1] ** 2) * v13 * v12
        w2213 = w2213 * (lv2213 ** (-0.5) - (lv2213 - exp1[0] ** 4 * v12) ** (-0.5) - (lv2213 - exp1[1] ** 4 * v13) ** (
            -0.5) + (lv2213 - exp1[1] ** 4 * v13 - exp1[0] ** 4 * v12) ** (-0.5))
        w2213 = w2213 * weight_xi[0, 2] * weight_xi[0, 1]
        m2213 = (X[2, :] - X[0, :]).reshape(-1, 1) @ (X[1, :] - X[0, :]).reshape(1, -1)
        m2213 = 0.5 * w2213 * (m2213 + m2213.T)

        # 3,3 1,2
        w3312 = 0.5 * (1 - pi[2, 0]) * exp1[0] ** 3 * exp1[1] * exp1[2] * (
                (v21 * v31 - exp1[0] ** 4) ** (-0.5) - (v21 * v31) ** (-0.5))
        w3312 = w3312 * weight_xi[2, 0] * weight_xi[1, 0]
        m3312 = (X[0, :] - X[2, :]).reshape(-1, 1) @ (X[0, :] - X[1, :]).reshape(1, -1)
        m3312 = 0.5 * w3312 * (m3312 + m3312.T)

        # 1,2 2,1
        w1221 = 0.5 * exp1[0] * exp1[1] * exp1[2] ** 2 * exp1[3] ** 2
        lv1221 = (exp1[0] ** 2 + exp1[1] ** 2) * v23 * v14
        w1221 = w1221 * (lv1221 ** (-0.5) - (lv1221 - exp1[1] ** 4 * v14) ** (-0.5) - (lv1221 - exp1[0] ** 4 * v23) ** (
            -0.5) + (lv1221 - exp1[1] ** 4 * v14 - exp1[0] ** 4 * v23) ** (-0.5))
        w1221 = w1221 * weight_xi[1, 2] * weight_xi[0, 3]
        m1221 = (X[2, :] - X[1, :]).reshape(-1, 1) @ (X[3, :] - X[0, :]).reshape(1, -1)
        m1221 = 0.5 * w1221 * (m1221 + m1221.T)

        # 1,3 3,1
        w1331 = 0.5 * exp1[0] ** 2 * exp1[1] ** 2 * exp1[2] * exp1[3]
        lv1331 = (exp1[0] ** 2 + exp1[1] ** 2 + exp1[2] ** 2 + exp1[3] ** 2) * v41 * v32
        w1331 = w1331 * (lv1331 ** (-0.5) - (lv1331 - (exp1[1] ** 2 + exp1[2] ** 2) ** 2 * v41) ** (-0.5) - (
                lv1331 - (exp1[0] ** 2 + exp1[3] ** 2) ** 2 * v32) ** (-0.5) + (
                                 lv1331 - (exp1[0] ** 2 + exp1[3] ** 2) ** 2 * v32 - (
                                 exp1[1] ** 2 + exp1[2] ** 2) ** 2 * v41) ** (-0.5))
        w1331 = w1331 * weight_xi[2, 1] * weight_xi[3, 0]
        m1331 = (X[1, :] - X[2, :]).reshape(-1, 1) @ (X[0, :] - X[3, :]).reshape(1, -1)
        m1331 = 0.5 * w1331 * (m1331 + m1331.T)

        # 2,3 3,2
        w2332 = 0.5 * exp1[0] ** 3 * exp1[1] ** 3
        lv2332 = (exp1[0] ** 2 + exp1[1] ** 2) * v21 * v12
        w2332 = w2332 * (lv2332 ** (-0.5) - (lv2332 - exp1[1] ** 4 * v21) ** (-0.5) - (lv2332 - exp1[0] ** 4 * v12) ** (
            -0.5) + (lv2332 - exp1[1] ** 4 * v21 - exp1[0] ** 4 * v12) ** (-0.5))
        w2332 = w2332 * weight_xi[0, 1] * weight_xi[1, 0]
        m2332 = (X[1, :] - X[0, :]).reshape(-1, 1) @ (X[0, :] - X[1, :]).reshape(1, -1)
        m2332 = 0.5 * w2332 * (m2332 + m2332.T)

        # 1,2 2,3
        w1223 = 0.5 * exp1[0] * exp1[1] ** 2 * exp1[2] ** 2 * (
                (v12 * v23 - exp1[1] ** 4) ** (-0.5) - (v12 * v23) ** (-0.5))
        w1223 = w1223 * weight_xi[1, 2] * weight_xi[0, 1]
        m1223 = (X[2, :] - X[1, :]).reshape(-1, 1) @ (X[1, :] - X[0, :]).reshape(1, -1)
        m1223 = 0.5 * w1223 * (m1223 + m1223.T)

        # 2,1 1,3
        w2113 = 0.5 * exp1[0] * exp1[1] ** 2 * exp1[2] ** 2 * exp1[3]
        lv2113 = (exp1[0] ** 2 + exp1[1] ** 2 + exp1[3] ** 2) * v13 * v42
        w2113 = w2113 * (lv2113 ** (-0.5) - (lv2113 - (exp1[1] ** 2 + exp1[3] ** 2) ** 2 * v13) ** (-0.5) - (
                lv2113 - exp1[0] ** 4 * v42) ** (-0.5) + (
                                 lv2113 - (exp1[1] ** 2 + exp1[3] ** 2) ** 2 * v13 - exp1[0] ** 4 * v42) ** (-0.5))
        w2113 = w2113 * weight_xi[0, 2] * weight_xi[3, 1]
        m2113 = (X[2, :] - X[0, :]).reshape(-1, 1) @ (X[1, :] - X[3, :]).reshape(1, -1)
        m2113 = 0.5 * w2113 * (m2113 + m2113.T)

        # 2,3 3,1
        w2331 = 0.5 * exp1[0] ** 3 * exp1[1] ** 2 * exp1[3]
        lv2331 = (exp1[0] ** 2 + exp1[1] ** 2 + exp1[3] ** 2) * v41 * v12
        w2331 = w2331 * (lv2331 ** (-0.5) - (lv2331 - (exp1[0] ** 2 + exp1[3] ** 2) ** 2 * v12) ** (-0.5) - (
                lv2331 - exp1[1] ** 4 * v41) ** (-0.5) + (
                                 lv2331 - (exp1[0] ** 2 + exp1[3] ** 2) ** 2 * v12 - exp1[1] ** 4 * v41) ** (-0.5))
        w2331 = w2331 * weight_xi[0, 1] * weight_xi[3, 0]
        m2331 = (X[1, :] - X[0, :]).reshape(-1, 1) @ (X[0, :] - X[3, :]).reshape(1, -1)
        m2331 = 0.5 * w2331 * (m2331 + m2331.T)

        current = m1122 + m1133 + m2233 + 2 * m1123 + 2 * m2213 + 2 * m3312 + m1221 + m1331 + m2332 + 2 * m1223 + 2 * m2113 + 2 * m2331
        w = w + np.array(
            [w1122, w1133, w2233, 2 * w1123, 2 * w2213, 2 * w3312, w1221, w1331, w2332, 2 * w1223, 2 * w2113,
             2 * w2331]).reshape(-1)
        mtrcs = mtrcs + np.array([m1122, m1133, m2233, m1123, m2213, m3312, m1221, m1331, m2332, m1223, m2113, m2331])
        fm = fm + current

    return fm / 24, w / 24, mtrcs / 24


# In[10]:


def compute_M0_block(exp_block, X_block, p=None):
    if p is None:
        p = X_block.shape[1]

    exp_block = exp_block.reshape(-1, 1)
    permutations = list(itertools.permutations([0, 1, 2]))
    fm = np.zeros((p, p))
    w = np.zeros(6)
    mtrcs = np.zeros((6, p, p))

    for perm in permutations:
        exp1 = exp_block[list(perm)].reshape(-1, 1)
        X = X_block[list(perm), :]
        vec = exp1 ** 2
        pi = np.sqrt(vec.T / (vec + 2 * vec.T))
        weight_xi = (1 - 2 * pi ** 2) / (1 - pi)

        v12 = exp1[0] ** 2 + 2 * exp1[1] ** 2
        v21 = exp1[1] ** 2 + 2 * exp1[0] ** 2
        v23 = exp1[1] ** 2 + 2 * exp1[2] ** 2
        v31 = exp1[2] ** 2 + 2 * exp1[0] ** 2

        # 123 123
        w123123 = (1 - 2 * pi[1, 2]) * exp1[1] * exp1[2] ** 2 / np.sqrt(3 * v23) + pi[1, 2] ** 2 * exp1[1] * exp1[
            2] / np.sqrt(3)
        w123123 = w123123 * weight_xi[1, 2] * weight_xi[1, 2]
        m123123 = (X[2, :] - X[1, :]).reshape(-1, 1) @ (X[2, :] - X[1, :]).reshape(1, -1)
        m123123 = 0.5 * w123123 * (m123123 + m123123.T)

        # 123 132
        w123132 = (1 - pi[1, 2]) * (1 - pi[2, 1]) * exp1[1] ** 2 * exp1[2] ** 2 / np.sqrt(
            3 * (exp1[1] ** 2 + exp1[2] ** 2) ** 2 + 3 * exp1[1] ** 2 * exp1[2] ** 2)
        w123132 = w123132 * weight_xi[1, 2] * weight_xi[2, 1]
        m123132 = (X[2, :] - X[1, :]).reshape(-1, 1) @ (X[1, :] - X[2, :]).reshape(1, -1)
        m123132 = 0.5 * w123132 * (m123132 + m123132.T)

        # 123 213
        w123213 = (1 - pi[1, 2]) * (1 - pi[0, 2]) * exp1[0] * exp1[1] * exp1[2] ** 2 / np.sqrt(
            3 * (2 * exp1[1] ** 2 * exp1[2] ** 2 + 2 * exp1[0] ** 2 * exp1[2] ** 2 + exp1[0] ** 2 * exp1[1] ** 2))
        w123213 = w123213 * weight_xi[1, 2] * weight_xi[0, 2]
        m123213 = (X[2, :] - X[1, :]).reshape(-1, 1) @ (X[2, :] - X[0, :]).reshape(1, -1)
        m123213 = 0.5 * w123213 * (m123213 + m123213.T)

        # 123 231
        w123231 = exp1[0] ** 2 * exp1[1] * exp1[2] ** 2 / np.sqrt(3)
        l1v123231 = ((exp1[0] ** 2 + exp1[1] ** 2 + exp1[2] ** 2) * v23 * v31) ** (-0.5)
        l2v123231 = (((exp1[0] ** 2 + exp1[2] ** 2) * (exp1[1] ** 2 + exp1[2] ** 2) + exp1[0] ** 2 * exp1[
            2] ** 2) * v31) ** (-0.5)
        l3v123231 = (((exp1[0] ** 2 + exp1[2] ** 2) * (exp1[0] ** 2 + exp1[1] ** 2) + exp1[0] ** 2 * exp1[
            1] ** 2) * v23) ** (-0.5)
        l4v123231 = (2 * exp1[0] ** 4 * exp1[2] ** 2 + 3 * exp1[0] ** 2 * exp1[1] ** 2 * exp1[2] ** 2 + exp1[0] ** 2 *
                     exp1[2] ** 4 + exp1[1] ** 2 * exp1[2] ** 4 + exp1[0] ** 4 * exp1[1] ** 2) ** (-0.5)
        w123231 = w123231 * (l1v123231 - l2v123231 - l3v123231 + l4v123231)
        w123231 = w123231 * weight_xi[1, 2] * weight_xi[2, 0]
        m123231 = (X[2, :] - X[1, :]).reshape(-1, 1) @ (X[0, :] - X[2, :]).reshape(1, -1)
        m123231 = 0.5 * w123231 * (m123231 + m123231.T)

        # 123 312
        w123312 = exp1[0] * exp1[1] ** 2 * exp1[2] ** 2 / np.sqrt(3)
        l1v123312 = ((exp1[2] ** 2 + exp1[0] ** 2 + exp1[1] ** 2) * v12 * v23) ** (-0.5)
        l2v123312 = (((exp1[2] ** 2 + exp1[1] ** 2) * (exp1[0] ** 2 + exp1[1] ** 2) + exp1[2] ** 2 * exp1[
            1] ** 2) * v23) ** (-0.5)
        l3v123312 = (((exp1[2] ** 2 + exp1[1] ** 2) * (exp1[2] ** 2 + exp1[0] ** 2) + exp1[2] ** 2 * exp1[
            0] ** 2) * v12) ** (-0.5)
        l4v123312 = (2 * exp1[2] ** 4 * exp1[1] ** 2 + 3 * exp1[2] ** 2 * exp1[0] ** 2 * exp1[1] ** 2 + exp1[2] ** 2 *
                     exp1[1] ** 4 + exp1[0] ** 2 * exp1[1] ** 4 + exp1[2] ** 4 * exp1[0] ** 2) ** (-0.5)
        w123312 = w123312 * (l1v123312 - l2v123312 - l3v123312 + l4v123312)
        w123312 = w123312 * weight_xi[0, 1] * weight_xi[1, 2]
        m123312 = (X[1, :] - X[0, :]).reshape(-1, 1) @ (X[2, :] - X[1, :]).reshape(1, -1)
        m123312 = 0.5 * w123312 * (m123312 + m123312.T)

        # 123 321
        w123321 = exp1[0] ** 2 * exp1[1] ** 2 * exp1[2] ** 2 / np.sqrt(3)
        vv12 = exp1[0] ** 2 + exp1[1] ** 2
        vv23 = exp1[1] ** 2 + exp1[2] ** 2
        vv13 = exp1[0] ** 2 + exp1[2] ** 2
        l1v123321 = (v23 * v21 * vv12 * vv23) ** (-0.5)
        l2v123321 = (v21 * (vv12 * vv23 * exp1[2] ** 2 + exp1[1] ** 2 * (
                exp1[2] ** 2 * v21 + exp1[0] ** 2 * exp1[1] ** 2))) ** (-0.5)
        l3v123321 = (v23 * (vv12 * vv23 * exp1[0] ** 2 + exp1[1] ** 2 * (
                exp1[2] ** 2 * v21 + exp1[0] ** 2 * exp1[1] ** 2))) ** (-0.5)
        l4v123321 = (vv12 * vv13 * exp1[1] ** 2 * exp1[2] ** 2 + vv23 * vv13 * exp1[0] ** 2 * exp1[
            1] ** 2 + vv12 * vv23 * exp1[0] ** 2 * exp1[2] ** 2) ** (-0.5)
        w123321 = w123321 * (l1v123321 - l2v123321 - l3v123321 + l4v123321)
        w123321 = w123321 * weight_xi[1, 2] * weight_xi[1, 0]
        m123321 = (X[0, :] - X[1, :]).reshape(-1, 1) @ (X[2, :] - X[1, :]).reshape(1, -1)
        m123321 = 0.5 * w123321 * (m123321 + m123321.T)

        w = w + np.array([w123123, w123132, w123213, w123231, w123312, w123321]).reshape(-1)
        mtrcs = mtrcs + np.array([m123123, m123132, m123213, m123231, m123312, m123321])
        fm = fm + m123123 + m123132 + m123213 + m123231 + m123312 + m123321

    return fm / 6, w / 6, mtrcs / 6


# In[11]:


def plugin(data, beta):
    # plugin estimator of covariance when delta >0
    beta = np.array(beta).reshape(-1, 1)

    X_beta = data.X @ beta.reshape(-1, 1)
    ep1 = np.exp(X_beta).reshape(-1, 1)
    vec = np.exp(2 * X_beta).reshape(-1, 1)

    # estimate N^delta*C_alpha
    div = data.N * np.sqrt(2) * np.sum(data.A) / ((data.N - 1) * np.sum(ep1))

    # estimate H
    p = data.X.shape[1]  # 动态获取特征维度
    H = np.zeros((p, p))
    for i in range(data.N):
        pi_i2i3 = np.sqrt(vec.T / (vec[i] + 2 * vec.T))
        weight_H = (1 - 2 * pi_i2i3 ** 2) * ep1[i] * ep1.reshape(1, -1) / (np.sqrt(3) * (1 - pi_i2i3))
        di_all = data.X[i, :] - data.X
        # compute H
        H += di_all.T @ (di_all * (weight_H.reshape(-1, 1)))

    M = None
    # estimate M1
    num_blocks1 = int(data.N / 4)
    indices1 = np.array_split(np.arange(data.N), num_blocks1)
    M1 = np.zeros((p, p))
    for ids in indices1:
        M1 = M1 + compute_M1_block(ep1[ids], data.X[ids, :], p)[0]  # 传入特征维度p
    M1 = M1 / num_blocks1

    if data.delta == 0:
        # estimate M0
        num_blocks0 = int(data.N / 3)
        indices0 = np.array_split(np.arange(data.N), num_blocks0)
        M0 = np.zeros((p, p))
        for ids in indices0:
            M0 = M0 + compute_M0_block(ep1[ids[0:3]], data.X[ids[0:3], :], p)[0]  # 传入特征维度p
        M0 = M0 / num_blocks0
        M = (M1 + M0 / div)
    else:
        M = M1
    H = H / (data.N * (data.N - 1))
    cov = np.linalg.inv(H) @ M @ np.linalg.inv(H) / (div * data.N)
    return cov, M, H


# In[12]:

if __name__ == "__main__":
    from joblib import Parallel, delayed
    import pickle
    from tqdm import tqdm

    beta = [-0.2, 0.2, -0.1, 0.1, 0]
    C_max = 25
    C_min = 9


    def map_fun(b):
        d = Pop(N, beta, delta, C_min, C_max, N + b + int(10000 * delta))
        true_beta = np.array(beta).reshape(-1)
        data_NR = Pop_NR(d)
        beta_hat = data_NR.run(running_parameter=beta)
        beta_hat = beta_hat.reshape(-1)
        est_std = np.sqrt(np.diag(plugin(d, beta_hat)[0])).reshape(-1)
        bi_cover = np.logical_and(((beta_hat - 1.96 * est_std) < true_beta), ((beta_hat + 1.96 * est_std) > true_beta))
        print(f"{b}:{bi_cover}")
        return beta_hat, true_beta, est_std, bi_cover


    B = 1000
    Tasks = list(range(B))
    for N in [5000]:
        for delta in [0.25]:
            d0 = Pop(N, beta, delta, C_min, C_max)
            print("N: ", d0.N)
            print("beta: ", beta)
            print("C_max: ", C_max)
            print("C_min: ", C_min)
            print("delta: ", delta)
            print("alpha:", d0.alpha)
            print("C_alpha:", d0.C_alpha)
            print("network nonzereos:", np.sum(d0.A))

            Results = Parallel(n_jobs=-1, backend='loky', verbose=10)(delayed(map_fun)(b) for b in range(B))

            estimators = np.array([est.reshape(-1) for est, _, _, _ in Results])  # B x (p+1)
            parameters = np.array([par.reshape(-1) for _, par, _, _ in Results])  # B x (p+1)
            plug_std = np.array([est_std.reshape(-1) for _, _, est_std, _ in Results])  # B x (p+1)
            cover = np.array([cv.reshape(-1) for _, _, _, cv in Results])  # B x (p+1)
            with open(f'results{N}delta{int(100 * delta)}.pkl', 'wb') as f:
                pickle.dump(Results, f)

            covariance = (estimators.T @ estimators) / B - (
                        estimators.mean(axis=0).reshape(-1, 1) @ estimators.mean(axis=0).reshape(1, -1))
            monte_std = np.sqrt(np.diag(covariance))
            ARE = np.mean(np.abs((plug_std / monte_std) - 1), axis=0)
            RMSE = np.sqrt(np.mean((estimators - parameters) ** 2, axis=0))
            std_estimation = np.mean(plug_std, axis=0)
            cover_rate = np.mean(cover, axis=0)

            print(f"N:{N};monte:", monte_std)
            print(f"N:{N};plug:", std_estimation)
            print(f"N:{N};RMSE:", RMSE)
            print(f"N:{N};ARE:", ARE)
            print(f"N:{N};cover:", cover_rate)


    # # results

    # In[13]:
    import pickle

    beta = [-0.2, 0.2, -0.1, 0.1, 0]
    C_max = 25
    C_min = 9
    for delta in [0.25]:
        B = 1000
        for N in [5000, 10000, 20000, 30000]:
            file_path = f'./results{N}delta{int(delta * 100)}.pkl'

            with open(file_path, 'rb') as file:
                Results = pickle.load(file)
                estimators = np.array([est.reshape(-1) for est, _, _, _ in Results])  # B x (p+1)
                parameters = np.array([par.reshape(-1) for _, par, _, _ in Results])  # B x (p+1)
                plug_std = np.array([est_std.reshape(-1) for _, _, est_std, _ in Results])  # B x (p+1)
                cover = np.array([cv.reshape(-1) for _, _, _, cv in Results])  # B x (p+1)

                covariance = (estimators.T @ estimators) / B - (
                            estimators.mean(axis=0).reshape(-1, 1) @ estimators.mean(axis=0).reshape(1, -1))
                monte_std = np.sqrt(np.diag(covariance))
                ARE = np.mean(np.abs((plug_std / monte_std) - 1), axis=0)
                RMSE = np.sqrt(np.mean((estimators - parameters) ** 2, axis=0))
                std_estimation = np.mean(plug_std, axis=0)
                cover_rate = np.mean(cover, axis=0)
                print("N: ", N)
                print("delta: ", delta)
                #             print(f"N:{N};monte:", monte_std)
                #             print(f"N:{N};plug:", std_estimation)
                print(f"N:{N};RMSE:", RMSE)
                print(f"N:{N};ARE:", ARE)
                print(f"N:{N};cover:", cover_rate)

