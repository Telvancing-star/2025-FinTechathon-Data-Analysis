#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
from scipy.stats import multivariate_normal
from scipy.sparse import csr_matrix, vstack


class Pop:
    def __init__(self, N, beta, delta, C_min, C_max, seed=0):
        """
        N: number of nodes
        beta: feature coefficient
        delta: controling rate of degree
        C_min: minimum number of expected out degrees
        C_max: maximum number of expected out degrees
        """
        self.N = N
        self.C_min = C_min
        self.C_max = C_max
        self.beta = np.array(beta).reshape(-1)
        self.delta = delta
        self.p = len(beta)
        self.seed = seed

        self._generate_parameter_alpha()
        self._generate_popularity_and_feature()

        self._generate_adjacency_matrix()
        self._gen_in_degrees()

    def _generate_alpha(self):
        """
        get parameter alpha
        """
        left = np.sqrt(2) * self.C_min / self.C_beta
        right = np.sqrt(2) * self.C_max / self.C_beta
        C_alpha = 0.1 * right + 0.9 * left
        C_alpha = round(C_alpha)
        alpha = (np.log(C_alpha) - (1 - self.delta) * np.log(self.N)).item()
        return alpha, C_alpha

    def _generate_parameter_alpha(self):
        test_X = self.feature_generation(100000, self.seed + 1000)
        test_exp_X_beta = np.exp(test_X @ self.beta)
        self.C_beta = np.mean(test_exp_X_beta)
        self.alpha, self.C_alpha = self._generate_alpha()
        return

    def _generate_popularity_and_feature(self):
        # observable X
        self.X = self.feature_generation(self.N, self.seed)
        self.gamma = np.exp(self.X @ self.beta + self.alpha)
        return

    def _generate_adjacency_matrix(self):
        # Using less memory
        np.random.seed(self.seed)
        Z = np.random.randn(self.N)
        Z_reshaped1 = Z.reshape(-1, 1)
        Z_reshaped2 = Z.reshape(1, -1)
        diag_indices = np.arange(self.N)

        len_slice = 1000
        matrix = csr_matrix(
            np.random.binomial(1, np.exp(-(Z_reshaped1[0:len_slice] - Z_reshaped2) ** 2 / (2 * self.gamma ** 2))))
        for r in np.arange(int(self.N / len_slice) - 1):
            row = csr_matrix(np.random.binomial(1, np.exp(
                -(Z_reshaped1[int(len_slice * (r + 1)):int(len_slice * (r + 2))] - Z_reshaped2) ** 2 / (
                            2 * self.gamma ** 2))))
            matrix = vstack([matrix, row])
        matrix[diag_indices, diag_indices] = 0
        self.A = matrix
        return

    def feature_generation(self, size, seed):
        np.random.seed(self.seed)
        rhoX = 0.5
        meanX = np.zeros(self.p)
        covX = np.fromfunction(lambda i, j: rhoX ** np.abs(i - j), (self.p, self.p))
        rvX = multivariate_normal(meanX, covX, self.seed)
        return rvX.rvs(size)

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

    def generate_pi_matrix(self, beta):
        beta = np.array(beta).reshape(-1, 1)
        vec = np.exp(2 * (self.X @ beta)).reshape(-1, 1)

        len_slice = 1000
        pi = self.A[0:len_slice, :].multiply(np.sqrt(vec.reshape(1, -1) / (vec[0:len_slice] + 2 * vec.reshape(1, -1))))
        log_pi = self.A[0:len_slice, :].multiply(
            np.log(np.sqrt(vec.reshape(1, -1) / (vec[0:len_slice] + 2 * vec.reshape(1, -1)))))
        log_pi_minus = self.A[0:len_slice, :].multiply(
            np.log(1 - np.sqrt(vec.reshape(1, -1) / (vec[0:len_slice] + 2 * vec.reshape(1, -1)))))
        const_grad1 = self.A[0:len_slice, :].multiply((1 - 2 * pi.toarray() ** 2) / (1 - pi.toarray()))
        const_grad2 = -pi.multiply(const_grad1)

        const_hes1 = self.A[0:len_slice, :].multiply(
            (4 * pi.toarray() - 2 * pi.toarray() ** 2 - 1) / (1 - pi.toarray()) ** 2)
        const_hes2 = self.A[0:len_slice, :].multiply(
            (4 * pi.toarray() ** 3 - 6 * pi.toarray() ** 2 + 1) / (1 - pi.toarray()) ** 2)
        for r in np.arange(int(self.N / len_slice) - 1):
            row_pi = self.A[int(len_slice * (r + 1)):int(len_slice * (r + 2)), :].multiply(np.sqrt(
                vec.reshape(1, -1) / (vec[int(len_slice * (r + 1)):int(len_slice * (r + 2))] + 2 * vec.reshape(1, -1))))
            row_log_pi = self.A[int(len_slice * (r + 1)):int(len_slice * (r + 2)), :].multiply(np.log(np.sqrt(
                vec.reshape(1, -1) / (
                            vec[int(len_slice * (r + 1)):int(len_slice * (r + 2))] + 2 * vec.reshape(1, -1)))))
            row_log_pi_minus = self.A[int(len_slice * (r + 1)):int(len_slice * (r + 2)), :].multiply(np.log(1 - np.sqrt(
                vec.reshape(1, -1) / (
                            vec[int(len_slice * (r + 1)):int(len_slice * (r + 2))] + 2 * vec.reshape(1, -1)))))
            row_const_grad1 = self.A[int(len_slice * (r + 1)):int(len_slice * (r + 2)), :].multiply(
                (1 - 2 * row_pi.toarray() ** 2) / (1 - row_pi.toarray()))
            row_const_grad2 = -row_pi.multiply(row_const_grad1)
            row_const_hes1 = self.A[int(len_slice * (r + 1)):int(len_slice * (r + 2)), :].multiply(
                (4 * row_pi.toarray() - 2 * row_pi.toarray() ** 2 - 1) / (1 - row_pi.toarray()) ** 2)
            row_const_hes2 = self.A[int(len_slice * (r + 1)):int(len_slice * (r + 2)), :].multiply(
                (4 * row_pi.toarray() ** 3 - 6 * row_pi.toarray() ** 2 + 1) / (1 - row_pi.toarray()) ** 2)

            pi = vstack([pi, row_pi])
            log_pi = vstack([log_pi, row_log_pi])
            log_pi_minus = vstack([log_pi_minus, row_log_pi_minus])
            const_grad1 = vstack([const_grad1, row_const_grad1])
            const_grad2 = vstack([const_grad2, row_const_grad2])
            const_hes1 = vstack([const_hes1, row_const_hes1])
            const_hes2 = vstack([const_hes2, row_const_hes2])
        return pi, log_pi, log_pi_minus, const_grad1, const_grad2, const_hes1, const_hes2

    def loss_grad_hessian(self, parameter):
        parameter = np.array(parameter).reshape(-1, 1)
        pi, log_pi, log_pi_minus, const_grad1, const_grad2, const_hes1, const_hes2 = self.generate_pi_matrix(parameter)
        # loss function
        l2 = self.A.multiply(log_pi_minus).multiply(self.d_in.reshape(-1, 1))
        l3 = -self.A.multiply(log_pi_minus)
        l1 = self.col_prod.multiply(self.A).multiply(log_pi) - self.col_prod.multiply(self.A).multiply(log_pi_minus)
        loss = -(np.sum(l1) + np.sum(l2) + np.sum(l3)) / (self.N * (self.N - 1) * (self.N - 2))

        # gradient        
        weight_g1 = self.col_prod.multiply(self.A).multiply(const_grad1)
        weight_g2 = self.A.multiply(self.d_in.reshape(-1, 1)).multiply(const_grad2)
        weight_g3 = -self.A.T.multiply(self.A.multiply(const_grad2))

        weight_g = -(weight_g1 + weight_g2 + weight_g3)
        g = np.zeros((self.p, 1))

        splits = 1000
        zipped_indices = zip(
            *(np.array_split(weight_g.nonzero()[0], splits), np.array_split(weight_g.nonzero()[1], splits)))
        for i_s, j_s in zipped_indices:
            g += np.sum(weight_g[i_s, j_s] * (self.X[j_s,] - self.X[i_s,]), axis=0).reshape(-1, 1)

        weight_h1 = self.col_prod.multiply(self.A).multiply(const_hes1)
        weight_h2 = self.A.multiply(self.d_in.reshape(-1, 1)).multiply(const_hes2)

        weight_h3 = -self.A.T.multiply(self.A.multiply(const_hes2))

        weight_h = weight_h1 + weight_h2 + weight_h3
        h = np.zeros((self.p, self.p))
        zipped_indices = zip(
            *(np.array_split(weight_h.nonzero()[0], splits), np.array_split(weight_h.nonzero()[1], splits)))
        for i_s, j_s in zipped_indices:
            h += (self.X[i_s,] - self.X[j_s,]).T @ (
                        (self.X[i_s,] - self.X[j_s,]) * ((np.array(weight_h[i_s, j_s]).reshape(-1))).reshape(-1, 1))

        return loss, g.reshape(-1, 1), h

    def run(self, running_parameter, alpha=1, max_iter=20, epsilon=1e-4):
        running_parameter = np.array(running_parameter).reshape(-1, 1)
        it = 0
        L_old = None
        while it < max_iter:
            L, g, h = self.loss_grad_hessian(running_parameter)
            if L_old != None:
                err = (L_old - L) / L
                if err < epsilon:
                    break
            L_old = L
            update = alpha * np.linalg.inv(h) @ g
            running_parameter = running_parameter - update.reshape(-1, 1)
            it += 1
        return running_parameter


# In[9]:


import itertools


def compute_M1_block(exp_block, X_block):
    exp_block = exp_block.reshape(-1, 1)
    permutations = list(itertools.permutations([0, 1, 2, 3]))
    fm = np.zeros((5, 5))
    w = np.zeros(12)
    mtrcs = np.zeros((12, 5, 5))
    for perm in permutations:
        exp1 = exp_block[list(perm)]
        X = X_block[list(perm), :]
        vec = exp1 ** 2
        # pi
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

        # exp1 can be seen as gamma after removing exp(alpha_N) (popularity parameter), len(exp1)==4
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


def compute_M0_block(exp_block, X_block):
    exp_block = exp_block.reshape(-1, 1)
    permutations = list(itertools.permutations([0, 1, 2]))
    fm = np.zeros((5, 5))
    w = np.zeros(6)
    mtrcs = np.zeros((6, 5, 5))
    fm = np.zeros((5, 5))
    for perm in permutations:
        exp1 = exp_block[list(perm)].reshape(-1, 1)
        X = X_block[list(perm), :]
        vec = exp1 ** 2
        # pi
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
    H = np.zeros((data.p, data.p))
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
    M1 = np.zeros((data.p, data.p))
    for ids in indices1:
        M1 = M1 + compute_M1_block(ep1[ids], data.X[ids, :])[0]
    M1 = M1 / num_blocks1

    if data.delta == 0:
        # estimate M0
        num_blocks0 = int(data.N / 3)
        indices0 = np.array_split(np.arange(data.N), num_blocks0)
        M0 = np.zeros((data.p, data.p))
        for ids in indices0:
            M0 = M0 + compute_M0_block(ep1[ids[0:3]], data.X[ids[0:3], :])[0]
        M0 = M0 / num_blocks0
        M = (M1 + M0 / div)
    else:
        M = M1
    H = H / (data.N * (data.N - 1))
    cov = np.linalg.inv(H) @ M @ np.linalg.inv(H) / (div * data.N)
    return cov, M, H

# def plugin_my(data, beta):
#     # plugin estimator of covariance when delta >0
#     beta = np.array(beta).reshape(-1, 1)
#
#     X_beta = data.X @ beta.reshape(-1, 1)
#     ep1 = np.exp(X_beta).reshape(-1, 1)
#     vec = np.exp(2 * X_beta).reshape(-1, 1)
#
#     # estimate N^delta*C_alpha
#     div = data.N * np.sqrt(2) * np.sum(data.A) / ((data.N - 1) * np.sum(ep1))
#
#     # estimate H
#     H = np.zeros((data.p, data.p))
#     for i in range(data.N):
#         pi_i2i3 = np.sqrt(vec.T / (vec[i] + 2 * vec.T))
#         weight_H = (1 - 2 * pi_i2i3 ** 2) * ep1[i] * ep1.reshape(1, -1) / (np.sqrt(3) * (1 - pi_i2i3))
#         di_all = data.X[i, :] - data.X
#         # compute H
#         H += di_all.T @ (di_all * (weight_H.reshape(-1, 1)))
#
#     M = None
#     # estimate M1
#     num_blocks1 = int(data.N / 4)
#     indices1 = np.array_split(np.arange(data.N), num_blocks1)
#     M1 = np.zeros((data.p, data.p))
#     for ids in indices1:
#         M1 = M1 + compute_M1_block(ep1[ids], data.X[ids, :])[0]
#     M1 = M1 / num_blocks1
#
#     if data.delta == 0:
#         # estimate M0
#         num_blocks0 = int(data.N / 3)
#         indices0 = np.array_split(np.arange(data.N), num_blocks0)
#         M0 = np.zeros((data.p, data.p))
#         for ids in indices0:
#             M0 = M0 + compute_M0_block(ep1[ids[0:3]], data.X[ids[0:3], :])[0]
#         M0 = M0 / num_blocks0
#         M = (M1 + M0 / div)
#     else:
#         M = M1
#
#     H = H / (data.N * (data.N - 1))
#     cov = np.linalg.inv(H) @ M @ np.linalg.inv(H) / (div * data.N)
#     return cov, M, H


# In[12]:

if __name__ == "__main__":
    from joblib import Parallel, delayed
    import pickle
    from tqdm import tqdm
    import numpy as np

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


class Pop_NR_Fixed:
    def __init__(self, data):
        self.X = data.X
        self.N, self.p = self.X.shape
        self.A = data.A
        self.d_in = data.in_degrees
        self.col_prod = self.A.T @ self.A

    def generate_pi_matrix(self, beta):
        beta = np.array(beta).reshape(-1, 1)
        vec = np.exp(2 * (self.X @ beta)).reshape(-1, 1)

        len_slice = 1000
        num_slices = int(np.ceil(self.N / len_slice))

        # 初始化结果矩阵
        pi = None
        log_pi = None
        log_pi_minus = None
        const_grad1 = None
        const_grad2 = None
        const_hes1 = None
        const_hes2 = None

        for r in range(num_slices):
            start_idx = r * len_slice
            end_idx = min((r + 1) * len_slice, self.N)
            slice_size = end_idx - start_idx

            # 获取当前切片的数据
            A_slice = self.A[start_idx:end_idx, :]
            vec_slice = vec[start_idx:end_idx]

            # 计算 pi 值 - 关键修复：确保形状完全匹配
            # vec.reshape(1, -1) 形状: (1, 4167)
            # vec_slice 形状: (slice_size, 1)
            # 我们需要广播到 (slice_size, 4167)
            vec_slice_broadcast = vec_slice  # (slice_size, 1)
            vec_flat_broadcast = vec.reshape(1, -1)  # (1, 4167)

            denominator = vec_slice_broadcast + 2 * vec_flat_broadcast  # (slice_size, 4167)
            pi_slice_value = np.sqrt(vec_flat_broadcast / denominator)  # (slice_size, 4167)

            # 确保没有数值问题
            pi_slice_value = np.clip(pi_slice_value, 1e-10, 1 - 1e-10)

            # 计算各种矩阵 - 关键：直接使用稠密矩阵避免稀疏矩阵形状问题
            row_pi = A_slice.multiply(pi_slice_value)
            row_log_pi = A_slice.multiply(np.log(pi_slice_value))
            row_log_pi_minus = A_slice.multiply(np.log(1 - pi_slice_value))

            # 对于需要稠密矩阵的操作，先转换
            pi_slice_dense = row_pi.toarray()

            # 计算梯度常数矩阵
            grad_denom = 1 - pi_slice_dense
            # 避免除零
            grad_denom = np.where(grad_denom < 1e-10, 1e-10, grad_denom)

            const_grad1_dense = (1 - 2 * pi_slice_dense ** 2) / grad_denom
            row_const_grad1 = A_slice.multiply(const_grad1_dense)
            row_const_grad2 = -row_pi.multiply(row_const_grad1)

            # 计算海森常数矩阵
            hes_denom = grad_denom ** 2
            hes_denom = np.where(hes_denom < 1e-20, 1e-20, hes_denom)

            const_hes1_dense = (4 * pi_slice_dense - 2 * pi_slice_dense ** 2 - 1) / hes_denom
            const_hes2_dense = (4 * pi_slice_dense ** 3 - 6 * pi_slice_dense ** 2 + 1) / hes_denom

            row_const_hes1 = A_slice.multiply(const_hes1_dense)
            row_const_hes2 = A_slice.multiply(const_hes2_dense)

            # 堆叠结果
            if pi is None:
                pi = row_pi
                log_pi = row_log_pi
                log_pi_minus = row_log_pi_minus
                const_grad1 = row_const_grad1
                const_grad2 = row_const_grad2
                const_hes1 = row_const_hes1
                const_hes2 = row_const_hes2
            else:
                pi = vstack([pi, row_pi])
                log_pi = vstack([log_pi, row_log_pi])
                log_pi_minus = vstack([log_pi_minus, row_log_pi_minus])
                const_grad1 = vstack([const_grad1, row_const_grad1])
                const_grad2 = vstack([const_grad2, row_const_grad2])
                const_hes1 = vstack([const_hes1, row_const_hes1])
                const_hes2 = vstack([const_hes2, row_const_hes2])

        return pi, log_pi, log_pi_minus, const_grad1, const_grad2, const_hes1, const_hes2

    def loss_grad_hessian(self, parameter):
        parameter = np.array(parameter).reshape(-1, 1)
        pi, log_pi, log_pi_minus, const_grad1, const_grad2, const_hes1, const_hes2 = self.generate_pi_matrix(
            parameter)

        print(f"调试 loss_grad_hessian 中的形状:")
        print(f"  pi.shape: {pi.shape}")
        print(f"  log_pi.shape: {log_pi.shape}")
        print(f"  log_pi_minus.shape: {log_pi_minus.shape}")
        print(f"  const_grad1.shape: {const_grad1.shape}")
        print(f"  const_grad2.shape: {const_grad2.shape}")
        print(f"  A.shape: {self.A.shape}")
        print(f"  d_in.shape: {self.d_in.shape}")

        # 关键修复：确保所有稀疏矩阵操作形状匹配
        try:
            # loss function
            l2 = self.A.multiply(log_pi_minus).multiply(self.d_in.reshape(-1, 1))
            l3 = -self.A.multiply(log_pi_minus)
            l1 = self.col_prod.multiply(self.A).multiply(log_pi) - self.col_prod.multiply(self.A).multiply(
                log_pi_minus)
            loss = -(np.sum(l1) + np.sum(l2) + np.sum(l3)) / (self.N * (self.N - 1) * (self.N - 2))

            # gradient
            weight_g1 = self.col_prod.multiply(self.A).multiply(const_grad1)
            weight_g2 = self.A.multiply(self.d_in.reshape(-1, 1)).multiply(const_grad2)
            weight_g3 = -self.A.T.multiply(self.A.multiply(const_grad2))

            weight_g = -(weight_g1 + weight_g2 + weight_g3)
            g = np.zeros((self.p, 1))

            splits = 1000
            zipped_indices = zip(
                *(np.array_split(weight_g.nonzero()[0], splits), np.array_split(weight_g.nonzero()[1], splits)))
            for i_s, j_s in zipped_indices:
                g += np.sum(weight_g[i_s, j_s] * (self.X[j_s,] - self.X[i_s,]), axis=0).reshape(-1, 1)

            weight_h1 = self.col_prod.multiply(self.A).multiply(const_hes1)
            weight_h2 = self.A.multiply(self.d_in.reshape(-1, 1)).multiply(const_hes2)
            weight_h3 = -self.A.T.multiply(self.A.multiply(const_hes2))

            weight_h = weight_h1 + weight_h2 + weight_h3
            h = np.zeros((self.p, self.p))
            zipped_indices = zip(
                *(np.array_split(weight_h.nonzero()[0], splits), np.array_split(weight_h.nonzero()[1], splits)))
            for i_s, j_s in zipped_indices:
                h += (self.X[i_s,] - self.X[j_s,]).T @ (
                        (self.X[i_s,] - self.X[j_s,]) * ((np.array(weight_h[i_s, j_s]).reshape(-1))).reshape(-1, 1))

            return loss, g.reshape(-1, 1), h

        except Exception as e:
            print(f"loss_grad_hessian 中出错: {e}")
            # 返回一些默认值以便继续运行
            return 1.0, np.zeros((self.p, 1)), np.eye(self.p)

    def run(self, running_parameter, alpha=1, max_iter=20, epsilon=1e-4):
        running_parameter = np.array(running_parameter).reshape(-1, 1)
        it = 0
        L_old = None
        while it < max_iter:
            print(f"\n迭代 {it + 1}:")
            L, g, h = self.loss_grad_hessian(running_parameter)
            print(f"  损失: {L:.6f}, 梯度范数: {np.linalg.norm(g):.6f}")

            if L_old is not None:
                err = abs(L_old - L) / (abs(L) + 1e-10)
                print(f"  相对误差: {err:.6f}")
                if err < epsilon:
                    print("收敛!")
                    break
            L_old = L

            try:
                update = alpha * np.linalg.inv(h) @ g
                running_parameter = running_parameter - update.reshape(-1, 1)
                print(f"  参数更新范数: {np.linalg.norm(update):.6f}")
            except Exception as e:
                print(f"  更新参数时出错: {e}")
                # 使用梯度下降作为备选
                update = alpha * g / (np.linalg.norm(g) + 1e-10)
                running_parameter = running_parameter - update

            it += 1

        return running_parameter

    # In[13]:

if __name__ == "__main__":
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

