import numpy as np


class CaMF():
    def __init__(self, D, lambda_1, lambda_2, gamma_1, gamma_2, beta, max_iter):
        self.D = D
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.beta = beta
        self.max_iter = max_iter

    def get_I_i(self, i):
        return self.R.tocsr().getrow(i).indices

    def get_U_j(self, j):
        return self.R.tocsc().getcol(j).indices

    def get_p_i(self, i):
        A = np.zeros((self.D, self.D))

        # \sum_{j \in \mathbb{I}_i} q_j q_j^\top
        for j in self.get_I_i(i):
            A += self.Q[j].reshape(-1, 1).dot(self.Q[j].reshape(1, -1))

        # \beta \sum_{j} q_j q_j^\top
        for j in range(self.M):
            A += self.beta * \
                self.Q[j].reshape(-1, 1).dot(self.Q[j].reshape(1, -1))

        # \lambda_1 I
        A += self.lambda_1 * np.identity(self.D)

        b = np.zeros(self.D)

        # \sum_{j \in \mathbb{I}_i} r_{ij} q_j
        for j in self.get_I_i(i):
            b += self.R[i, j] * self.Q[j]

        # -\lambda_1 U^\top x_i
        b -= self.lambda_1 * self.U.T.dot(self.X[i])

        return np.linalg.solve(A, b)

    def get_P(self):
        P = np.empty((self.N, self.D))

        for i in range(self.N):
            P[i] = self.get_p_i(i)

        return P

    def get_q_j(self, j):
        A = np.zeros((self.D, self.D))

        # \sum_{i \in \mathbb{U}_j} p_i p_i^\top
        for i in self.get_U_j(j):
            A += self.P[i].reshape(-1, 1).dot(self.P[i].reshape(1, -1))

        # \beta \sum_{i} p_i p_i^\top
        for i in range(self.N):
            A += self.beta * \
                self.P[i].reshape(-1, 1).dot(self.P[i].reshape(1, -1))

        # \lambda_2 I
        A += self.lambda_2 * np.identity(self.D)

        b = np.zeros(self.D)

        # \sum_{i \in \mathbb{U}_j} r_{ij} p_i
        for i in self.get_U_j(j):
            b += self.R[i, j] * self.P[i]

        # -\lambda_2 V^\top y_j
        b -= self.lambda_1 * self.V.T.dot(self.Y[j])

        return np.linalg.solve(A, b)

    def get_Q(self):
        Q = np.empty((self.M, self.D))

        for j in range(self.M):
            Q[j] = self.get_q_j(j)

        return Q

    def get_U(self):
        A = self.X.T.dot(self.X) + self.gamma_1 / \
            self.lambda_1 * np.identity(self.F)
        B = self.X.T.dot(self.P)

        return np.linalg.solve(A, B)

    def get_V(self):
        A = self.Y.T.dot(self.Y) + self.gamma_2 / \
            self.lambda_2 * np.identity(self.L)
        B = self.Y.T.dot(self.Q)

        return np.linalg.solve(A, B)

    def fit(self, R, X, Y):
        self.R = R
        self.X = X
        self.Y = Y

        self.N, self.M = R.shape
        _, self.F = X.shape
        _, self.L = Y.shape

        self.P = np.random.randn(self.N, self.D)
        self.Q = np.random.randn(self.M, self.D)
        self.U = np.random.randn(self.F, self.D)
        self.V = np.random.randn(self.L, self.D)

        for it in range(self.max_iter):
            self.P = self.get_P()
            self.Q = self.get_Q()
            self.U = self.get_U()
            self.V = self.get_V()

        return self
