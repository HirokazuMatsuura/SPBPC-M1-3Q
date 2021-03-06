import numpy as np
import random

# ベクトルに直して掛け算
class EnergyRNN:
    def __init__(self, dim, d, beta):
        self.dim = dim
        self.W = np.zeros((dim, dim, dim, dim))
        self.theta = np.zeros((dim, dim))
        self.const = 0
        self.E = 0
        self.d = d
        self.beta = beta

    def __set_weight(self, weight, i, j, k, l):
        self.W[i, j, k, l] = weight

    def __set_theta(self, theta, i, j):
        self.theta[i, j] = theta

    def __set_const(self, const):
        self.const = const

    def __calc_selected_threshold(self, dim, i, j):
        X = np.zeros((dim, dim))
        X[i, j] = 1
        E = EnergyEQP(X, self.d, self.beta)
        theta = E.calc() - self.const
        self.__set_theta(theta, i, j)

    def __calc_selected_weight(self, dim, i, j, k, l):
        X = np.zeros((dim, dim))
        X[i, j] = 1
        X[k, l] = 1
        E = EnergyEQP(X, self.d, self.beta)
        weight = -2 * (E.calc() - (self.theta[i, j] + self.theta[k, l]) - self.const)
        self.__set_weight(weight, i, j, k, l)

    def __calc_selected_const(self, dim):
        X = np.zeros((dim, dim))
        E = EnergyEQP(X, self.d, self.beta)
        const = E.calc()
        self.__set_const(const)

    def calc_coefficient(self):
        self.__calc_selected_const(self.dim)

        for i in range(self.dim):
            for j in range(self.dim):
                self.__calc_selected_threshold(self.dim, i, j)

        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    for l in range(self.dim):
                        self.__calc_selected_weight(self.dim, i, j, k, l)

    def print_w(self):
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    for l in range(self.dim):
                        print("w[" + str(i + 1) + " " + str(j + 1) + ", " + str(k + 1) + " " + str(l + 1) + "] = " + str(int(self.W[i, j, k, l])))

    def print_theta(self):
        for i in range(self.dim):
            for j in range(self.dim):
                print("theta[" + str(i + 1) + ", " + str(j + 1) + "] = " + str(int(self.theta[i, j])))

class EnergyEQP:
    def __init__(self, X, d, beta):
        self.X = X
        self.d = d
        self.beta = beta

    def __calc_hor(self):
        mat = self.X
        mat = mat - 1
        mat = mat.sum(1)
        mat = np.dot(mat, mat)
        mat = mat.sum(0)
        return mat

    def __calc_ver(self):
        mat = self.X
        mat = mat - 1
        mat = mat.sum(0)
        mat = np.dot(mat, mat)
        mat = mat.sum(0)
        return mat

    def __El(self):
        mat = 0
        for n in range(self.X.shape[0]):
            for m in range(self.X.shape[0]):
                for j in range(self.X.shape[0]):
                    if j + 1 == self.X.shape[0]:
                        mat += self.X[n, j] * self.X[m, 0] * self.d[n, m]
                    else:
                        mat += self.X[n, j] * self.X[m, (j + 1)] * self.d[n, m]
        return mat

    def __Ec(self):
        return self.__calc_hor() + self.__calc_ver()

    def __Etotal(self):
        return self.__El() + self.beta * self.__Ec()


    def calc(self):
        return self.__Etotal()

class DeterministicModel:
    def __init__(self, W):
        self.W = W
        self.X = np.zeros((4, 4))

    def __sigmoid(self, index_k, index_l):
        sig = 0
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[0]):
                sig += self.W[i, j, index_k, index_l] * self.X[i, j]
        return sig

    def __update(self, index_i, index_j):
        if self.__sigmoid(index_i, index_j) >= 0:
            self.X[index_i, index_j] = 1
        else:
            self.X[index_i, index_j] = 0
    
    def update_model(self):
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[0]):
                self.__update(i, j)

    def print_x(self):
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[0]):
                print("X_" + str(i) + "_" + str(j) + " = " + str(self.X[i, j]))

class ProbalisticModel:
    def __init__(self, W, alpha, d, beta):
        self.W = W
        self.X = np.zeros((4, 4))
        self.alpha = alpha
        self.d = d
        self.beta = beta

    def __p_E(self, index_k, index_l):
        A = 0.5
        E = EnergyEQP(self.X, self.d, self.beta)
        return A * np.exp((-1) * self.alpha * E.calc())

    def __update(self, index_i, index_j):
        bin = [0, 1]
        prob_w = [1 - self.__p_E(index_i, index_j), self.__p_E(index_i, index_j)]
        choice = random.choices(bin, k=1, weights=prob_w)
        self.X[index_i, index_j] = choice[0]
    
    def update_model(self):
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[0]):
                self.__update(i, j)

    def print_x(self):
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[0]):
                print("X_" + str(i) + "_" + str(j) + " = " + str(self.X[i, j]))

if __name__ == "__main__":
    # (i)
    print("(i)")
    dim = 4
    d = np.random.rand(4, 4)
    beta = 0.05
    
    E = EnergyRNN(dim, d, beta)
    E.calc_coefficient()
    E.print_theta()
    E.print_w()

    # (ii)
    print("(ii)")
    DM = DeterministicModel(E.W)
    DM.update_model()
    DM.print_x()

    # (iii)
    print("(iii)")
    alpha = 0.01
    PM = ProbalisticModel(E.W, alpha, d, beta)
    PM.update_model()
    PM.print_x()
