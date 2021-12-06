import numpy as np

# ベクトルに直して掛け算
class EnergyRNN:
    def __init__(self, dim):
        self.dim = dim
        self.W = np.zeros((dim, dim, dim, dim))
        self.theta = np.zeros((dim, dim))
        self.const = 0
        self.E = 0

    def __set_weight(self, weight, i, j, k, l):
        self.W[i, j, k, l] = weight

    def __set_theta(self, theta, i, j):
        self.theta[i, j] = theta

    def __set_const(self, const):
        self.const = const

    def __calc_selected_threshold(self, dim, i, j):
        X = np.zeros((dim, dim))
        X[i, j] = 1
        E = EnergyEQP(X)
        theta = E.calc() - self.const
        self.__set_theta(theta, i, j)

    def __calc_selected_weight(self, dim, i, j, k, l):
        X = np.zeros((dim, dim))
        X[i, j] = 1
        X[k, l] = 1
        E = EnergyEQP(X)
        weight = -2 * (E.calc() - (self.theta[i, j] + self.theta[k, l]) - self.const)
        self.__set_weight(weight, i, j, k, l)

    def __calc_selected_const(self, dim):
        X = np.zeros((dim, dim))
        E = EnergyEQP(X)
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
    def __init__(self, X):
        self.X = X

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

    def __calc_al_to_br(self):
        mat = self.X
        mat = mat - 1
        dim = self.X.shape[0]
        diag_sum = 0
        for i in range((-1) * (dim - 1), dim):
            diag_sum += np.square(np.diag(mat, k=i).sum(0))
        return diag_sum

    def __calc_ar_to_bl(self):
        mat = np.fliplr(self.X)
        mat = mat - 1
        dim = self.X.shape[0]
        diag_sum = 0
        for i in range((-1) * (dim - 1), dim):
            diag_sum += np.square(np.diag(mat, k=i).sum(0))
        return diag_sum

    def calc(self):
        energy = 0
        energy += self.__calc_hor()
        energy += self.__calc_ver()
        energy += self.__calc_al_to_br()
        energy += self.__calc_ar_to_bl()
        return energy

def main(dim):
    E = EnergyRNN(dim)
    E.calc_coefficient()
    E.print_theta()
    E.print_w()

if __name__ == "__main__":
    # (i)
    dim = 4
    main(dim)

    dim = 8
    main(dim)

    # (ii)

    # (iii)