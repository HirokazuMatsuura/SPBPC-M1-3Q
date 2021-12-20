import numpy as np
import sys
import assignment2 as as2

"""
constant number
"""
N = 4

def Energy(X):
    x1, x2, x3, x4 = X[0], X[1], X[2], X[3]
    X1 = (1 * x1 + -1 * x2 + 2 * x3 + -2 * x4) ** 2
    X2 = (-2 * x1 + 1 * x2 + 1 * x3 + -1 * x4 + 1) ** 2
    X3 = (2 * x1 + -3 * x2 + -1 * x3 + 1 * x4 + 2) ** 2
    return X1 + X2 + X3

def C_calculation(N):
    X = [0] * N
    return Energy(X)

def theta_calculation(N):
    C = C_calculation(N)
    theta1 = Energy([1, 0, 0, 0]) - C
    theta2 = Energy([0, 1, 0, 0]) - C
    theta3 = Energy([0, 0, 1, 0]) - C
    theta4 = Energy([0, 0, 0, 1]) - C
    return [theta1, theta2, theta3, theta4]

def weight_element(N, i, j, C, theta):
    X = [0] * N
    if i == j:
        X[i] = 1
        return 2 * (theta[i] + C - Energy(X))
    else:
        X[i] = 1
        X[j] = 1
        return (theta[i] + theta[j] + C - Energy(X))


def weight_calculation(N, C, theta):
    weight = [[0] * N for i in range(N)]
    for i, row in enumerate(weight):
        for j, element in enumerate(row):
            weight[i][j] = weight_element(N, i, j, C, theta)
    return weight

def remake_weight(theta, weight):
    weight.insert(0, theta)
    for i, row in enumerate(weight):
        weight[i].insert(0, 0)
    return weight


if __name__ == "__main__":
    C = C_calculation(N)
    theta = theta_calculation(N)
    weight = weight_calculation(N, C, theta)

    print("C: " + str(C))
    print("theta: " + str(theta))
    print("w: " + str(weight))

    """
    for applying RNN, theta add weight
    theta[n] -> weight[0][n]
    """
    weight = np.array(remake_weight(theta, weight))

    """
    fixed the index of 0 like x = [1, 0or1, 0or1, 0or1, 0or1]
    """
    N = 5
    n_count = as2.BinaryPermutation(N)

    """
    you can change this parameter
    updating, alpha
    """
    updating = 100
    alpha = 0.01

    for n in range(updating):
        xi = np.random.randint(0, 1 + 1, (1, N))[0]
        xi[0] = 1
        count = as2.BinaryPermutation(N)
        for i in range(updating):
            """
            select the index in order, such as 1, 2, .., N, 1, 2, .., N
            (index "0" is fixed)
            """
            index = i % (N - 1) + 1
            s = as2.summation(xi, weight[:, index])
            p = as2.sigmoid(alpha, s)
            # probabilistic and binary model
            state = as2.probability_choice(p)
            xi = as2.value_writing(index, state, xi)
            count.search(xi)
            # check the ergodicity
            if i == (updating - 1):
                n_count.search(xi)
    print("updating = " + str(updating))

    # Probability of occurrence over time
    print("Probability of occurrence over time")
    for element in count.count:
        if element.label[0] == "0":
            continue
        else:
            print("[x1, x2, x3, x4] = " + element.label[1:N] + ": " + str(float(element.count) / float(updating)))
            # print(str(element.count/updating))
            # print(element.label[1:N])
    
    # Probability of state in the population in the final state
    print("Probability of state in the population in the final state")
    for element in n_count.count:
        if element.label[0] == "0":
            continue
        else:
            print("[x1, x2, x3, x4] = " + element.label[1:N] + ": " + str(float(element.count) / float(updating)))
            # print(str(element.count/updating))
            # print(element.label[1:N])