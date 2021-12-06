import numpy as np
import random
import math

# シグモイド関数
def sigmoid(alpha, s):
    e = math.e
    sigmoid = 1 / (1 + e ** (alpha * -s))
    return sigmoid

RANGE_MIN = -10
RANGE_MAX = 10
RAND_MAX = 2 ** 32 - 1
N = 10

xi = np.random.randint(RANGE_MIN, RANGE_MAX, (1, N))
wi = np.random.randint(RANGE_MIN, RANGE_MAX, (1, N))
s = np.sum(xi * wi)

alpha = 0.001
p = sigmoid(alpha, s)

trial = 10000
count = 0
for i in range(trial):
    random_parameter = random.uniform(0, RAND_MAX)
    if random_parameter <= (p * RAND_MAX):
        y = 1
    else:
        y = 0
    count += y

print("alpha = " + str(alpha) + " trial = " + str(trial))
print("p = " + str(p))
print("y = 1 probabirity: " + str(float(count) / float(trial)))