import numpy as np
import random
import math
import sys
import itertools

"""
constant number
"""
RANGE_MIN = -10
RANGE_MAX = 10
RAND_MAX = 2 ** 32 - 1
N = 5

"""
function
"""
def summation(xi, wi):
    """
    y = xi * wi
    return y
    """
    return np.sum(xi * wi)

def sigmoid(alpha, s):
    """
    sigmoid function
    """
    e = math.e
    sigmoid = 1 / (1 + e ** (alpha * -s))
    return sigmoid

def probability(p, RAND_MAX):
    """
    compare generating the random number with p * RAND_MAX
    if rand() <= p * RAND_MAX then returning 1, else returning 0
    """
    random_parameter = random.uniform(0, RAND_MAX)
    if random_parameter <= (p * RAND_MAX):
        return  1
    else:
        return 0

def value_writing(index, value, np_array):
    """
    if you choose index 1 and change value from 0 to 1
    [0, 0, 0, 0, 0] => [0, 1, 0, 0, 0]
    """
    np_array[index] = value
    return np_array

class Permutations:
    def __init__(self, label, count):
        """
        for example, the label is "0000" and the count is 10
        the count means times of the case
        """
        self.label = label
        self.count = count

    def increment(self):
        self.count = self.count + 1

class BinaryPermutation:
    """
    N = 2 => [0, 0], [0, 1], [1, 0], [1, 1]
    """

    def __init__(self, N):
        self.count = list()
        bin_array = [0, 1]
        """
        bin_array = [0, 1]
        the number of permitation_array is 2 ^ N
        """
        for perm in list(itertools.product(bin_array, repeat=N)):
            to_label = self._add_string(list(perm))
            item = Permutations(to_label, 0)
            self.count.append(item)

    def _add_string(self, array):
        strings = ""
        for a in array:
            strings += str(a)
        return strings
   
    def _add_count(self, index):
        self.count[index].increment()

    """
    {label: "00", count: 0}, {label: "01", count: 0}, ...
    ([0, 1] is given)=>
    {label: "00", count: 0}, {label: "01", count: 1}, ...
    """
    def search(self, np_list):
        search_label = self._add_string(np_list)
        for i, perm_class in enumerate(self.count):
            if perm_class.label == search_label:
                self._add_count(i)
                break

if __name__ == "__main__":
    """
    fixed the index of 0 like x = [1, 0or1, 0or1, 0or1, 0or1]
    """
    xi = np.random.randint(0, 1 + 1, (1, N))[0]
    xi[0] = 1

    wij = np.random.randint(RANGE_MIN, RANGE_MAX + 1, (N, N))
    count = BinaryPermutation(N)

    """
    you can change this parameter
    updating, alpha
    """
    updating = 1000
    alpha = 0.01

    for i in range(updating):
        """
        select the index in order, such as 1, 2, .., N, 1, 2, .., N
        (index "0" is fixed)
        """
        index = i % (N - 1) + 1
        s = summation(xi, wij[:, index])
        p = sigmoid(alpha, s)
        state = probability(p, RAND_MAX)
        xi = value_writing(index, state, xi)
        count.search(xi)

    print("updating = " + str(updating))
    for element in count.count:
        if element.label[0] == "0":
            continue
        else:
            print("[x1, x2, x3, x4] = " + element.label[1:N] + ": " + str(float(element.count) / float(updating)))
            # print(str(element.count/updating))
            # print(element.label[1:N])