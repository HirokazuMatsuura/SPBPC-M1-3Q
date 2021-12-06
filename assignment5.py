import numpy as np

# ベクトルに直して掛け算
class EnergyRNN:
    def __init__(self) -> None:
        pass

    def set_value(self, neuron: np, weight: np):
        self.neuron = neuron
        self.weight = weight
        
    def calc(self):
        energy = 0
        for i, w_row in enumerate(self.weight):
            for j, w in enumerate(w_row):
                energy += (- 1/2) * w * self.neuron[i] * self.neuron[j]
        self.energy = energy
        return energy

class EnergyEQP:
    def __init__(self) -> None:
        pass

    def calc_weight(find_weight):
        find_weight["x"]
        find_weight["y"]



    def calu_energy(neuron):
        energy = 0

        temp_neuron = neuron
        temp_neuron = temp_neuron - 1
        temp_neuron = temp_neuron.sum(1)
        temp_neuron = np.dot(temp_neuron)
        temp_neuron = temp_neuron.sum(0)
        energy += temp_neuron[0, 0]

        temp_neuron = neuron
        temp_neuron = temp_neuron - 1
        temp_neuron = temp_neuron.sum(0)
        temp_neuron = np.dot(temp_neuron)
        temp_neuron = temp_neuron.sum(1)
        energy += temp_neuron[0, 0]

        



        temp_neuron = neuron.clone()

class Weight:
    def __init__(self) -> None:
        pass

    def to_vec(self, mat: np):
        return mat.reshape([-1, 1])

    def to_mat(self, vec: np):
        return vec.reshape([np.sqrt(len(vec)), np.sqrt(len(vec))])

        

