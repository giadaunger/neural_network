import math

def Neuron(x_arr) -> float:
    weights = [42, 31, -10, 5.2]
    z = 0
    bias = 5

    for i in range(len(x_arr)):
        z += x_arr[i] * weights[i]
        z += bias
    y = 1 / (1 + math.exp(-z))
    return y
    
neuron1 = Neuron([0.5, -0.3, 0.8])
print("Neuron output:", neuron1)