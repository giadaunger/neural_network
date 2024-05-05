import numpy as np

class Layer:
    def __init__(self, amount_of_neurons, input_size):
        self.weights = 2 * np.random.rand(input_size, amount_of_neurons) - 1
        self.bias = 5

    def layer_forward(self, inputs):
        result = np.dot(inputs, self.weights) + self.bias
        return result

l1 = Layer(2, 5)
inputs = np.array([0.5, -0.3, 0.8, 0.1, 1])  
outputs = l1.layer_forward(inputs)
print(outputs)
