import numpy as np

class Neuron:
    def __init__(self, num_inputs):
        self.weights = np.random.randn(num_inputs) 
        self.bias = np.random.randn(1) 
        
    def activate(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        activation = self.activation_function(weighted_sum)
        return activation

    def activation_function(self, x):
        return 1 / (1 + np.exp(-x))

 
# Exampe usage
num_inputs = 3
neuron = Neuron(num_inputs)

inputs = np.array([0.5, -0.3, 0.8])
output = neuron.activate(inputs)
print("Neuron output:", output)