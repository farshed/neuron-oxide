import math
import json
import random

with open('data.json') as file:
  data = json.load(file)

class NeuralNetwork:
  def __init__(self):
    self.weights = [random.random(), random.random()]
    self.bias = random.random()
    self.learning_rate = 0.1

  def predict(self, inp):
    sum_ = sum(inp[i] * weight for i, weight in enumerate(self.weights)) + self.bias
    return sigmoid(sum_)
  
  def train(self, inputs, outputs, epochs):
    for _ in range(epochs):
      for j, inp in enumerate(inputs):
        output = self.predict(inp)

        error = outputs[j] - output
        delta = derivative(output)

        for k in range(len(self.weights)):
          self.weights[k] += self.learning_rate * error * inp[k] * delta

        self.bias += self.learning_rate * error * delta


def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def derivative(x):
  return x * (1-x)


# Train and test

inputs = data['training_inputs']
outputs = data['training_outputs']
test_inputs = data['test_inputs']

neural_net = NeuralNetwork()
neural_net.train(inputs, outputs, 10000)


for inp in test_inputs:
	prediction = neural_net.predict(inp);
	print(f'Input: {inp}, Prediction: {"{:.1f}".format(prediction)}')

