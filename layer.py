import random
import numpy
import math

def logistic(value):
    return 1 / (1 + math.exp(value * (-1)))

def update(weights, sigma, inputs, coef):
    for i in range(len(weights)):
        weights[i] += coef * sigma * inputs[i]

def countHiddenLayerError(layer, i):
    sigmas = [neuron['sigma'] for neuron in layer.neurons]
    weights = []

    for k in range(len(layer.neurons)):
        neuronWeights = layer.neurons[k]['weights']
        weights.append([neuronWeights[i]])

    return numpy.dot(sigmas, weights)


def generateWeights(prevLayerSize):
    weights = []

    for i in range(prevLayerSize):
        weight = random.uniform(-1, 1)
        weights.append(weight)

    return numpy.array(weights)

class InputLayer():
    def __init__(self, layerSize):
        self.nextLayer = None
        self.prevLayer = None
        self.neurons = []

        for i in range(layerSize):
            neuron = {}
            neuron['activation'] = logistic
            self.neurons.append(neuron)




    def size(self):
        return len(self.neurons)




    def activate(self, inputs):
        self.outputs = []

        for i in range(self.size()):
            fn = self.neurons[i]['activation']
            result = fn(inputs[i])
            self.outputs.append(result)

        return self.nextLayer.activate(self.outputs)




    def add(self, layer):
        self.nextLayer = layer



class Layer():
    def __init__(self, layerSize, prevLayer):
        self.nextLayer = None
        self.prevLayer = prevLayer
        self.neurons = []

        for i in range(layerSize):
            neuron = {}
            neuron['weights'] = generateWeights(prevLayer.size())
            neuron['activation'] = logistic
            self.neurons.append(neuron)





    def size(self):
        return len(self.neurons)




    def activate(self, inputs):
        inputs_col = [[value] for value in inputs]
        inputs_vec = numpy.array(inputs_col)
        self.inputs = inputs
        self.outputs = []

        for i in range(self.size()):
            weights = self.neurons[i]['weights']
            dot = numpy.dot(weights, inputs_vec)
            fn = self.neurons[i]['activation']
            result = fn(dot)
            self.outputs.append(result)

        if self.nextLayer: return self.nextLayer.activate(self.outputs)
        else: return self.outputs




    def add(self, layer):
        self.nextLayer = layer



    def updateWeights(self, requiredOutput, coef):
        if not self.nextLayer:
            for i in range(len(self.neurons)):
                error = requiredOutput[i] - self.outputs[i]
                sigma = self.outputs[i] * (1 - self.outputs[i]) * error
                self.neurons[i]['sigma'] = sigma
                weights = self.neurons[i]['weights']
                update(weights, sigma, self.inputs, coef)

            self.prevLayer.updateWeights(requiredOutput, coef)

        else:
            for i in range(len(self.neurons)):
                error = countHiddenLayerError(self.nextLayer, i)
                sigma = self.outputs[i] * (1 - self.outputs[i]) * error
                self.neurons[i]['sigma'] = sigma
                inputs = self.inputs
                weights = self.neurons[i]['weights']
                update(weights, sigma, inputs, coef)

            if self.prevLayer.prevLayer:
                    self.prevLayer.updateWeights(requiredOutput, coef)
