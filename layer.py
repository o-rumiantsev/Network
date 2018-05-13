import random
import numpy
import math

def logistic(value):
    return 1 / (1 + math.exp(value * (-1)))

def update(weights, sigmas, inputs, coef):
    weights += numpy.dot(sigmas, [inputs]) * coef

def countHiddenLayerErrors(nextLayer, size):
    sigmas = nextLayer.sigmas
    errors = []
    weights = numpy.transpose(nextLayer.weights)

    for k in range(size):
        unitWeights = weights[k]
        err = numpy.dot(unitWeights, sigmas)
        errors.append(err)

    return errors


def generateWeights(layerSize, prevLayerSize):
    weights = []

    for i in range(layerSize):
        unitWeights = []

        for k in range(prevLayerSize):
            unitWeight = random.uniform(-1, 1)
            unitWeights.append(unitWeight)

        weights.append(unitWeights)

    return numpy.array(weights)

class InputLayer():
    def __init__(self, layerSize):
        self.nextLayer = None
        self.prevLayer = None
        self.activation = logistic
        self.size = layerSize

    def activate(self, inputs):
        self.outputs = list(map(self.activation, inputs))
        return self.nextLayer.activate(self.outputs)

    def add(self, layer):
        self.nextLayer = layer



class Layer():
    def __init__(self, layerSize, prevLayer):
        self.nextLayer = None
        self.prevLayer = prevLayer
        self.activation = logistic
        self.size = layerSize
        self.weights = generateWeights(layerSize, prevLayer.size)

    def activate(self, inputs):
        inputsCol = numpy.array([[value] for value in inputs])
        self.inputs = inputs
        dot = numpy.dot(self.weights, inputsCol)
        self.outputs = list(map(self.activation, dot))

        if self.nextLayer: return self.nextLayer.activate(self.outputs)
        else: return self.outputs

    def add(self, layer):
        self.nextLayer = layer

    def updateWeights(self, requiredOutput, coef):
        if not self.nextLayer:
            requiredOutput = numpy.array([[val] for val in requiredOutput])
            output = numpy.array([[val] for val in self.outputs])
            errors = requiredOutput - output
            self.sigmas = list(map(lambda x: x * (1 - x), self.outputs))

            for i in range(self.size):
                self.sigmas[i] = [self.sigmas[i] * errors[i][0]]

            update(self.weights, self.sigmas, self.inputs, coef)

            if self.prevLayer.prevLayer:
                self.prevLayer.updateWeights(requiredOutput, coef)

        else:
            errors = countHiddenLayerErrors(self.nextLayer, self.size)
            self.sigmas = list(map(lambda x: x * (1 - x), self.outputs))

            for i in range(self.size):
                self.sigmas[i] = [self.sigmas[i] * errors[i][0]]

            update(self.weights, self.sigmas, self.inputs, coef)

            if self.prevLayer.prevLayer:
                    self.prevLayer.updateWeights(requiredOutput, coef)
