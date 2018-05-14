import random
import numpy
import math

def logistic(value):
    return 1 / (1 + math.exp(value * (-1)))


def countHiddenLayerErrors(nextLayer, size):
    sigmas = nextLayer.sigmas
    weights = numpy.transpose(nextLayer.weights)
    errors = numpy.dot(weights, sigmas)
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
        self.activationType = 'logistic'
        self.activation = logistic
        self.size = layerSize

    def activate(self, inputs):
        self.outputs = [self.activation(val) for val in inputs]
        return self.nextLayer.activate(self.outputs)

    def add(self, layer):
        self.nextLayer = layer

class Layer():
    def __init__(self, layerSize, prevLayer):
        self.nextLayer = None
        self.prevLayer = prevLayer
        self.activationType = 'logistic'
        self.activation = logistic
        self.size = layerSize
        self.weights = generateWeights(layerSize, prevLayer.size)

    def activate(self, inputs):
        inputsCol = numpy.transpose([inputs])
        self.inputs = [inputs]
        dot = numpy.dot(self.weights, inputsCol)
        self.outputs = [self.activation(row[0]) for row in dot]

        if self.nextLayer: return self.nextLayer.activate(self.outputs)
        else: return self.outputs

    def add(self, layer):
        self.nextLayer = layer

    def setWeights(self, weights):
        self.weights = numpy.array(weights)

    def updateWeights(self, requiredOutput, coef):
        if not self.nextLayer:
            requiredOutput = numpy.transpose([requiredOutput])
            errors = requiredOutput - numpy.transpose([self.outputs])
            constants = [val * (1 - val) for val in self.outputs]
            dot = numpy.dot(errors, [constants])
            diagonal = [numpy.diagonal(dot)]
            self.sigmas = numpy.transpose(diagonal)

            self.weights = self.weights + numpy.dot(self.sigmas, self.inputs) * coef

            if self.prevLayer.prevLayer:
                self.prevLayer.updateWeights(None, coef)

        else:
            errors = countHiddenLayerErrors(self.nextLayer, self.size)
            constants = [val * (1 - val) for val in self.outputs]
            dot = numpy.dot(errors, [constants])
            diagonal = [numpy.diagonal(dot)]
            self.sigmas = numpy.transpose(diagonal)

            self.weights = self.weights + numpy.dot(self.sigmas, self.inputs) * coef

            if self.prevLayer.prevLayer:
                    self.prevLayer.updateWeights(None, coef)
