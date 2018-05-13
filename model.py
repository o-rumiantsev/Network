from layer import *

class Model():
    def __init__(self, coef):
        self.coef = coef
        self.inputLayer = None
        self.output = None

    def add(self, layerSize):
        if not self.inputLayer:
            self.inputLayer = InputLayer(layerSize)
            self.output = self.inputLayer
        else:
            newLayer = Layer(layerSize, self.output)
            self.output.add(newLayer)
            self.output = newLayer

    def run(self, inputs):
        return self.inputLayer.activate(inputs)

    def train(self, trainSet, maxEpoch):
        for i in range(maxEpoch):
            for k in range(len(trainSet)):
                res = self.run(trainSet[k][0])
                self.output.updateWeights(trainSet[k][1], self.coef)
