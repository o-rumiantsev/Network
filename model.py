from layer import *
import json

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

    def saveConfig(self, path):
        config = {}
        layers = []
        layer = self.inputLayer
        position = 0

        while layer:
            configLayer = {}
            configLayer['layer'] = position
            configLayer['activation'] = layer.activationType
            configLayer['size'] = layer.size

            if layer.prevLayer:
                listWeights = []
                for i in range(layer.size):
                    listWeights.append(list(layer.weights[i]))

                configLayer['weights'] = listWeights

            position += 1
            layers.append(configLayer)
            layer = layer.nextLayer

        config['layers'] = layers
        config['coef'] = self.coef

        configFile = open(path, 'w')
        configFile.write(json.dumps(config, indent = 4))

    def loadConfig(self, path):
        self.inputLayer = None
        self.output = None

        configFile = open(path, 'r')
        configJson = configFile.read()
        config = json.loads(configJson)
        configFile.close()

        self.coef = config['coef']
        layersConfig = config['layers']

        self.add(layersConfig[0]['size'])
        
        for i in range(1, len(layersConfig)):
            size = layersConfig[i]['size']
            self.add(size)
            weights = layersConfig[i]['weights']
            self.output.setWeights(weights)


    def run(self, inputs):
        return self.inputLayer.activate(inputs)


    def train(self, trainSet, maxEpoch):
        for i in range(maxEpoch):
            print(i)
            results = []

            coef = self.coef
            run = self.run
            updateWeights = self.output.updateWeights

            for k in range(len(trainSet)):
                res = self.run(trainSet[k][0])
                updateWeights(trainSet[k][1], coef)

                pred = max(res)
                number = res.index(pred)
                label = trainSet[k][1].index(1)
                if number == label:
                    results.append(1)
                else:
                    results.append(0)


            correct = list(filter(lambda x: x == 1, results))
            diff = len(results) - len(correct)
            coef = diff / len(results)
            print("Accuracy " + str((1 - coef) * 100) + "%")
