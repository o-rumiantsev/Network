from model import Model

model = Model(0.9)

model.add(5)
model.add(10)
model.add(2)


trainSet = []

trainSet.append([[0, 0.1, 0.2, 0.3, 0.4], [1, 0]])
trainSet.append([[0, 0.2, 0.4, 0.6, 0.8], [1, 0]])
trainSet.append([[0.5, 0.2, 0.1, 0.01, 0.005], [0, 1]])
trainSet.append([[0.1, 0.01, 0.005, 0.001, 0.0005], [0, 1]])
trainSet.append([[0.1, 0.7, 0.6, 0.4, 0.3], [0, 1]])

model.train(trainSet, 500)

test = [[0, 1, 5, 6, 7], [1, 2, 3, 4, 5], [5, 1, 0.1, 0.05, 0.01], [1, 7, 6, 4, 3], [1, 4, 3, 7, 6]]

for i in range(len(test)):
    testlist = list(map(lambda x: x / max(test[i]), test[i]))
    res = model.run(testlist)
    diff = res[0] - res[1]
    if diff > 0:
        print('Asceding ' + str(test[i]) + ' ' + str(int(res[0] * 100)) + '%')
    else:
        print('Desceding ' + str(test[i]) + ' ' + str(int(res[1] * 100)) + '%')
