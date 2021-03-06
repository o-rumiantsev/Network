
from model import Model

model = Model(0.9)

model.add(2)
model.add(10)
model.add(2)


trainSet = [[[0, 1], [1, 0]], [[1, 0], [1, 0]], [[0, 0], [0, 1]], [[1, 1], [0, 1]]]
model.train(trainSet, 7000)

test = [[0, 1], [1, 0], [0, 0], [1, 1], [0.5, 0.5], [1, 0.5], [0.5, 1], [1, 0.2]]

for i in range(len(test)):
    res = model.run(test[i])
    diff = res[0] - res[1]
    if diff > 0:
        print('Return 1 ' + str(test[i]) + ' ' + str(int(res[0] * 100)) + '%')
    else:
        print('Return 0 ' + str(test[i]) + ' ' + str(int(res[1] * 100)) + '%')
