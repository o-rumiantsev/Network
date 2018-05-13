from model import Model

model = Model(0.9)

model.add(2)
model.add(10)
model.add(1)


print(model.run([0, 1]))
print(model.run([1, 1]))

trainSet = [[[0, 1], [1]], [[1, 0], [1]], [[0, 0], [0]], [[1, 1], [0]]]

model.train(trainSet, 10000)

print()
print(model.run([0, 1]))
print(model.run([1, 1]))

print(model.run([1, 0]))
print(model.run([0, 0]))
