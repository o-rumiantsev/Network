from mnist import MNIST
from model import Model

mndata = MNIST('./handwritten')

images, lables = mndata.load_training()
test_imgs, test_lbls = mndata.load_testing()

images = list(map(lambda x: list(map(lambda y: y / 255, x)), images))
test_imgs = list(map(lambda x: list(map(lambda y: y / 255, x)), test_imgs))

model = Model(0.1)
model.add(784)
model.add(500)
model.add(10)

model.loadConfig('./config/handwritten.json')
# trainSet = []
#
# for i in range(1000):
#     check = [0] * 10
#     check[lables[i]] = 1
#     trainSet.append([images[i], check])
#
# model.train(trainSet, 100)

results = []

for i in range(len(test_imgs)):
    print('Testing...' + str(i))
    res = model.run(test_imgs[i])
    pred = max(res)
    number = res.index(pred)
    print(str(number) + " " + str(test_lbls[i]))
    if number == test_lbls[i]:
        results.append(1)
    else:
        results.append(0)

correct = list(filter(lambda x: x == 1, results))
diff = len(results) - len(correct)
coef = diff / len(results)
print("Final Accuracy " + str((1 - coef) * 100) + "%")
