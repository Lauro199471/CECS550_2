from nueralNetwork import NeuralNetwork
from layer import Layer

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

cat_images = np.random.randn(700, 2) + np.array([0, -3])

mouse_images = np.random.randn(700, 2) + np.array([3, 3])
dog_images = np.random.randn(700, 2) + np.array([-3, 3])

feature_set = np.vstack([cat_images, mouse_images, dog_images])
labels = np.array([0] * 700 + [1] * 700 + [2] * 700)

one_hot_labels = np.zeros((2100, 3))
for i in range(2100):
    one_hot_labels[i, labels[i]] = 1

# Prediction
# print("target: ", 2)
# print("X: ", feature_set[1500])
# print("Prediction: ", nn.predict(feature_set[1500]))

# x = [1, 4, 5]
# x = np.reshape(x, (1, -1))
# one_hot_labels = [0.1, 0.05]

nntest = NeuralNetwork()
nntest.add_layer(Layer(4, activation='sigmoid'))  # , w=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], b=[0.5, 0.5]))
nntest.add_layer(Layer(4, activation='sigmoid'))  # , w=[0.7, 0.8, 0.9, 0.1], b=[0.5, 0.5]))
nntest.add_layer(Layer(4, activation='sigmoid'))  # , w=[0.7, 0.8, 0.9, 0.1, 0.2, 0.3], b=[0.5, 0.5, 0.5]))
nntest.add_layer(Layer(3, activation='sigmoid'))  # , w=[0.7, 0.8, 0.9, 0.1, 0.2, 0.3], b=[0.5, 0.5]))
nntest.add_layer(Layer(3, activation='softmax'))  # , w=[0.7, 0.8, 0.9, 0.1, 0.2, 0.3], b=[0.5, 0.5]))

# nntest.backpropagation(x, one_hot_labels)

errors = nntest.train(feature_set, one_hot_labels, 0.1, 10000)

plt.plot(errors)
plt.title('Changes in MSE')
plt.xlabel('Epoch (every 10th)')
plt.ylabel('MSE')
plt.show()

print("\ngood")
