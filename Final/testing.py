from nueralNetwork import NeuralNetwork
from layer import Layer

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

cat_images = np.random.randn(700, 2) + np.array([0, -3])

mouse_images = np.random.randn(700, 2) + np.array([3, 3])
dog_images = np.random.randn(700, 2) + np.array([-3, 3])


labels = np.array([0] * 700 + [1] * 700 + [2] * 700)

one_hot_labels = np.zeros((2100, 3))
for i in range(2100):
    print(labels[i])
    one_hot_labels[i, labels[i]] = 1
    



