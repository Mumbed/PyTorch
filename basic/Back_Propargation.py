#2-layer NN model
import numpy as np

data = np.loadtxt("training.txt")
print(data.shape)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))



