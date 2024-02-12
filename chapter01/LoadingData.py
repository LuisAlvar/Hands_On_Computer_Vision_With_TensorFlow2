import numpy as np
import mnist

np.random.seed(42)

# Loading the training and testing data:
xTrain, yTrain = mnist.train_images(), mnist.train_labels()
xTest, yTest = mnist.test_images(), mnist.test_labels()
numClasses = 10 # classes are the digits from 0 to 9
