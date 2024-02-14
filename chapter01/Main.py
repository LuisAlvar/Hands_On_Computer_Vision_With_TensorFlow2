from SimpleNetwork import SimpleNetwork
import numpy as np
import mnist 

if __name__ == '__main__':

  np.random.seed(42)

  # Loading the training and testing data:
  xTrain, yTrain = mnist.train_images(), mnist.train_labels()
  xTest, yTest = mnist.test_images(), mnist.test_labels()
  numClasses = 10 # classes are the digits from 0 to 9

  # We transform the images into column vectors (as inputs for our NN):
  xTrain, xTest = xTrain.reshape(-1, 28*28), xTest.reshape(-1, 28*28)

  # We "one-hot" the labels (as targets for our NN), for instance, transform label `4` into vector `[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]`
  yTrain =  np.eye(numClasses)[yTrain]

  # Network for MNIST images, with 2 hidden layers of size 64 and 32
  mnist_classifier = SimpleNetwork(xTrain.shape[1], numClasses, [64, 32])

  # ... and we evaluate its accuracy on the MNIST  test set:
  accracy = mnist_classifier.evaluateAccuracy(xTest, yTest)
  print("accuracy = {:.2f}%".format(accracy  * 100))