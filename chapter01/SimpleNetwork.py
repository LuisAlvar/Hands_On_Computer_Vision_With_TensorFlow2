import numpy as np 
from FullyConnectedLayer import FullyConnectedLayer

# Apply the sigmoid function to the elements of x
def sigmoid(x): 
  return 1 / (1 + np.exp(-x) ) # y


class SimpleNetwork(object):
  """
  A simple fully-connected NN.
  Args:
    numInputs (int): The input vector size / number of input values.
    numOutputs (int): The output vector size
    hiddenLayersSizes (list): A list of sizes for each hidden layer to be added to the network
  Attributes:
    layers (list): The list of layers forming this simple network
  """
  def __init__(self, numInputs, numOutputs, hiddenLayersSizes=(64,32)):
    super().__init__()
    # We build the list of layers composing the network:
    sizes = [numInputs, *hiddenLayersSizes, numOutputs]
    self.layers = [
      FullyConnectedLayer(sizes[i], sizes[i+1], sigmoid)
      for i in range(len(sizes) - 1)
    ]

  def forward(self, x):
    # Forward the input vector `x` through the layers
    for layer in self.layers: # from the input layer to the output one 
      x = layer.forward(x)
    return x

  def predict(self, x):
    # Compute the output corresponding to `x` , and return the index of the largest output value
    estimations  = self.forward(x)
    best_class = np.argmax(estimations)
    return best_class
  
  def evaluateAccuracy(self, xVal, yVal):
    # Evaluate the network's accuracy on a validation dataset.
    numCorrects = 0
    for i in range(len(xVal)):
      if self.predict(xVal[i]) == yVal[i]:
        numCorrects += 1
    return numCorrects / len(xVal)
