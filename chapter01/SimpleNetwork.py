import numpy as np 
from layer import FullyConnectedLayer

# Apply the sigmoid function to the elements of x
def sigmoid(x): 
  return 1 / (1 + np.exp(-x)) # y


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
    