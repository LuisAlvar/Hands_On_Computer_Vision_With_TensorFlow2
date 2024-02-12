import numpy as np 
from layer import FullyConnectedLayer

# Apply the sigmoid function to the elements of x
def sigmoid(x): 
  return 1 / (1 + np.exp(-x)) # y


class SimpleNetwork(object):
  """
  A simple fully-connected NN.
  Args:
    numInputs ()
    numOutputs ()
    hiddenLayersSizes ()
  Attributes:

  """