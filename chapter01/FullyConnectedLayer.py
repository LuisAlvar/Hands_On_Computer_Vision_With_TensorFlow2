import numpy as np

class FullyConnectedLayer(object):
  """
  A simply fully-connected NN layer
  Args:
    numInputs (int): The input vector size/number of input values.
    layerSize (int): The output vector size/number of neurons
    activationFn (callable): the activaiton function for this layer.
  Attributes:
    W (ndarray): The weight values for each input
    b (ndarray): The bias value, added to the weighted sum
    size (int): The layer size/number of neurons
    activationFn (callable): The neurons' activation function 
  """
  def __init__(self, numInputs, layerSize, activationFn):
    super().__init__()
    # Randomly initializaing the weight vector and bias value:
    self.W = np.random.standard_normal((numInputs, layerSize))
    self.b = np.random.rand(layerSize)
    self.size = layerSize
    self.activationFn = activationFn

  def forward(self, x):
    # Forward the input signal through the neuron.
    z = np.dot(x, self.W) + self.b
    return self.activationFn(z)

# random seed generator
np.random.seed(42)

# random input column-vectors of 2 values 
x1 = np.random.uniform(-1, 1, 2).reshape(1, 2);
x2 = np.random.uniform(-1, 1, 2).reshape(1, 2);

print("input vector ---> ",  end="")
print(x1)
print("input vector ---> ",  end="")
print(x2)

# declare the step function as the activation function 
reluFn = lambda y: np.maximum(y, 0)
layer = FullyConnectedLayer(2, 3, reluFn)

# our layer can process x1 and x2 separately...
out1 = layer.forward(x1)
out2 = layer.forward(x2)

print("y = f(z1) = ",  end="")
print(out1)
print("y = f(z2) = ",  end="")
print(out2)

x12 = np.concatenate((x1, x2)) # stack of input vectors, of shape
out12 = layer.forward(x12)

"python.exe" "c:/Workspace/tensorflow/chapter01/FullyConnectedLayer.py"