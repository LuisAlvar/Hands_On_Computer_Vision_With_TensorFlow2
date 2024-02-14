import numpy as np

class Neuron(object):
  """
  A simple feed-forward artificial neuron.
  Args:
    numInputs (int): The input vector size / number of input values.
    activationFn (callable): The activation function.
  Attributes:
    W (ndarray): The weight values for each input
    b (float): The bias value, added to the weighted sum
    activationFn (callable): the activation function
  """
  def __init__(self, numInputs, activationFn):
    super().__init__()
    # Randomly initializaing the weight vector and bias value:
    self.W = np.random.rand(numInputs)
    self.b = np.random.rand(1)
    self.activationFn = activationFn

  def forward(self, x):
    # Forward the input signal through the neuron.
    z = np.dot(x, self.W) + self.b
    return self.activationFn(z)
  


if __name__ == '__main__':

  # random seed generator
  np.random.seed(42)

  # input vector
  x = np.random.rand(3).reshape(1,3)
  print("input vector ---> ",  end="")
  print(x)

  # declare the step function as the activation function 
  step_fn = lambda y: 0 if y <= 0  else 1

  perceptron = Neuron(numInputs=x.size, activationFn=step_fn)
  out = perceptron.forward(x)

  print("weight vector ---> ", end="")
  print(perceptron.W)
  print("bias ---> ", end="")
  print(perceptron.b)

  print("y = f(z) = ", end="")
  print(out)

  # "python.exe" "c:/Workspace/tensorflow/chapter01/Neuron.py"