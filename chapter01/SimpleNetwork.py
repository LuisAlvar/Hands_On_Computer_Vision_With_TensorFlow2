import numpy as np 
from FullyConnectedLayer import FullyConnectedLayer

# Apply the sigmoid function to the elements of x
def sigmoid(x): 
  return 1 / (1 + np.exp(-x) ) # y

def derivated_sigmoid(y): # sigmoid derivative function
  return y * (1-y)

def loss_L2(pred, target): # L2 loss function 
  # opt. for results not depending on the batch size (pred.shape[0]), we divide the loss by it.a                xccccc                              Q211112
  return np.sum(np.square(pred - target)) / pred.shape[0]

def derivated_loss_L2(pred, target): # L2 derivative function
  # we could add the batch size division here too, but it wouldnt really affect the training 
  # (just scaling down the derviatves)
  return 2 * (pred - target)

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
  def __init__(self,
                numInputs,
                numOutputs,
                hiddenLayersSizes=(64,32),
                lossFn=loss_L2,
                dLossFn=derivated_loss_L2):
    super().__init__()
    # We build the list of layers composing the network:
    sizes = [numInputs, *hiddenLayersSizes, numOutputs]
    self.layers = [
      FullyConnectedLayer(sizes[i], sizes[i+1], sigmoid)
      for i in range(len(sizes) - 1)
    ]
    self.lossFn = lossFn
    self.dLossFn = dLossFn

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
  
  def backward(self, dLdY):
    # Back-propagate the loss derivative from last to 1sy layer
    for layer in reversed(self.layers):
      dLdY = layer.backward(dLdY)
    return dLdY

  def optimize(self, epsilon):
    # Optimize the parameters according to the stored gradients
    for layer in self.layers:
      layer.optimize(epsilon)
  
  def train(self, xTrain, yTrain, xVal, yVal, batchSize=32, numEpochs=5, learningRate=5e-3):
    # Train (and evaluate) the network on the provided dataset
    numBatchesPerEpoch = len(xTrain) # batch_size
    loss, accuracy = [], []
    for i in range(numEpochs): # for each training epoch
      epochLoss=0
      for b in range(numBatchesPerEpoch): # for each batch
        # Get batch:
        bIdx = b * batchSize
        bIdxE = bIdx + batchSize
        x,yTrue = xTrain[bIdx:bIdxE], yTrain[bIdx, bIdxE]
        # Optimize on batch:
        y = self.forward(x) # forward pass
        epochLoss += self.lossFn(y, yTrue) # loss
        dLdY = self.dLossFn(y, yTrue) # loss derivation 
        self.backward(dLdY) # back-propagation pass
        self.optimize(learningRate) # optimization
      loss.append(epochLoss / numBatchesPerEpoch)
      # After each epoch, we "validate" our network, i.e., we measure its accurcy over the test/validation set: 
      accuracy.append(self.evaluateAccuracy(xVal, yVal))
      print("Epoch {:4d} training loss = {:.6f} | val accuracy = {:.2f}%".format(i, loss[i], accuracy[i] * 100))



if __name__ == '__main__':
  losses, accuracies = mnist_classifier.train(xTrain, yTrain, xTest, yTest, batchSize=30, numEpochs=500)