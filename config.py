class ResNet:

  """
    Stores ResNet18 Parameters
  """

  def __init__(self, filters, ksize, strides, pool, blocks, outputsize, inputsize):
    self.filters = filters
    self.ksize = ksize
    self.strides = strides
    self.pool = pool
    self.blocks = blocks
    self.outputsize = outputsize
    self.inputsize = inputsize


class CNN:
  
  """
    Stores CNN Parameters
  """

  def __init__(self, kernels, ksize, pool, dropout, outputsize, inputsize):
    self.kernels = kernels
    self.ksize = ksize
    self.pool = pool
    self.dropout = dropout
    self.outputsize = outputsize
    self.inputsize = inputsize