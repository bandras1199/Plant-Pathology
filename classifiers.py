import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization,\
                                    Add, MaxPooling2D, AveragePooling2D,\
                                    Flatten, Dense, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
import gc

class garbage_cb(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs = None):
    gc.collect()

def var_init():
  global kernel_size
  global strides
  global kernel_initializer
  global padding
  global kernel_regularizer

  kernel_size = (3,3)  #shortcutba (1,1)
  strides = (1, 1)
  kernel_initializer = "he_normal"
  padding = "same"
  kernel_regularizer = l2(1.e-2)

def basic_block(X, filters, first_layer_flag):

  """Basic 3x3 convolutional block with shortcut for ResNet18

    Args: 
			    X (tensor array): Input of the convolutional block
          filter (list): Size of filters used in the block
          first_layer_flag (bool): Indicates if the filter size changed or not
          
		Returns:	
			    X (tensor array): Output of the convolutional block
  """

  X_shortcut = X

  X= Conv2D(filters=filters[0], 
          kernel_size=kernel_size,
          strides=(strides if not first_layer_flag else 2),
          padding=padding,
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer)(X)          
  X = BatchNormalization(axis=3)(X)
  X = Activation("relu")(X)

  X = Conv2D(filters=filters[1], 
          kernel_size=kernel_size,
          strides=strides,
          padding=padding,
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer)(X)
  X = BatchNormalization(axis=3)(X)

  if first_layer_flag:
    # needs 1x1 conv in shortcut to match sizes
    X_shortcut = Conv2D(filters=K.int_shape(X)[3],
                          kernel_size=(1, 1),
                          strides=(2,2),
                          padding="same",
                          kernel_initializer="he_normal",
                          kernel_regularizer=kernel_regularizer)(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)
  
  X = Add()([X_shortcut, X])
  X = Activation("relu")(X)

  return X


def ResNet18t(input_shape,output_num):

  """ Builds a ResNet18 model

    Args: 
			    input_shape (np.ndarray): Shape of input image
          output_num (int): Number of neurons in the final FC layer

		Returns:	
			    Keras model
  """
  X_input = Input(input_shape)
  var_init()

  X = Conv2D(filters = 64, 
             kernel_size = (7, 7), 
             strides = (2, 2), 
             padding=padding,
             kernel_initializer=kernel_initializer,
             kernel_regularizer=kernel_regularizer)(X_input)
  X = BatchNormalization(axis = 3)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D(pool_size = (3, 3), 
                   strides=(2, 2),
                   padding='same')(X)

  f = 64 # size of first filter
  num_blocks = [2, 2, 2, 2] # number of conv block in each layer
  for i in num_blocks:
    for j in range(i):
      X = basic_block(X, [f,f], first_layer_flag= (j == 0 and i!=0))
    f *= 2
    
  X = AveragePooling2D(pool_size=(2, 2))(X)
  X = Flatten()(X)
  X = Dense(units=output_num,
            activation="softmax")(X)
  model = Model(inputs = X_input, outputs = X)

  return model