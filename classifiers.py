import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization,\
                                    Add, MaxPooling2D, GlobalAveragePooling2D,\
                                    Flatten, Dense, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
import numpy as np
import random
import gc

def ArtificialData(data, size):

	"""Creates artificial data for debug purposes

	Args:
		data (np.ndarray): Sample to copy the shape of original dataset
		size (int): Sample size of artificial dataset

	Returns:
		d_x (np.ndarray): Artificial dataset
		d_y (np.ndarray): Labels for the dataset (one-hot)

	"""

	d_x = np.zeros(([size] + list(data.shape[1:])), dtype = 'uint8')
	d_y = np.zeros((size), dtype = 'uint8')

	num_classes = 4
	stepsize = int(d_x.shape[0] / num_classes)

	for cl in range(num_classes):
		mu = cl*2
		sigma = 1
		d_x[cl*stepsize:(cl+1)*stepsize,:,:,:] = np.random.normal(mu,sigma,[stepsize] + list(d_x.shape[1:]))
		d_y[cl*stepsize:(cl+1)*stepsize] = np.tile(cl, stepsize)

	randorder = np.arange(d_x.shape[0])
	random.shuffle(randorder)
	d_x = d_x[randorder]
	d_y = d_y[randorder]
	d_y = np.eye(num_classes)[d_y]

	return d_x, d_y
 
 
class garbage_cb(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs = None):
		gc.collect()


def basic_block(X, f, strides, flag_first_entry):

    """Basic 3x3 convolutional block with shortcut for ResNet18

		Args: 
			X (tensor array): Input of the convolutional block
			f (int): Size of filters used in both convolution
			strides (tuple): Size of stride used in the first convolution of the block
			flag_first_entry (bool): Indicates if it is the first block of first layer

		Returns:	
			X (tensor array): Output of the convolutional block
	"""


    X_shortcut = X
    
    if not flag_first_entry:
        X = BatchNormalization()(X)
        X = Activation("relu")(X)
         
    X = Conv2D(filters=f, 
                kernel_size=(3, 3),
                strides=strides, # (2,2) in first blocks
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(X)

    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Conv2D(filters=f, 
                kernel_size=(3, 3),
                strides=(1,1), # (1,1) in every other
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(X)

    if K.int_shape(X_shortcut)[3] != K.int_shape(X)[3]:         
        X_shortcut = Conv2D(filters=K.int_shape(X)[3],
                          kernel_size=(1, 1),
                          strides=(2,2),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(X_shortcut)

    X = Add()([X_shortcut, X])
    return X

def ResNet18t(input_shape,output_num):
    
    """ Builds a ResNet18 model

		Args: 
			input_shape (np.ndarray): Shape of input image
			output_num (int): Number of neurons in the final FC layer

		Returns:	
			Keras model
    """
    X_input = Input(shape=input_shape)
    X = Conv2D(  filters=64, 
                    kernel_size=(7,7),
                    strides=(2,2), 
                    padding="same",
                    kernel_initializer="he_normal",
                    kernel_regularizer=l2(1.e-4))(X_input)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(X)

    filters = 64
    num_blocks = [2, 2, 2, 2]
    for i,val in enumerate(num_blocks): # cycle over layers
        for j in range(val): # cycle over blocks in layers
            flag_first_layer = (i == 0)
            flag_first_block = (j == 0)
            flag_first_entry = flag_first_layer and flag_first_block
            if flag_first_block and not flag_first_layer:
                strides = (2,2)
            else:
                strides = (1,1)               
            X = basic_block(X, filters, strides, flag_first_entry)    
        filters *= 2

    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = GlobalAveragePooling2D()(X)
    X = Flatten()(X)
    X_out = Dense(units=output_num, kernel_initializer="he_normal",
                  activation="softmax")(X)
    model = Model(inputs = X_input, outputs = X_out)
    return model