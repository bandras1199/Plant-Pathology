import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization,\
                                    Add, AveragePooling2D, GlobalAveragePooling2D,\
                                    Flatten, Dense
from tensorflow.keras.models import Model


def relu_bn(inputs):
	relu = ReLU()(inputs)
	bn = BatchNormalization()(relu)

	return bn


def residual_block(x, downsample, filters):
    y = Conv2D(kernel_size=(3, 3),
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=(3, 3),
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=(1, 1),
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out


def build_ResNet(r):

	"""
		Builds a ResNet18 with given parameters (r)
		sources:
			- https://github.com/calmisential/TensorFlow2.0_ResNet
			- https://towardsdatascience.com/building-a-resnet-in-keras-e8f1322a49ba
	
	"""

	inputs = Input(shape=r.inputsize)
	t = BatchNormalization()(inputs)
	t = Conv2D(	filters = r.filters[0], 
				 kernel_size = r.ksize, 
				 strides= r.strides, padding="same")(t)
	t = relu_bn(t)

	for i in range(len(r.blocks)):
		block = r.blocks[i]
		for j in range(block):
			t = residual_block(t, downsample=(j==0 and i!=0), filters=r.filters[i])

	t = AveragePooling2D(4)(t)
	t = Flatten()(t)

	outputs = Dense(units = r.outputsize, activation = "softmax")(t)
	model = Model(inputs, outputs)
	model.compile(optimizer='adam',
				loss='CategoricalCrossentropy',
				metrics=['accuracy'])
	return model