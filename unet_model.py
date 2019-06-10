import tensorflow as tf
from tensorflow import keras
import numpy as np


# internal function to build one block of 2 cascade conv2d
def conv_block(x, k_size, s_size, channels):
	''' input:	x --- input tensor of shape=(B,H,W,C)
				k_size --- scalar, kernel size
				s_size --- scalar, stride size
				channels --- scalar, output channels
		output:	features --- output feature map '''
	# 1st conv
	features = keras.layers.Conv2D(filters=channels,
								   kernel_size=k_size,
								   strides=s_size,
								   padding='SAME',
								   activation='relu',
								   kernel_initializer='he_normal')(x)
	# 2nd conv
	features = keras.layers.Conv2D(filters=channels,
								   kernel_size=k_size,
								   strides=s_size,
								   padding='SAME',
								   activation='relu',
								   kernel_initializer='he_normal')(features)
	# over
	return features

# internal function to concatenate skip connection
def concat_channel(x, y):
	''' input:	x --- base tensor, shape=(B,H,W,Cx), dtype=tf.float32
				y --- tensor to be appended, shape=(B,H,W,Cy), dtype=tf.float32
		output:	features --- concatenated tensor, shape=(B,H,W,Cx+Cy), dtype=tf.float32 '''
	features = keras.layers.Concatenate(axis=-1)([x,y])
	# over
	return features

class UNet(object):
	# constructor
	def __init__(self, input_img, num_classes):
		''' input:	input_img --- input tensor, format=BHWC, dtype=tf.float32
					num_classes --- num of classes '''
		self.X = input_img
		self.logits_before_softmax = None
		self.NUM_CLASSES = num_classes
		self.create()

	# build the model
	def create(self):
		# 1st encoder
		features1 = conv_block(self.X, k_size=3, s_size=1, channels=64)
		shape1 = tf.shape(features1)
		# 1st max pool
		pool1 = keras.layers.MaxPool2D(pool_size=2, strides=2, padding='SAME')(features1)

		# 2nd encoder
		features2 = conv_block(pool1, k_size=3, s_size=1, channels=128)
		shape2 = tf.shape(features2)
		# 2nd max pool
		pool2 = keras.layers.MaxPool2D(pool_size=2, strides=2, padding='SAME')(features2)

		# 3rd encoder
		features3 = conv_block(pool2, k_size=3, s_size=1, channels=256)
		shape3 = tf.shape(features3)
		# 3rd max pool
		pool3 = keras.layers.MaxPool2D(pool_size=2, strides=2, padding='SAME')(features3)

		# 4th encoder
		features4 = conv_block(pool3, k_size=3, s_size=1, channels=512)
		shape4 = tf.shape(features4)
		# 4th max pool
		pool4 = keras.layers.MaxPool2D(pool_size=2, strides=2, padding='SAME')(features4)

		# 5th encoder
		features5 = conv_block(pool4, k_size=3, s_size=1, channels=1024)

		# 1st deconv
		deconv1 = keras.layers.Conv2DTranspose(filters=512,
											   kernel_size=2,
											   strides=2,
											   padding='SAME')(features5)
		# if input dim is even, the deconvolution will exactly recover the input dim
		# else, the deconvolution will output a dim+1 larger result
		deconv1 = deconv1[:][0:shape4[1]][0:shape4[2]][:]
		concat1 = concat_channel(features4, deconv1)
		# 6th encoder
		features6 = conv_block(concat1, k_size=3, s_size=1, channels=512)

		# 2nd deconv
		deconv2 = keras.layers.Conv2DTranspose(filters=256,
											   kernel_size=2,
											   strides=2,
											   padding='SAME')(features6)
		deconv2 = deconv1[:][0:shape3[1]][0:shape3[2]][:]
		concat2 = concat_channel(features3, deconv2)
		# 7th encoder
		features7 = conv_block(concat2, k_size=3, s_size=1, channels=256)

		# 3rd deconv
		deconv3 = keras.layers.Conv2DTranspose(filters=128,
											   kernel_size=2,
											   strides=2,
											   padding='SAME')(features7)
		deconv3 = deconv1[:][0:shape2[1]][0:shape2[2]][:]
		concat3 = concat_channel(features2, deconv3)
		# 8th encoder
		features8 = conv_block(concat3, k_size=3, s_size=1, channels=128)

		# 4th deconv
		deconv4 = keras.layers.Conv2DTranspose(filters=64,
											   kernel_size=2,
											   strides=2,
											   padding='SAME')(features8)
		deconv4 = deconv1[:][0:shape1[1]][0:shape1[2]][:]
		concat4 = concat_channel(features1, deconv4)
		# 9th encoder
		features9 = conv_block(concat4, k_size=3, s_size=1, channels=64)

		# output logits before softmax
		self.logits_before_softmax = features9


# compute loss function (cross entropy in all pixels)
def loss_func(labels, logits_before_softmax):
	''' input:	labels --- sparse labels with shape [batch, H, W], dtype = tf.uint8
				logits_before_softmax --- logits before softmax with shape [batch, H, W, NUM_CLASSES], dtype = tf.float
		output:	loss --- scalar cross entropy, dtype = tf.float '''
	batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.to_int32(labels),
												  logits=logits_before_softmax,
												  name='loss')
	# reduce the batch loss to a mean scalar
	loss = tf.reduce_mean(batch_loss)
	# over
	return loss
