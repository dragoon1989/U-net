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
		self.NUM_CLASSES = num_classes
		self.create()

	# build the model
	def create(self):
		# 1st encoder
		features1 = conv_block(self.X, k_size=3, s_size=1, channels=64)
		# 1st max pool
		pool1 = keras.layers.MaxPool2D(pool_size=2, strides=2, padding='SAME')(features1)
		
		# 2nd encoder
		features2 = conv_block(pool1, k_size=3, s_size=1, channels=128)
		# 2nd max pool
		pool2 = keras.layers.MaxPool2D(pool_size=2, strides=2, padding='SAME')(features2)
		
		# 3rd encoder
		features3 = conv_block(pool2, k_size=3, s_size=1, channels=256)
		# 3rd max pool
		pool3 = keras.layers.MaxPool2D(pool_size=2, strides=2, padding='SAME')(features3)
		
		# 4th encoder
		features4 = conv_block(pool3, k_size=3, s_size=1, channels=512)
		# 4th max pool
		pool4 = keras.layers.MaxPool2D(pool_size=2, strides=2, padding='SAME')(features4)
		
		# 5th encoder
		features5 = conv_block(pool4, k_size=3, s_size=1, channels=1024)
		
		# 1st deconv
		deconv1 = 
		
		# 1st conv
		conv1 = keras.layers.Conv2D(filters=64,
									kernel_size=(7,7), strides=(1,1), padding='SAME',
									data_format='channels_last', activation=None)(self.X)
		# use bn
		bn = keras.layers.BatchNormalization(axis=3)(conv1)
		# activation
		features = keras.layers.Activation('relu')(bn)
		# 1st pool
		#features, mask1 = tf_unpool.max_pool_with_argmax(features, 2)
		features, mask1 = pooling.max_pool_with_argmax(net=features, ksize=[1,2,2,1], strides=[1,2,2,1])

		# 2nd conv
		conv2 = keras.layers.Conv2D(filters=64,
									kernel_size=(7,7), strides=(1,1), padding='SAME',
									data_format='channels_last', activation=None)(features)
		# use bn
		bn = keras.layers.BatchNormalization(axis=3)(conv2)
		# activation
		features = keras.layers.Activation('relu')(bn)
		# 2nd pool
		#features, mask2 = tf_unpool.max_pool_with_argmax(features, 2)
		features, mask2 = pooling.max_pool_with_argmax(net=features, ksize=[1,2,2,1], strides=[1,2,2,1])

		# 3rd conv
		conv3 = keras.layers.Conv2D(filters=64,
									kernel_size=(7,7), strides=(1,1), padding='SAME',
									data_format='channels_last', activation=None)(features)
		# use bn
		bn = keras.layers.BatchNormalization(axis=3)(conv3)
		# activation
		features = keras.layers.Activation('relu')(bn)
		# 3rd pool
		#features, mask3 = tf_unpool.max_pool_with_argmax(features, 2)
		features, mask3 = pooling.max_pool_with_argmax(net=features, ksize=[1,2,2,1], strides=[1,2,2,1])

		# 4th conv
		conv4 = keras.layers.Conv2D(filters=64,
									kernel_size=(7,7), strides=(1,1), padding='SAME',
									data_format='channels_last', activation=None)(features)
		# use bn
		bn = keras.layers.BatchNormalization(axis=3)(conv4)
		# activation
		features = keras.layers.Activation('relu')(bn)
		# 4th pool
		#features, mask4 = tf_unpool.max_pool_with_argmax(features, 2)
		#features, mask4 = pooling.max_pool_with_argmax(net=features, ksize=[1,2,2,1], strides=[1,2,2,1])

		# 1st upsample
		#features = tf_unpool.un_max_pool(features, mask4, 2)
		#features = pooling.unpool(features, mask4, ksize=[1, 2, 2, 1])
		# 1st 'deconv'
		deconv1 = keras.layers.Conv2D(filters=64,
									kernel_size=(7,7), strides=(1,1), padding='SAME',
									data_format='channels_last', activation=None)(features)
		# 2nd upsample
		#features = tf_unpool.un_max_pool(deconv1, mask3, 2)
		features = pooling.unpool(features, mask3, ksize=[1, 2, 2, 1])
		# 2nd 'deconv'
		deconv2 = keras.layers.Conv2D(filters=64,
									kernel_size=(7,7), strides=(1,1), padding='SAME',
									data_format='channels_last', activation=None)(features)

		# 3rd upsample
		#features = tf_unpool.un_max_pool(deconv2, mask2, 2)
		features = pooling.unpool(features, mask2, ksize=[1, 2, 2, 1])
		# 3rd 'deconv'
		deconv3 = keras.layers.Conv2D(filters=64,
									kernel_size=(7,7), strides=(1,1), padding='SAME',
									data_format='channels_last', activation=None)(features)

		# 4th upsample
		#features = tf_unpool.un_max_pool(deconv3, mask1, 2)
		features = pooling.unpool(features, mask1, ksize=[1, 2, 2, 1])
		# 4th 'deconv', output classes predictions
		deconv4 = keras.layers.Conv2D(filters=self.NUM_CLASSES,
									kernel_size=(7,7), strides=(1,1), padding='SAME',
									data_format='channels_last', activation=None)(features)
		# no softmax
		self.logits_before_softmax = deconv4

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

# compute loss function with rebalancing weights
def weighted_loss_func(labels, logits_before_softmax, weights):
	''' input:	labels --- sparse labels with shape [batch, H, W], dtype = tf.uint8
				logits_before_softmax --- logits before softmax with shape [batch, H, W, NUM_CLASSES], dtype = tf.float
				weights --- 1D array, dtype = tf.float32
		output:	loss --- scalar weighted cross entropy, dtype = tf.float32 '''
	# get the NUM_CLASSES
	num_classes = tf.shape(weights)[0]
	# flatten the input labels to 1D vector with length =batch*H*W
	B = tf.shape(labels)[0]
	H = tf.shape(labels)[1]
	W = tf.shape(labels)[2]
	labels = tf.reshape(labels, shape=[B*H*W])
	# convert input sparse labels to one-hot codes (shape = [B*H*W, NUM_CLASSES])
	labels = tf.one_hot(indices=labels, depth=num_classes, dtype=tf.float32)
	# compute logits after softmax
	logits = tf.nn.softmax(logits_before_softmax)
	# reshape the logits to [B*H*W, NUM_CLASSES]
	logits = tf.reshape(logits, shape=[B*H*W, num_classes])
	# compute weighted cross entropy
	batch_loss = tf.reduce_sum(-tf.multiply(labels*tf.log(logits + 1e-9), weights),axis=[1])
	# reduce the batch loss to loss
	loss = tf.reduce_mean(batch_loss)
	# over
	return loss
