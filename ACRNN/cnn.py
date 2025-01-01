# cnn layer
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

class cnn:
	def __init__(self, weight_stddev = 0.1, bias_constant = 0.1, padding = "SAME"):
			self.weight_stddev = weight_stddev   
			self.bias_constant = bias_constant   
			self.padding = padding
                               
                                          
	def weight_variable(self, shape):           
												
		initial = tf.compat.v1.truncated_normal(shape, stddev = self.weight_stddev)
		return tf.Variable(initial) 


	def bias_variable(self, shape):            
		initial = tf.constant(self.bias_constant, shape = shape)
		return tf.Variable(initial)



	def conv2d(self, x, W, kernel_stride):     
		return tf.nn.conv2d(x, W, strides=[1, kernel_stride, kernel_stride, 1], padding=self.padding)


	def apply_conv2d(self, x, filter_height, filter_width, in_channels, out_channels, kernel_stride, train_phase):
		weight = self.weight_variable([filter_height, filter_width, in_channels, out_channels])
		bias = self.bias_variable([out_channels]) 
		conv_2d = tf.add(self.conv2d(x, weight, kernel_stride), bias)

		conv_2d_bn = self.batch_norm_cnv_2d(conv_2d, train_phase)
		return tf.nn.relu(conv_2d_bn)


	def batch_norm_cnv_2d(self, inputs, train_phase):  # 2차원에서 axis 3은 channel을 의미
		bn_layer = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.993, epsilon=1e-5, scale=False)
		return bn_layer(inputs, training = train_phase)


	def batch_norm_out(self, inputs, train_phase):  # 아마 out에서의 inputs 형식이 channel이 axis 1일 것임
		bn_layer = tf.keras.layers.BatchNormaliztion(axis=1, momentum=0.993, epsilon=1e-5, scale=False)
		#inputs = tf.ensure_shape(inputs, [15, 1, None, 40])
		return bn_layer(inputs, training = train_phase)

 
	def apply_max_pooling(self, x, pooling_height, pooling_width, pooling_stride):
	# API: must ksize[0]=ksize[4]=1, strides[0]=strides[4]=1
		return tf.nn.max_pool(x, ksize=[1, pooling_height, pooling_width, 1], strides=[1, pooling_stride, pooling_stride, 1], padding=self.padding)


	def apply_fully_connect(self, x, x_size, fc_size, train_phase):
		weight = self.weight_variable([x_size, fc_size])
		bias = self.bias_variable([fc_size])
		fc = tf.add(tf.matmul(x, weight), bias)
		fc_bn = self.batch_norm_out(fc, train_phase)
		return tf.nn.relu(fc_bn)


	def apply_readout(self, x, x_size, readout_size):
		readout_weight = self.weight_variable([x_size, readout_size])
		readout_bias = self.bias_variable([readout_size])
		return tf.add(tf.matmul(x, readout_weight), readout_bias)