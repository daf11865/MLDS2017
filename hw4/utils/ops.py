import tensorflow as tf

def linear(x, out_size, scope, bias = True,
					w_initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32),
					b_initializer = tf.constant_initializer(0.0),
					activation = None):

	shape = x.get_shape().as_list()
	in_size = shape[-1]

	with tf.variable_scope(scope):
		x_flat = tf.reshape(x, [-1,in_size])
		w = tf.get_variable("w", initializer=w_initializer, shape=[in_size, out_size])
		b = tf.get_variable("b", initializer=b_initializer, shape=[out_size]) if bias else 0.
		y = tf.matmul(x_flat, w) +  b
		y = activation(y) if activation else y
	return y

def fusion(x, y, out_size, scope, bias = True,
					w_initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32),
					b_initializer = tf.constant_initializer(0.0),
					activation = None):

	shape = x.get_shape().as_list()
	x_size = shape[1]
	shape = y.get_shape().as_list()
	y_size = shape[1]

	with tf.variable_scope(scope):
		t = tf.get_variable(name="t", initializer=w_initializer, shape=[x_size*y_size, out_size])
		t = tf.reshape(t, [out_size*y_size, x_size])
		temp = tf.matmul(t, x, transpose_b=True)
		temp = tf.reshape(tf.transpose(temp),[-1, out_size, y_size])
		z = tf.matmul(temp, tf.reshape(y,[-1,y_size,1]) )
		z = tf.reshape(z,shape=[-1,out_size])

	return z

def deconv(x, out_depth, scope, k=5, s=2,
					w_initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32),
					b_initializer = tf.constant_initializer(0.0),
					normalizer = None,
					is_train = True,
					activation = None):

	shape = x.get_shape().as_list()
	in_depth = shape[-1]
	out_shape = [shape[0], shape[1]*s, shape[2]*s, out_depth]

	with tf.variable_scope(scope):
		w = tf.get_variable("w", initializer=w_initializer, shape=[k, k, out_depth, in_depth])
		b = tf.get_variable("b", initializer=b_initializer, shape=[out_depth])
		y = tf.nn.conv2d_transpose(x, w, output_shape=out_shape, strides=[1,s,s,1]) + b
		y = normalizer(y, is_training=is_train) if normalizer else y
		y = activation(y) if activation else y

	return y

def conv(x, out_depth, scope, k=5, s=2,
					w_initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32),
					b_initializer = tf.constant_initializer(0.0),
					normalizer = None,
					is_train = True,
					residual = False,
					activation = None):

	shape = x.get_shape().as_list()
	in_depth = shape[-1]

	with tf.variable_scope(scope):
		w = tf.get_variable("w", initializer=w_initializer, shape=[k, k, in_depth, out_depth])
		b = tf.get_variable("b", initializer=b_initializer, shape=[out_depth])
		y = tf.nn.conv2d(x, w, strides=[1,s,s,1], padding='SAME') + b
		y = normalizer(y, is_training=is_train) if normalizer else y
		if residual:
			assert x.get_shape().as_list() == y.get_shape().as_list()
			y = x + y
		y = activation(y) if activation else y

	return y

def batchnorm(x, is_train, scope):
		return tf.contrib.layers.batch_norm(x, decay=0.99, center=True, scale=True, epsilon=1e-5, is_training=is_train, updates_collections=None, scope=scope)

def lrelu(x, leak=0.2):
	return tf.maximum(x, leak*x)

def crelu(x):
	return tf.nn.relu(tf.concat([x, -x], axis=-1))


def upsampling(x, scale = 2):
	[bs, h, w, c] = x.get_shape().as_list()
	return tf.image.resize_nearest_neighbor(x, [h*scale, w*scale])

if __name__ == "__main__":
	tf.reset_default_graph()

	a = tf.reshape(tf.constant(range(16),dtype=tf.float32),[1,4,4,1])
	w = tf.get_variable("w", initializer=tf.constant_initializer(1.0), shape=[5,5,1,1])
	out=tf.nn.conv2d_transpose(a,w,output_shape = [1,8,8,1],strides=[1,2,2,1],padding='SAME')

	sess=tf.Session()
	sess.run(tf.global_variables_initializer())

	out.eval(session=sess)
