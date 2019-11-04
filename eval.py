import tensorflow as tf
import numpy as np
import time

"""Simple test focused on new features of TF 2.0

- Eager Execution vs. Graph Execution

- tf.function and AutoGraph

- tf.keras vs. Keras

"""

def init_data():
	X_raw = np.array([2013, 2014, 2015, 2016, 2017, 2018], dtype=np.float32)
	y_raw = np.array([12000, 14000, 15000, 16500, 17500, 19000], dtype=np.float32)

	X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
	y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

	return X, y

# an example under Graph Execution of TF 1.x
def graph_execution():
	X, y = init_data()
	learning_rate_ = tf.placeholder(dtype=tf.float32)
	X_ = tf.placeholder(dtype=tf.float32, shape=[5])
	y_ = tf.placeholder(dtype=tf.float32, shape=[5])
	a = tf.get_variable('a', dtype=tf.float32, shape=[], initializer=tf.zeros_initializer)
	b = tf.get_variable('b', dtype=tf.float32, shape=[], initializer=tf.zeros_initializer)

	y_pred = a * X_ + b # Linear regression
	loss = tf.constant(0.5) * tf.reduce_sum(tf.square(y_pred - y_))
	train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_).minimize(loss)

	num_epoch = 10000
	learning_rate = 1e-3
	start_time = time.time()
	with tf.Session() as sess:
	    tf.global_variables_initializer().run()
	    for e in range(num_epoch):
	        sess.run(train_op, feed_dict={X_: X, y_: y, learning_rate_: learning_rate})
	    print(sess.run([a, b]))
	print("Graph Execution: ", time.time()-start_time)

# an example under Eager Execution of TF 2.0
def eager_execution():
	X, y = init_data()
	X = tf.constant(X)
	y = tf.constant(y)
	a = tf.Variable(initial_value=0.)
	b = tf.Variable(initial_value=0.)
	variables = [a, b]

	num_epoch = 10000
	optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
	start_time = time.time()
	for e in range(num_epoch):
	    with tf.GradientTape() as tape:
	        y_pred = a * X + b
	        loss = 0.5 * tf.reduce_sum(tf.square(y_pred - y))
	    grads = tape.gradient(loss, variables)
	    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
	print(a, b)
	print("Eager Execution: ", time.time()-start_time)

# test the performance improvement under tf.function		
@tf.function
def train_one_step(X, y, variables):
	with tf.GradientTape() as tape:
		y_pred = variables[0] * X + variables[1]
		loss = 0.5 * tf.reduce_sum(tf.square(y_pred - y))
	grads = tape.gradient(loss, variables)
	optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

def tf_function():
	X, y = init_data()
	X = tf.constant(X)
	y = tf.constant(y)
	a = tf.Variable(initial_value=0.)
	b = tf.Variable(initial_value=0.)
	variables = [a, b]

	num_epoch = 10000
	optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
	start_time = time.time()
	for e in range(num_epoch):
		train_one_step(X, y, variables)
	print(a, b)
	print("@tf.function: ", time.time()-start_time)


if __name__ == '__main__':
	# graph_execution()
	eager_execution()
	tf_function()
