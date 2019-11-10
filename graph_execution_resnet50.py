import tensorflow as tf
import tensorflow_datasets as tfds
import time

num_batches = 114
batch_size = 32
learning_rate = 0.001

dataset = tfds.load("tf_flowers", split=tfds.Split.TRAIN, as_supervised=True)
dataset = dataset.map(lambda img, label: (tf.image.resize(img, [224, 224]) / 255.0, label)).shuffle(1024).batch(32)
print(dataset) # <DatasetV1Adapter shapes: ((?, 224, 224, 3), (?,)), types: (tf.float32, tf.int64)>
learning_rate_ = tf.placeholder(dtype=tf.float32)
x_true = tf.placeholder(dtype=tf.float32, shape=[32,224,224,3])
y_true = tf.placeholder(dtype=tf.float32, shape=[32])
iterator = dataset.make_one_shot_iterator()
ele = iterator.get_next()
model = tf.keras.applications.resnet.ResNet50(weights=None, classes=5)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
y_pred = model(x_true)
loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y_true, y_pred=y_pred)
train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_).minimize(loss)

start_time = time.time()
with tf.Session() as sess:
	tf.global_variables_initializer().run()
	for e in range(num_batches):
		(images,labels) = sess.run(ele)
		sess.run(train_op, feed_dict={x_true: images, y_true: labels, learning_rate_: learning_rate})

print("Time cost: ", time.time()-start_time)