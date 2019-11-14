import tensorboard
import tensorflow as tf
tf.enable_eager_execution()
import tempfile
import zipfile
import os
import time

batch_size = 128
num_classes = 10
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

def prepare_data():	
	# Prepare the training date
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)	
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')
	# convert class vectors to binary class matrices
	y_train = tf.keras.utils.to_categorical(y_train, num_classes)
	y_test = tf.keras.utils.to_categorical(y_test, num_classes)
	return (x_train, y_train), (x_test, y_test)

def model_without_pruned():
	# Build the MNIST model
	(x_train, y_train), (x_test, y_test) = prepare_data()
	epochs = 12
	l = tf.keras.layers
	model = tf.keras.Sequential([
		l.Conv2D(
			32, 5, padding='same', activation='relu', input_shape=input_shape),
		l.MaxPooling2D((2, 2), (2, 2), padding='same'),
		l.BatchNormalization(),
		l.Conv2D(64, 5, padding='same', activation='relu'),
		l.MaxPooling2D((2, 2), (2, 2), padding='same'),
		l.Flatten(),
		l.Dense(1024, activation='relu'),
		l.Dropout(0.4),
		l.Dense(num_classes, activation='softmax')
	])
	model.summary()

	# Train model without pruning
	logdir = tempfile.mkdtemp()
	print('Writing training logs to ' + logdir)
	# %tensorboard --logdir={logdir}
	callbacks = [tf.keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=0)]
	model.compile(
		loss=tf.keras.losses.categorical_crossentropy,
		optimizer='adam',
		metrics=['accuracy'])
	model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=callbacks,
          validation_data=(x_test, y_test))
	start_test = time.time()
	score = model.evaluate(x_test, y_test, verbose=0)
	end_test = time.time()
	print('Test latency:', end_test - start_test)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	# Save model
	# _, orginal_file = tempfile.mkstemp('.h5')
	orginal_file = './orginal_model_file.h5'
	print('Saving model to: ', orginal_file)
	tf.keras.models.save_model(model, orginal_file, include_optimizer=False)

"""
Train a pruned MNIST
We provide a prune_low_magnitude() API to train models with removed connections. 
The Keras-based API can be applied at the level of 
1. individual layers, 
2. or the entire model. 
"""
from tensorflow_model_optimization.sparsity import keras as sparsity
import numpy as np

def layer_pruned_model():
	#Build a pruned model layer by layer
	epochs = 12
	(x_train, y_train), (x_test, y_test) = prepare_data()
	num_train_samples = x_train.shape[0]
	end_step = np.ceil(1.0 * num_train_samples / batch_size).astype(np.int32) * epochs
	print('End step: ' + str(end_step))
	pruning_params = {
		'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
	                                                   final_sparsity=0.90,
	                                                   begin_step=2000,
	                                                   end_step=end_step,
	                                                   frequency=100)}

	#build the model
	l = tf.keras.layers	
	pruned_model = tf.keras.Sequential([
	    sparsity.prune_low_magnitude(
	        l.Conv2D(32, 5, padding='same', activation='relu'),
	        input_shape=input_shape,
	        **pruning_params),
	    l.MaxPooling2D((2, 2), (2, 2), padding='same'),
	    l.BatchNormalization(),
	    sparsity.prune_low_magnitude(
	        l.Conv2D(64, 5, padding='same', activation='relu'), **pruning_params),
	    l.MaxPooling2D((2, 2), (2, 2), padding='same'),
	    l.Flatten(),
	    sparsity.prune_low_magnitude(l.Dense(1024, activation='relu'),
	                                 **pruning_params),
	    l.Dropout(0.4),
	    sparsity.prune_low_magnitude(l.Dense(num_classes, activation='softmax'),
	                                 **pruning_params)
	])

	pruned_model.summary()

	logdir = tempfile.mkdtemp()
	print('Writing training logs to ' + logdir)
	# %tensorboard --logdir={logdir}

	# train the model
	pruned_model.compile(
	    loss=tf.keras.losses.categorical_crossentropy,
	    optimizer='adam',
	    metrics=['accuracy'])
	callbacks = [
	    sparsity.UpdatePruningStep(),
	    sparsity.PruningSummaries(log_dir=logdir, profile_batch=0)
	]

	pruned_model.fit(x_train, y_train,
	          batch_size=batch_size,
	          epochs=10,
	          verbose=1,
	          callbacks=callbacks,
	          validation_data=(x_test, y_test))
	score = pruned_model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	# Save and restore
	checkpoint_file = './pruned_checkpoint_file.h5'
	# _, checkpoint_file = tempfile.mkstemp('.h5')
	print('Saving pruned model to: ', checkpoint_file)
	# saved_model() sets include_optimizer to True by default. Spelling it out here
	# to highlight.
	tf.keras.models.save_model(pruned_model, checkpoint_file, include_optimizer=True)

	with sparsity.prune_scope():
	  restored_model = tf.keras.models.load_model(checkpoint_file)

	restored_model.fit(x_train, y_train,
	                   batch_size=batch_size,
	                   epochs=2,
	                   verbose=1,
	                   callbacks=callbacks,
	                   validation_data=(x_test, y_test))

	start_test = time.time()
	score = restored_model.evaluate(x_test, y_test, verbose=0)
	end_test = time.time()
	print('Test latency:', end_test - start_test)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	final_model = sparsity.strip_pruning(pruned_model)
	final_model.summary()
	layer_pruned_file = './layer_pruned_file.h5'
	# _, layer_pruned_file = tempfile.mkstemp('.h5')
	print('Saving pruned model to: ', layer_pruned_file)
	tf.keras.models.save_model(final_model, layer_pruned_file, include_optimizer=False)

def model_info(file_name):
	_, zip1 = tempfile.mkstemp('.zip') 
	with zipfile.ZipFile(zip1, 'w', compression=zipfile.ZIP_DEFLATED) as f:
		f.write(file_name)
	print('-------------------------\n', file_name)
	print("Size of the model before compression: %.2f Mb" % 
		(os.path.getsize(file_name) / float(2**20)))
	print("Size of the model after compression: %.2f Mb" % 
		(os.path.getsize(zip1) / float(2**20)))
	print('-------------------------')

def eval_tflite(tflite_model_file):
	(x_train, y_train), (x_test, y_test) = prepare_data()
	interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
	interpreter.allocate_tensors()
	input_index = interpreter.get_input_details()[0]["index"]
	output_index = interpreter.get_output_details()[0]["index"]
	total_seen = 0
	num_correct = 0
	start_test = time.time()
	for img, label in zip(x_test, y_test):
		inp = img.reshape((1, 28, 28, 1))
		total_seen += 1
		interpreter.set_tensor(input_index, inp)
		interpreter.invoke()
		predictions = interpreter.get_tensor(output_index)
		if np.argmax(predictions) == np.argmax(label):
			num_correct += 1
	end_test = time.time()
	print('Test latency: ', end_test - start_test)
	print('Test accuracy:', float(num_correct) / float(total_seen))


def quantize_tflite(keras_file, quant_file, precision=None):

	converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)
	if precision=='int8':
		converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
		# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
		# converter.inference_input_type = tf.uint8
		# converter.inference_output_type = tf.uint8
	if precision=='float16':
		converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
		converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]

	tflite_quant_model = converter.convert()
	with open(quant_file, 'wb') as f:
		f.write(tflite_quant_model)

	eval_tflite(quant_file)

model_without_pruned() #'./orginal_model_file.h5'
layer_pruned_model() #'./layer_pruned_file.h5'

model_nopruned_file = './orginal_model_file.h5'
layer_pruned_file = './layer_pruned_file.h5'

nopruned_tflite_file = './nopruned_tflite_file.h5'
pruned_tflite_file = './pruned_tflite_file.h5'
sparse_tflile_quant8_file = './sparse_tflile_quant8_file.h5'
sparse_tflile_quant16_file = './sparse_tflile_quant16_file.h5'

quantize_tflite(model_nopruned_file, nopruned_tflite_file)
quantize_tflite(layer_pruned_file, pruned_tflite_file)
quantize_tflite(layer_pruned_file, sparse_tflile_quant16_file, 'float16')
quantize_tflite(layer_pruned_file, sparse_tflile_quant8_file, 'int8')


model_info(model_nopruned_file)
model_info(layer_pruned_file)
model_info(nopruned_tflite_file)
model_info(pruned_tflite_file)
model_info(sparse_tflile_quant16_file)
model_info(sparse_tflile_quant8_file)
