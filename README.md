# Tensorflow 2.0

## 1. New Features of TF2
[TensorFlow 2.0 is now available!](https://medium.com/tensorflow/tensorflow-2-0-is-now-available-57d706c2a9ab)

1. tight integration of [Keras into TensorFlow](https://www.tensorflow.org/guide/keras/overview)
2. [eager execution](https://www.tensorflow.org/guide/eager) by default (Pythonic function execution)
3. standardized on the [SavedModel](https://www.tensorflow.org/guide/saved_model) file format
4. [Distribution Strategy API](https://www.tensorflow.org/guide/distributed_training) to distribute training
5. tightly integrated with TensorRT, [converter API](https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/compiler/tensorrt/trt_convert.py#L766-L859), [TFTRT](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html)
6. better performance with [tf.data](https://www.tensorflow.org/guide/data)
7. better performance with [@tf.function](https://www.tensorflow.org/tutorials/customization/performance) and [AutoGraph](https://www.tensorflow.org/guide/function)
8. [TensorFlow Serving](),[TensorFlow Lite](), [TensorFlow.js]() and Swift for TensorFlow (future)
9. API Cleanup

## 2. Test Result

### Eager Execution, Graph Execution and tf.function(AutoGraph)

1. 传统的TensorFlow开发使用Graph Execution，首先通过变量和Placeholder来定义一个计算图，然后启动一个Session，通过TensorFlow引擎来执行这个计算图，最后给出我们需要的结果。不能实时计算、调试困难、灵活性差、无法使用 Python 原生控制语句。
2. Eager Execution无需构建图，操作返回具体值，而不是构建计算图以便稍后运行，python式方便调试。
3. 理想情况下, tf.function能让我们实现用动态图写(方便, 灵活), 用静态图跑(高效, 稳定)。理论上静态图的执行效率更高, 但是加速并不是一定的. 一般来说, 计算图越复杂, 加速效果越明显. 对于复杂的计算图, 比如训练深度学习模型, 获得的加速是巨大的. 

4. result

	| mode | model |latency (s)|
	| :----:  | :----: | :----: |
	| Graph Execution | Linear regression | 5.3317999839782715 |
	| Eager Execution | - | 22.863434553146362 |
	| tf.function | - | 3.6043503284454346 |

## 3. Concise Handbook of TF2
[TF2.0 wiki](https://tf.wiki/index.html)

- 模型的建立与调试： 使用动态图模式 **Eager Execution** 和高级 API 框架 **Keras**，结合可视化工具 **TensorBoard**，简易、快速地建立和调试模型；

- 模型的训练： 支持 CPU / 单 GPU / 单机多卡 GPU / 多机集群 / TPU 训练模型，充分利用海量数据和计算资源进行高效训练；

- 模型的部署： 通过 **TensorFlow Serving**、**TensorFlow Lite**、**TensorFlow.js** 等组件，使 TensorFlow 模型能够无缝地部署到服务器、移动端、嵌入式端和网页端等多种使用场景；

- 预训练模型调用： 通过 **TensorFlow Hub** 和 **Tensor2Tensor**，可以方便地调用预训练完毕的已有成熟模型。

### Basic
- Tensor: 张量 as 数组：标量（0 维数组）、向量（1 维数组）、矩阵（2 维数组）等；属性：形状（A.shape）、类型（A.dtype，默认float32）和值（A.numpy()）
- Variable：变量与普通张量的一个重要区别是其默认能够被 TensorFlow 的自动求导机制（tf.GradientTape()）所求导，因此往往被用于定义机器学习模型的参数。
- Operation
- Eager Execution（动态图）（vs Graph Execution（静态图））


### Model training and common modules
- Construction： `tf.keras.Model`, `tf.keras.layers`
- Loss functions： `tf.keras.losses`
- Optimizer： `tf.keras.optimizer`
- Evaluation： `tf.keras.metrics`

- Checkpoint： 只保存模型的参数，不保存模型的计算过程，因此一般用于在具有模型源代码的时候恢复之前训练好的模型参数。

	`tf.train.Checkpoint` vs. `tf.train.Saver`
	
	`checkpoint.save()` and `checkpoint.restore()`
	
- TensorBoard `tf.summary`
	
	`tf.summary.create_file_writer()` `tf.summary.scalar()` 
	
	`tensorboard --logdir=./tensorboard`
	
- Data `tf.data.Dataset`

- @tf.function for Graph Execution: 将模型转换为易于部署且高性能的 TensorFlow 图模型,一定的性能提升

- AutoGraph：将 Python 控制流转换为 TensorFlow 计算图

- tf.config：GPU 的使用与分配
 
 指定： `tf.config.experimental.set_visible_devices(devices=gpus[0:2], device_type='GPU')` or `CUDA_VISIBLE_DEVICES=0,1`
 
 分配： 
 tf.config.experimental.set_memory_growth
 tf.config.experimental.set_virtual_device_configuration

### Deployment

- Export

	1. SavedModel: 不仅包含参数的权值，还包含计算的流程（即计算图）
		`tf.saved_model.save()` and `tf.saved_model.load`
		
	2. Keras Sequential save（HDF5 格式）
		`model.save('mnist_cnn.h5')` and `keras.models.load_model("mnist_cnn.h5")`

- Tensorflow Serving
	[TF-Serving](https://www.tensorflow.org/tfx/guide/serving)

	~~~
	tensorflow_model_server \
	    --rest_api_port=端口号（如8501） \
	    --model_name=模型名 \
	    --model_base_path="SavedModel格式模型的文件夹绝对地址（不含版本号）"
	~~~

- TensorFlow Lite

	1. 模型转换(tflite_convert toolkit)
	
		Protocol Buffer to FlatBuffers (.tflite)
	
		~~~
		usage: tflite_convert [-h] --output_file OUTPUT_FILE
		                      (--saved_model_dir SAVED_MODEL_DIR | --keras_model_file KERAS_MODEL_FILE)
		  --output_file OUTPUT_FILE
		                        Full filepath of the output file.
		  --saved_model_dir SAVED_MODEL_DIR
		                        Full path of the directory containing the SavedModel.
		  --keras_model_file KERAS_MODEL_FILE
		                        Full filepath of HDF5 file containing tf.Keras model.
		~~~

	2. 边缘部署
	3. Model quantization [Lite Model optimization](https://www.tensorflow.org/lite/performance/model_optimization)

		
- TensorFlow.js
- TensorFlow Hub
	[hub site](https://www.tensorflow.org/hub)


### Large-scale training

- Distributed
- TPU (Tensor Processing Unit)

### Performance Optimization

- better performance with [tf.data](https://www.tensorflow.org/guide/data_performance)
- 减少operation，多用矩阵张量运算而不是for循环
- GPU acceleration： CNN>RNN>RL
	

## 4. Model optimization

### TensorFlow Model Optimization Toolkit

[model_optimization](https://www.tensorflow.org/model_optimization)

- Weight pruning

	~~~
	import tensorflow_model_optimization as tfmot
	
	model = build_your_model()  
	
	pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
	                        initial_sparsity=0.0, final_sparsity=0.5,
	                        begin_step=2000, end_step=4000)
	
	model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, pruning_schedule=pruning_schedule)
	
	... 
	
	model_for_pruning.fit(...)
	~~~
	
	
- Post-Training Quantization (in TF-Lite)

	~~~
	import tensorflow as tf

	converter = tf.lite.TFLiteConverter.from_saved_model('saved/1')
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	tflite_quant_model = converter.convert()
	~~~

### TFTRT

	  ~~~
	  import tensorflow as tf
	  from tensorflow.python.compiler.tensorrt import trt_convert as trt
	  '''
	  DEFAULT_TRT_CONVERSION_PARAMS = trt.TrtConversionParams(
	    rewriter_config_template=None,
	    max_workspace_size_bytes=DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES,
	    precision_mode=TrtPrecisionMode.FP32,
	    minimum_segment_size=3,
	    is_dynamic_op=True,
	    maximum_cached_engines=1,
	    use_calibration=True,
	    max_batch_size=1)
	  '''
	  params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
		 precision_mode='FP16')
	  converter = trt.TrtGraphConverterV2(
		 input_saved_model_dir="my_dir", conversion_params=params)
	  converter.convert()
	  converter.save(output_saved_model_dir)
	  ~~~
