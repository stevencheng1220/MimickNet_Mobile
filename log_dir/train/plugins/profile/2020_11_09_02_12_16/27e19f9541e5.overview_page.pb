�#	UQ��\��@UQ��\��@!UQ��\��@	���m�X@���m�X@!���m�X@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6UQ��\��@�E�n��Z@1�<��H\@A}���߻?It(CUL�@Y}�OW�@*��Jl�A��S�xՃA2P
Iterator::Model::Prefetch���V�@![��>A@)���V�@1[��>A@:Preprocessing2�
QIterator::Model::Prefetch::ForeverRepeat::BatchV2::MemoryCacheImpl::ParallelMapV2�~k'�r�@!'�zB6@@)�~k'�r�@1'�zB6@@:Preprocessing2�
[Iterator::Model::Prefetch::ForeverRepeat::BatchV2::MemoryCacheImpl::ParallelMapV2::Map::MapR�b$m�@!�S�(�@@)���_Di�@1D)��@@:Preprocessing2�
�Iterator::Model::Prefetch::ForeverRepeat::BatchV2::MemoryCacheImpl::ParallelMapV2::Map::Map::Map::Prefetch::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[1]::FlatMap[0]::TFRecord�����}@!lZ�{���?)�����}@1lZ�{���?:Advanced file read2�
VIterator::Model::Prefetch::ForeverRepeat::BatchV2::MemoryCacheImpl::ParallelMapV2::Mapv?T�r�@!Ő/&5@@)֌r&@1b���ǚ?:Preprocessing2�
`Iterator::Model::Prefetch::ForeverRepeat::BatchV2::MemoryCacheImpl::ParallelMapV2::Map::Map::Map��ң�6"@!�(��*�?)�X��!@1 �5ZI̕?:Preprocessing2�
�Iterator::Model::Prefetch::ForeverRepeat::BatchV2::MemoryCacheImpl::ParallelMapV2::Map::Map::Map::Prefetch::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[1]::FlatMap�0��}@!/��k��?)�ݳ���?1�����F?:Preprocessing2y
BIterator::Model::Prefetch::ForeverRepeat::BatchV2::MemoryCacheImpl��ô�r�@!�i`C@@)�4�BX��?1�t~I;:?:Preprocessing2�
yIterator::Model::Prefetch::ForeverRepeat::BatchV2::MemoryCacheImpl::ParallelMapV2::Map::Map::Map::Prefetch::ParallelMapV2�w���?!���8?)�w���?1���8?:Preprocessing2�
�Iterator::Model::Prefetch::ForeverRepeat::BatchV2::MemoryCacheImpl::ParallelMapV2::Map::Map::Map::Prefetch::ParallelMapV2::AssertCardinality::ParallelInterleaveV4��k	���?!�����7?)��k	���?1�����7?:Preprocessing2�
jIterator::Model::Prefetch::ForeverRepeat::BatchV2::MemoryCacheImpl::ParallelMapV2::Map::Map::Map::Prefetch5���k�?!� O��7?)5���k�?1� O��7?:Preprocessing2�
�Iterator::Model::Prefetch::ForeverRepeat::BatchV2::MemoryCacheImpl::ParallelMapV2::Map::Map::Map::Prefetch::ParallelMapV2::AssertCardinality�ajK��?!F�i1hC?)"P��H��?1�	��,�-?:Preprocessing2F
Iterator::Model�3g}�V�@!�;���>A@)bf��(ϴ?1����S)?:Preprocessing2u
>Iterator::Model::Prefetch::ForeverRepeat::BatchV2::MemoryCacheD��k�r�@!;�u>F@@)�}�֤ۢ?1b��j��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 98.5% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9���m�X@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�E�n��Z@�E�n��Z@!�E�n��Z@      ��!       "	�<��H\@�<��H\@!�<��H\@*      ��!       2	}���߻?}���߻?!}���߻?:	t(CUL�@t(CUL�@!t(CUL�@B      ��!       J	}�OW�@}�OW�@!}�OW�@R      ��!       Z	}�OW�@}�OW�@!}�OW�@JGPUY���m�X@b �"j
@gradient_tape/functional_1/conv2d_16/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�\ ��R�?!�\ ��R�?"j
@gradient_tape/functional_1/conv2d_17/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterD�P5���?!Q�Ϣ��?"i
?gradient_tape/functional_1/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�Am�e��?!r�?7|��?"A
functional_1/activation_16/Relu_FusedConv2D��|���?!��^'�~�?"h
?gradient_tape/functional_1/conv2d_16/Conv2D/Conv2DBackpropInputConv2DBackpropInputE�3����?!��U�C��?"@
functional_1/activation_1/Relu_FusedConv2DW�1��(�?!�<�a��?"A
functional_1/activation_17/Relu_FusedConv2D��?��"�?!������?"h
?gradient_tape/functional_1/conv2d_17/Conv2D/Conv2DBackpropInputConv2DBackpropInput��ab#�?!Yf�����?"g
>gradient_tape/functional_1/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput0�R���?!ܔIB��?"6
depthwise_2DepthwiseConv2dNative m��I�?!�����^�?Q      Y@Y�_OEV@a�H���%@qKw���I?y�V�8�i?"�
host�Your program is HIGHLY input-bound because 98.5% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: B 