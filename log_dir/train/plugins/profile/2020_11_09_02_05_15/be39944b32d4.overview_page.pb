�#	�z����@�z����@!�z����@	��s�8FX@��s�8FX@!��s�8FX@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�z����@��V|2t@1�N��҆N@A1��c��?I�~2Ƈ�
@YŐ�L2R�@*�S�*#A#���;#�A2P
Iterator::Model::Prefetchc ��R�@!�!��t�A@)c ��R�@1�!��t�A@:Preprocessing2�
QIterator::Model::Prefetch::ForeverRepeat::BatchV2::MemoryCacheImpl::ParallelMapV2,���n�@!_N���h?@),���n�@1_N���h?@:Preprocessing2�
[Iterator::Model::Prefetch::ForeverRepeat::BatchV2::MemoryCacheImpl::ParallelMapV2::Map::Map����Kh�@!�@�_?@)��7�d�@1vN��UZ?@:Preprocessing2�
�Iterator::Model::Prefetch::ForeverRepeat::BatchV2::MemoryCacheImpl::ParallelMapV2::Map::Map::Map::Prefetch::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[1]::FlatMap[0]::TFRecord���П�@!m���y�?)���П�@1m���y�?:Advanced file read2�
VIterator::Model::Prefetch::ForeverRepeat::BatchV2::MemoryCacheImpl::ParallelMapV2::MapY2��n�@!}���h?@)Iڍ>�*@1u�2����?:Preprocessing2�
`Iterator::Model::Prefetch::ForeverRepeat::BatchV2::MemoryCacheImpl::ParallelMapV2::Map::Map::Mapt���&"@!�3�j�?)'/2��!@1R%I��?:Preprocessing2�
�Iterator::Model::Prefetch::ForeverRepeat::BatchV2::MemoryCacheImpl::ParallelMapV2::Map::Map::Map::Prefetch::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[1]::FlatMap����⡃@!둖��|�?)/2�F��?1,���3G?:Preprocessing2y
BIterator::Model::Prefetch::ForeverRepeat::BatchV2::MemoryCacheImpl��So�@!�����h?@)l��g��?1������@?:Preprocessing2�
jIterator::Model::Prefetch::ForeverRepeat::BatchV2::MemoryCacheImpl::ParallelMapV2::Map::Map::Map::Prefetch-��x>�?!��,<?)-��x>�?1��,<?:Preprocessing2�
yIterator::Model::Prefetch::ForeverRepeat::BatchV2::MemoryCacheImpl::ParallelMapV2::Map::Map::Map::Prefetch::ParallelMapV2�"�Ƥ�?!vwc��;?)�"�Ƥ�?1vwc��;?:Preprocessing2�
�Iterator::Model::Prefetch::ForeverRepeat::BatchV2::MemoryCacheImpl::ParallelMapV2::Map::Map::Map::Prefetch::ParallelMapV2::AssertCardinality::ParallelInterleaveV4�z��9y�?!�ppw8?)�z��9y�?1�ppw8?:Preprocessing2�
�Iterator::Model::Prefetch::ForeverRepeat::BatchV2::MemoryCacheImpl::ParallelMapV2::Map::Map::Map::Prefetch::ParallelMapV2::AssertCardinality�EB[Υ�?!�`D?)ٖg)Y�?1�e]UJ/?:Preprocessing2F
Iterator::ModelM�^��R�@!(�<�z�A@)d��3�İ?1��^~]z'?:Preprocessing2u
>Iterator::Model::Prefetch::ForeverRepeat::BatchV2::MemoryCacheH�`o�@!�_�k�h?@)��Tka�?1���oGC?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 97.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9��s�8FX@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��V|2t@��V|2t@!��V|2t@      ��!       "	�N��҆N@�N��҆N@!�N��҆N@*      ��!       2	1��c��?1��c��?!1��c��?:	�~2Ƈ�
@�~2Ƈ�
@!�~2Ƈ�
@B      ��!       J	Ő�L2R�@Ő�L2R�@!Ő�L2R�@R      ��!       Z	Ő�L2R�@Ő�L2R�@!Ő�L2R�@JGPUY��s�8FX@b �"-
IteratorGetNext/_3_SendT����?!T����?"j
@gradient_tape/functional_1/conv2d_16/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterT�7l�p�?!�t�5'��?"i
?gradient_tape/functional_1/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter����䜩?!��ac`&�?"j
@gradient_tape/functional_1/conv2d_17/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�������?!��Dh�D�?"A
functional_1/activation_16/Relu_FusedConv2D+����T�?!�`8�xO�?"-
IteratorGetNext/_1_Send��T3_n�?!ޭm�^F�?"h
?gradient_tape/functional_1/conv2d_16/Conv2D/Conv2DBackpropInputConv2DBackpropInput��mΚ?!�~�E��?"6
depthwise_1DepthwiseConv2dNativeպK�z�?!��" �Z�?"j
@gradient_tape/functional_1/conv2d_14/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter#h��d�?!{Wɺ8��?"6
depthwise_2DepthwiseConv2dNative^[�U7`�?!1�#0<'�?Q      Y@Y���~HV@a;S�<�%@q T2�5y?y�.*�y{?"�
host�Your program is HIGHLY input-bound because 97.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: B 