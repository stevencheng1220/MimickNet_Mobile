�	yͫ:+��@yͫ:+��@!yͫ:+��@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-yͫ:+��@#1A��^@1��k�B��@A߿yq⫵?I� Q0c�@*�&1,�@)      p=2h
1Iterator::Model::Prefetch::ForeverRepeat::BatchV2�4��R@!/��P��X@)��LM@1�+5oX@:Preprocessing2F
Iterator::Model5�uX�?!Um�@g��?)P�Lۿ��?1ɟ����?:Preprocessing2u
>Iterator::Model::Prefetch::ForeverRepeat::BatchV2::MemoryCachee��7i�?!�c��X�?)��@��?1\���R�?:Preprocessing2P
Iterator::Model::Prefetchh�����?!�u�:�?)h�����?1�u�:�?:Preprocessing2y
BIterator::Model::Prefetch::ForeverRepeat::BatchV2::MemoryCacheImpl��׺��?!��-�x�?)��׺��?1��-�x�?:Preprocessing2_
(Iterator::Model::Prefetch::ForeverRepeat��.ޏ[@!Kr�b��X@)��+ٱ�?1�8�b$��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 11.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	#1A��^@#1A��^@!#1A��^@      ��!       "	��k�B��@��k�B��@!��k�B��@*      ��!       2	߿yq⫵?߿yq⫵?!߿yq⫵?:	� Q0c�@� Q0c�@!� Q0c�@B      ��!       J      ��!       R      ��!       Z      ��!       JGPUb �"�
cgradient_tape/functional_1/separable_conv2d_16/separable_conv2d/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter�0wQ�!�?!�0wQ�!�?"�
bgradient_tape/functional_1/separable_conv2d_1/separable_conv2d/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter����?!�8�D9��?"�
cgradient_tape/functional_1/separable_conv2d_17/separable_conv2d/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter�d��ʝ�?!�u�_`�?"�
cgradient_tape/functional_1/separable_conv2d_18/separable_conv2d/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFiltere+�X<�?!��t���?"�
cgradient_tape/functional_1/separable_conv2d_14/separable_conv2d/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilterU|sO�?!z
����?"�
cgradient_tape/functional_1/separable_conv2d_15/separable_conv2d/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilterQ�t#Q��?!T1@��?"�
bgradient_tape/functional_1/separable_conv2d_2/separable_conv2d/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilterxM�}�?!�K��?"�
bgradient_tape/functional_1/separable_conv2d_3/separable_conv2d/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter�.�}�?!�,���P�?"�
cgradient_tape/functional_1/separable_conv2d_12/separable_conv2d/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilterr�ޜ��?!�`�����?"�
bgradient_tape/functional_1/separable_conv2d_4/separable_conv2d/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter�Ym�iq�?!T4�$�?Q      Y@Y[ZZZZZ@aZZZZZZW@q*a%rND@y$<���6?"�	
both�Your program is POTENTIALLY input-bound because 11.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�40.0649% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 