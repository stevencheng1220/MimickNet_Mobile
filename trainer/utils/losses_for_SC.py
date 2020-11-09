# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
''' Added new custom loss functions. Written partially by Steven Cheng with template from Tensorflow's SSIM '''
''' Custom loss functions'''

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
import tensorflow as tf

def _verify_compatible_image_shapes(img1, img2):
    """Checks if two image tensors are compatible for applying SSIM or PSNR.
    This function checks if two sets of images have ranks at least 3, and if the
    last three dimensions match.
    Args:
    img1: Tensor containing the first image batch.
    img2: Tensor containing the second image batch.
    Returns:
    A tuple containing: the first tensor shape, the second tensor shape, and a
    list of control_flow_ops.Assert() ops implementing the checks.
    Raises:
    ValueError: When static shape check fails.
    """
    shape1 = img1.get_shape().with_rank_at_least(3)
    shape2 = img2.get_shape().with_rank_at_least(3)
    shape1[-3:].assert_is_compatible_with(shape2[-3:])

    if shape1.ndims is not None and shape2.ndims is not None:
        for dim1, dim2 in zip(reversed(shape1.dims[:-3]),
                              reversed(shape2.dims[:-3])):
            if not (dim1 == 1 or dim2 == 1 or dim1.is_compatible_with(dim2)):
                raise ValueError('Two images are not compatible: %s and %s' % (shape1, shape2))

    shape1, shape2 = array_ops.shape_n([img1, img2])

    checks = []
    checks.append(control_flow_ops.Assert(
      math_ops.greater_equal(array_ops.size(shape1), 3),
      [shape1, shape2], summarize=10))
    checks.append(control_flow_ops.Assert(
      math_ops.reduce_all(math_ops.equal(shape1[-3:], shape2[-3:])),
      [shape1, shape2], summarize=10))
    return shape1, shape2, checks


def _ssim_helper(x, y, reducer, max_val, compensation=1.0):
    _SSIM_K1 = 0.01
    _SSIM_K2 = 0.03

    c1 = (_SSIM_K1 * max_val) ** 2
    c2 = (_SSIM_K2 * max_val) ** 2

    # SSIM luminance measure is
    # (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1).

    mean0 = reducer(x)
    mean1 = reducer(y)
    num0 = mean0 * mean1 * 2.0
    den0 = math_ops.square(mean0) + math_ops.square(mean1)
    luminance = (num0 + c1) / (den0 + c1)
    # print(luminance.shape)

    # SSIM contrast-structure measure is
    #   (2 * cov_{xy} + c2) / (cov_{xx} + cov_{yy} + c2).
    # Note that `reducer` is a weighted sum with weight w_k, \sum_i w_i = 1, then
    #   cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
    #          = \sum_i w_i x_i y_i - (\sum_i w_i x_i) (\sum_j w_j y_j).
    num1 = reducer(x * y) * 2.0
    den1 = reducer(math_ops.square(x) + math_ops.square(y))
    c2 *= compensation
    cs = (num1 - num0 + c2) / (den1 - den0 + c2)  

    ''' Masking zero values '''
    zero = tf.constant(0, tf.float32)
    mask = tf.cast(tf.not_equal(x, zero), tf.float32)
    mask = tf.cast(reducer(mask), tf.bool)
    zeros = tf.zeros(shape=tf.shape(mask))
    
    luminance = tf.where(mask, luminance, zeros)
    cs = tf.where(mask, cs, zeros)
    
    # SSIM score is the product of the luminance and contrast-structure measures.
    return luminance, cs, mask


def _fspecial_gauss(size, sigma):
    size = ops.convert_to_tensor(size, dtypes.int32)
    sigma = ops.convert_to_tensor(sigma)

    coords = math_ops.cast(math_ops.range(size), sigma.dtype)
    coords -= math_ops.cast(size - 1, sigma.dtype) / 2.0

    g = math_ops.square(coords)
    g *= -0.5 / math_ops.square(sigma)

    g = array_ops.reshape(g, shape=[1, -1]) + array_ops.reshape(g, shape=[-1, 1])
    g = array_ops.reshape(g, shape=[1, -1])  # For tf.nn.softmax().
    g = nn_ops.softmax(g)
    return array_ops.reshape(g, shape=[size, size, 1, 1])


def _ssim_per_channel(img1, img2, max_val=1.0):
    filter_size = constant_op.constant(11, dtype=dtypes.int32)
    filter_sigma = constant_op.constant(1.5, dtype=img1.dtype)

    shape1, shape2 = array_ops.shape_n([img1, img2])
    checks = [
      control_flow_ops.Assert(math_ops.reduce_all(math_ops.greater_equal(
          shape1[-3:-1], filter_size)), [shape1, filter_size], summarize=8),
      control_flow_ops.Assert(math_ops.reduce_all(math_ops.greater_equal(
          shape2[-3:-1], filter_size)), [shape2, filter_size], summarize=8)]

    # Enforce the check to run before computation.
    with ops.control_dependencies(checks):
        img1 = array_ops.identity(img1)

    kernel = _fspecial_gauss(filter_size, filter_sigma)
    kernel = array_ops.tile(kernel, multiples=[1, 1, shape1[-1], 1])

    compensation = 1.0

    def reducer(x):
        shape = array_ops.shape(x)
        x = array_ops.reshape(x, shape=array_ops.concat([[-1], shape[-3:]], 0))
        y = nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
        return array_ops.reshape(y, array_ops.concat([shape[:-3],
                                                      array_ops.shape(y)[1:]], 0))

    luminance, cs, mask = _ssim_helper(img1, img2, reducer, max_val, compensation)
    ''' Checking things'''

    # Average over the second and the third from the last: height, width.
    axes = constant_op.constant([-3, -2], dtype=dtypes.int32)
    ssim_val = math_ops.reduce_sum(luminance * cs, axes)
    luminance = tf.keras.backend.flatten(luminance)
    cs = tf.keras.backend.flatten(cs)
    ssim_val = ssim_val / tf.cast(tf.shape(cs)[0], tf.float32)
    mask = tf.cast(luminance, dtype=tf.bool)
    luminance = tf.boolean_mask(luminance, mask)
    cs = tf.boolean_mask(cs, mask)

    # cs = math_ops.reduce_mean(cs, axes)
    # luminance = math_ops.reduce_mean(luminance, axes)
    return ssim_val, cs, luminance

def custom_ssim(img1, img2):
    max_val = 1
    _, _, checks = _verify_compatible_image_shapes(img1, img2)
    with ops.control_dependencies(checks):
        img1 = array_ops.identity(img1)

    # Need to convert the images to float32.  Scale max_val accordingly so that
    # SSIM is computed correctly.
    max_val = math_ops.cast(max_val, img1.dtype)
    max_val = tf.image.convert_image_dtype(max_val, dtypes.float32)
    img1 = tf.image.convert_image_dtype(img1, dtypes.float32)
    img2 = tf.image.convert_image_dtype(img2, dtypes.float32)
    ssim_per_channel, cs, luminance = _ssim_per_channel(img1, img2, max_val)
    # Compute average over color channels.

    '''Testing'''
    # return math_ops.reduce_mean(ssim_per_channel, [-1]), math_ops.reduce_mean(cs, [-1]), math_ops.reduce_mean(luminance, [-1]))
    cs = math_ops.reduce_mean(cs)
    luminance = math_ops.reduce_mean(luminance)
    return cs * luminance, cs, luminance



def custom_mse(x, y):
    x = tf.keras.backend.flatten(x)
    y = tf.keras.backend.flatten(y)
    mask = tf.cast(x, dtype=tf.bool)
    x = tf.boolean_mask(x, mask)
    y = tf.boolean_mask(y, mask)
    return tf.reshape(tf.math.reduce_mean(tf.math.square((x - y))), [-1])



def custom_mae(x, y):
    x = tf.keras.backend.flatten(x)
    y = tf.keras.backend.flatten(y)
    mask = tf.cast(x, dtype=tf.bool)
    x = tf.boolean_mask(x, mask)
    y = tf.boolean_mask(y, mask)
    return tf.reshape(tf.math.reduce_mean(tf.math.abs((x - y))), [-1])