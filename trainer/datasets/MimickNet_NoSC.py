import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa


""" MimickDataset with No Scan Convert """
class MimickDataset_NoSC():
    def __init__(self, divisible, bs, dataset, data_dir):
        self.divisible = divisible
        self.bs = bs
        self.dataset = dataset
        self.data_dir = data_dir

    def make_dataset(self, dataset_type):
        ds = tfds.load(self.dataset, data_dir=self.data_dir)
        dataset = ds[dataset_type]
        dataset = dataset.map(process)
        if (dataset_type == 'train' or dataset_type == 'validation'):
            dataset = dataset.map(make_shape_to_dimension, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
            dataset = dataset.batch(self.bs).repeat().prefetch(1)
        else:
            dataset = dataset.map(make_divisible, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
            dataset = dataset.batch(1).repeat().prefetch(1)
        return dataset
    


''' Cuts to -80 dB and normalizes images from 0 to 1 '''
def process(ele):
    ele['das'] = tf.reshape(ele['das']['dB'], [ele['height'], ele['width']])
    ele['das'] = tf.clip_by_value(ele['das'], -80, 0)
    ele['das'] = (ele['das'] - tf.reduce_min(ele['das']))/(tf.reduce_max(ele['das']) - tf.reduce_min(ele['das']))

    ele['dtce'] = tf.reshape(ele['dtce'], [ele['height'], ele['width']])
    ele['dtce'] = (ele['dtce'] - tf.reduce_min(ele['dtce']))/(tf.reduce_max(ele['dtce']) - tf.reduce_min(ele['dtce']))
    return ele



''' Pad or crop input image until it becomes input shape of max_dimension '''
def make_shape_to_dimension(ele):
    # Initialize variables
    max_dim = 512 
    height = tf.cast(tf.shape(ele['das'])[0], tf.int32)
    width = tf.cast(tf.shape(ele['das'])[1], tf.int32)
    das = ele['das']
    dtce = ele['dtce']

    # Continuously crop or reflect image until we obtain an image with dimensions max_dimension
    i = height
    while tf.less(i, max_dim):
        das = tf.pad(das, [[height-1,0],[0,0]],"REFLECT")
        dtce = tf.pad(dtce, [[height-1,0],[0,0]],"REFLECT")
        i = tf.add(i, height-1)

    i = width
    while tf.less(i, max_dim):
        das = tf.pad(das, [[0,0],[width-1,width-1]],"REFLECT")
        dtce = tf.pad(dtce, [[0,0],[width-1,width-1]],"REFLECT")
        i = tf.add(i, width-1)
  
    # Crop excess of image until we get an image with dimensions max_dimension
    das = tf.expand_dims(das, axis=2)
    dtce = tf.expand_dims(dtce, axis=2)
    centerX = tf.cast(tf.shape(das)[1] / 2, tf.int32)
    temp_height = tf.cast(tf.shape(das)[0], tf.int32)

    das = tf.image.crop_to_bounding_box(das, temp_height - max_dim, centerX - tf.cast(max_dim/2, tf.int32), max_dim, max_dim)
    dtce = tf.image.crop_to_bounding_box(dtce, temp_height - max_dim, centerX - tf.cast(max_dim/2, tf.int32), max_dim, max_dim)
    return das, dtce



''' Makes image dimension divisible by divisor, used for model prediction '''
def make_divisible(ele):
    height = tf.cast(tf.shape(ele['das'])[0], tf.int32)
    width = tf.cast(tf.shape(ele['das'])[1], tf.int32)
    height_sub = tf.math.floormod(height, 16)
    width_sub = tf.math.floormod(width, 16)
    ele['das'] = tf.expand_dims(ele['das'], axis = 2)
    ele['dtce'] = tf.expand_dims(ele['dtce'], axis = 2)
    ele['das'] = tf.image.resize_with_crop_or_pad(ele['das'], height-height_sub, width-width_sub)
    ele['dtce'] = tf.image.resize_with_crop_or_pad(ele['dtce'], height-height_sub, width-width_sub)
    return ele