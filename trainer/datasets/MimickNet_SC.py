import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

""" MimickDataset with Scan Convert """
class MimickDataset_SC():
    def __init__(self, divisible, bs, dataset, data_dir):
        self.divisible = divisible
        self.bs = bs
        self.dataset = dataset
        self.data_dir = data_dir

    def make_dataset(self, dataset_type):
        ds = tfds.load(self.dataset, data_dir=self.data_dir)
        dataset = ds[dataset_type].map(process)
        dataset = dataset.map(scan_convert)

        if (dataset_type == 'train' or dataset_type == 'validation'):
            dataset = dataset.map(make_shape)
            dataset = dataset.map(get_das_and_dtce, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
            dataset = dataset.batch(self.bs).repeat().prefetch(1)
        else:
            dataset = dataset.map(make_divisible, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
            dataset = dataset.batch(1).repeat().prefetch(1)

        return dataset



'''Cuts to -80 dB and normalizes images from 0 to 1'''
def process(ele):
    ele['das'] = tf.reshape(ele['das']['dB'], [ele['height'], ele['width']])
    ele['das'] = tf.clip_by_value(ele['das'], -80, 0)
    ele['das'] = (ele['das'] - tf.reduce_min(ele['das']))/(tf.reduce_max(ele['das']) - tf.reduce_min(ele['das']))

    ele['dtce'] = tf.reshape(ele['dtce'], [ele['height'], ele['width']])
    ele['dtce'] = (ele['dtce'] - tf.reduce_min(ele['dtce']))/(tf.reduce_max(ele['dtce']) - tf.reduce_min(ele['dtce']))
    return ele



'''Scan convert image using Tensorflow's image sparse warp'''
def scan_convert(ele):
    ''' Initialize variables '''
    ''' Math obtained here: https://math.stackexchange.com/questions/332743/calculating-the-coordinates-of-a-point-on-a-circles-circumference-from-the-radiu '''

    height = tf.cast(tf.shape(ele['das'])[0], tf.int32)
    width = tf.cast(tf.shape(ele['das'])[1], tf.int32)

    horizontal_pad = tf.cast((2048 - width) / 2, tf.int32)
    vertical_pad = tf.cast(2048 - height, tf.int32)
    x_segments = 5
    y_segments = 17

    ''' Pad Images with calculated pads to obtain 2048 to 2048 '''
    temp_das = tf.pad(ele['das'], [[vertical_pad, 0], 
                                [horizontal_pad, horizontal_pad]], 
                                "CONSTANT", constant_values = 0)

    temp_dtce = tf.pad(ele['dtce'], [[vertical_pad, 0], 
                                [horizontal_pad, horizontal_pad]], 
                                "CONSTANT", constant_values = 0)

    ''' Obtain start and end points '''
    # Initialize start points
    y = tf.cast(tf.linspace(0, height, y_segments), tf.int32) + vertical_pad
    x = tf.cast(tf.linspace(0, width, x_segments), tf.int32) + horizontal_pad
    x, y = tf.meshgrid(x, y)
    y, x = tf.reshape(y, [-1]), tf.reshape(x, [-1])
    start_points = tf.stack([y,x], axis=-1)
    start_points = tf.cast(tf.round(start_points), tf.float32)

    # Generate end points
    angles = tf.linspace(-1.0, 1.0, x_segments) * ele['final_angle']
    angles = tf.tile(angles , tf.constant([y_segments], tf.int32))
    y_coords = start_points[:,0]
    rad = tf.linspace(ele['initial_radius'], ele['final_radius'], y_segments)
    rad = tf.repeat(rad, x_segments)
    y_converted = y_coords - rad * (1-tf.cos(angles)) # y coordinates conversion
    x_converted = (tf.cast(horizontal_pad, tf.float32) + tf.cast(width/2, tf.float32)) + rad * tf.sin(angles)   # x coordinates conversion
    end_points = tf.stack([y_converted, x_converted], axis=-1)
    end_points = tf.cast(tf.round(end_points), tf.float32)

    # Add batch dimensions for start and end points
    start_points = tf.expand_dims(start_points, axis=0)
    end_points = tf.expand_dims(end_points, axis=0)

    '''Image Sparse Warp (to create scan convert image)'''
    # First ensure padded shape is set as 4096 and 4096 with batch size 1
    # and channels as 1.
    # Shape must be set beforehand or else sparse warp will fail
    temp_das = tf.expand_dims(temp_das, axis=0)
    temp_das = tf.expand_dims(temp_das, axis=3)
    temp_das = tf.ensure_shape(temp_das, shape=(1,2048,2048,1))
    temp_dtce = tf.expand_dims(temp_dtce, axis=0)
    temp_dtce = tf.expand_dims(temp_dtce, axis=3)
    temp_dtce = tf.ensure_shape(temp_dtce, shape=(1,2048,2048,1))

    # Image sparse warp to create scan convert images
    temp_das, _ = tfa.image.sparse_image_warp(temp_das, start_points, end_points)
    temp_dtce, _ = tfa.image.sparse_image_warp(temp_dtce, start_points, end_points)

    ''' Add mask to remove artifacts '''
    height, width = tf.shape(temp_das)[1], tf.shape(temp_das)[2]
    X, Y = tf.meshgrid(tf.linspace(0, width, width), tf.linspace(0, height, height))
    X, Y = tf.cast(X, tf.float32), tf.cast(Y, tf.float32)
    dist_from_center = tf.cast(tf.sqrt((X - tf.cast(width/2, tf.float32))**2 + (Y - (2048.0-ele['final_radius']))**2), tf.float32)
    mask = dist_from_center <= ele['final_radius']
    mask = tf.expand_dims(mask, axis=0)
    mask = tf.expand_dims(mask, axis=3)
    zeros = tf.zeros(shape=(1,height,width,1))
    temp_das = tf.where(mask, temp_das, zeros)
    temp_dtce = tf.where(mask, temp_dtce, zeros)

    ''' Trim off extra areas '''
    maxX = tf.math.multiply(ele['final_radius'], tf.sin(ele['final_angle']))
    minX = tf.math.multiply(ele['final_radius'], tf.sin(ele['initial_angle']))

    temp_das = tf.image.resize_with_crop_or_pad(temp_das, 2048, tf.cast(maxX-minX, tf.int32))
    temp_das = temp_das[:,vertical_pad:,:,:]
    temp_dtce = tf.image.resize_with_crop_or_pad(temp_dtce, 2048, tf.cast(maxX-minX, tf.int32))
    temp_dtce = temp_dtce[:,vertical_pad:,:,:]

    ele['das'] = tf.squeeze(temp_das, axis = 0)
    ele['dtce'] = tf.squeeze(temp_dtce, axis = 0)

    return ele



''' Resize image to a size of 512 by 512, keeping original aspect ratio'''
def make_shape(ele):
    max_dim = 512 
    ele['das'] = tf.image.resize(ele['das'], size=(max_dim,max_dim))
    ele['dtce'] = tf.image.resize(ele['dtce'], size=(max_dim,max_dim))

    return ele



''' Get Delay-and-Sum and DTCE images for training'''
def get_das_and_dtce(ele):
    return ele['das'], ele['dtce']



'''Makes image dimension divisible by divisor, used for model prediction'''
def make_divisible(ele):
    height = tf.cast(tf.shape(ele['das'])[0], tf.int32)
    width = tf.cast(tf.shape(ele['das'])[1], tf.int32)
    height_sub = tf.math.floormod(height, 16)
    width_sub = tf.math.floormod(width, 16)
    ele['das'] = tf.image.resize_with_crop_or_pad(ele['das'], height-height_sub, width-width_sub)
    ele['dtce'] = tf.image.resize_with_crop_or_pad(ele['dtce'], height-height_sub, width-width_sub)
    return ele