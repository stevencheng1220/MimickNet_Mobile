import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ReLU, MaxPool2D, Activation, Dropout, Conv2D, Conv2DTranspose

class MimickNet_Conv():
    def __init__(self, shape=(None, None, 1),
                   Activation=tf.keras.layers.ReLU(),
                   filters=[16, 16, 16, 16, 16],
                   filter_shape=(3, 3)):
        self.shape = shape
        self.Activation = Activation
        self.filters = filters
        self.filter_shape = filter_shape
        
    def load_model(self):
        inputs = Input(shape=self.shape)
        x = inputs
        downsample_path = []
        
        filters = self.filters
        filter_shape = self.filter_shape

        for idx, filter_num in enumerate(filters):
          x = Conv2D(filter_num, filter_shape, padding='same', kernel_regularizer=None)(x)
          x = Activation(activation=tf.nn.relu)(x)
          x = Conv2D(filter_num, filter_shape, padding='same', kernel_regularizer=None)(x)
          x = Activation(activation=tf.nn.relu)(x)
          if idx != len(filters)-1:
            downsample_path.append(x)
            x = MaxPool2D(padding='same')(x)

        downsample_path.reverse()
        reverse_filters = list(filters[:-1])
        reverse_filters.reverse()

        for idx, filter_num in enumerate(reverse_filters):
          x = Conv2DTranspose(filter_num, 2, 2, padding='same')(x)
          x = tf.keras.layers.concatenate([x, downsample_path[idx]])
          x = Conv2D(filter_num, filter_shape, padding='same', kernel_regularizer=None)(x)
          x = Activation(activation=tf.nn.relu)(x)
          x = Conv2D(filter_num, filter_shape, padding='same', kernel_regularizer=None)(x)
          x = Activation(activation=tf.nn.relu)(x)

        x = Conv2D(1, 1)(x)

        return Model(inputs, x)