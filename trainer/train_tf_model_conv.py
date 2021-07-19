import tensorflow as tf
from config import *
from utils import losses
from utils.MimickNet_Dataset import MimickDataset
from models.MimickNet_Conv import MimickNet_Conv

# Load Dataset
mimick = MimickDataset(divisible=16, bs=config.bs, dataset='duke_ultrasound', data_dir='gs://tfds-data/datasets')

# Information taken from https://www.tensorflow.org/datasets/catalog/duke_ultrasound
train_count = 2556 * config.bs/2
test_count = 438 * config.bs/2
val_count = 278 * config.bs/2

train_dataset = mimick.make_dataset(dataset_type='train')
validation_dataset = mimick.make_dataset(dataset_type='validation')
test_dataset = mimick.make_dataset(dataset_type='test')

# Load Model
# Reset tf session
tf.keras.backend.clear_session()

model = MimickNet_Conv(shape=(None,None,1),Activation=tf.keras.layers.ReLU(),filters=[16,16,16,16,16], filter_shape=(3,3)).load_model()

model.compile(optimizer=tf.keras.optimizers.Adam(0.002), loss='mae', 
              metrics=[losses.mae, losses.mse, losses.ssim, losses.psnr])

# Fit model
model.fit(train_dataset,
          steps_per_epoch=int(train_count/config.bs),
          epochs=int(config.epochs/5),
          validation_data=validation_dataset,
          validation_steps=int(val_count/config.bs),
          verbose=1)

# Save model
model.save("./trained_models/model_noSC_Conv_v1.h5")