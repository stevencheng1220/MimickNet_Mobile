import tensorflow as tf
from config import *

model_pwd = "trained_models/model_noSC_Conv_v1"

# Load model
# Reset tf session
tf.keras.backend.clear_session()
model = tf.keras.models.load_model(f"{model_pwd}.h5")

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open(f"{model_pwd}.tflite", 'wb') as f:
  f.write(tflite_model)