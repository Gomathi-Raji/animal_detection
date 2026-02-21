import tensorflow as tf
import os

model = tf.keras.models.load_model("DualNet_CX/audio_classifier.h5", compile=False)

save_folder = "tflite_models/DualNet_CX"
os.makedirs(save_folder, exist_ok=True)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Define the file path
tflite_model_path = os.path.join(save_folder, "audio_classifier.tflite")

# Save the converted model in the specified folder
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"Conversion successful! Saved as {tflite_model_path}")
