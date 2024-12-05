# evaluate_image.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import time

def evaluate_image(model_path, image_path, img_height=128, img_width=128):
    # Cargar el modelo
    model = tf.keras.models.load_model(model_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Cargar y preprocesar la imagen
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalizar la imagen

    # Medir el tiempo de inferencia
    start_time = time.time()
    prediction = model.predict(img_array)
    inference_time = time.time() - start_time

    # Obtener la clase predicha y las probabilidades
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_probabilities = prediction[0]
    confidence = class_probabilities[predicted_class]

    return {
        'predicted_class': predicted_class,
        'class_probabilities': class_probabilities,
        'confidence': confidence,
        'inference_time': inference_time
    }

if __name__ == "__main__":
    model_path = r"D:\PDI\CNN\fouling_cnn_model.h5"
    image_path = r"D:\PDI\annotatted\SAMPLES\mixed_fouling\example_image.jpg"
    result = evaluate_image(model_path, image_path)
    print(result)