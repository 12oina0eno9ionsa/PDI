import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
import os

# Directorio de datos de validación
validation_data_dir = r"D:\PDI\annotatted\SAMPLES\mixed_fouling"

# Parámetros
img_height, img_width = 256, 256
batch_size = 32

# Cargar el modelo entrenado
model_path = 'simple_cnn_model_final.keras'
model = tf.keras.models.load_model(model_path)

# Generador de datos para validación
test_datagen = ImageDataGenerator(rescale=1.0/255)

test_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Realizar predicciones y mostrar los resultados
class_names = ['high', 'low', 'mid']

image_files = [os.path.join(validation_data_dir, f) for f in os.listdir(validation_data_dir) if f.lower().endswith(('.jpg', '.png'))]
print(f"Image files found: {len(image_files)}")

for image_file in image_files:
    img = load_img(image_file, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    print(f"Image: {image_file}")
    print(f"Predicted: {class_names[predicted_class]} (confidence: {confidence:.3f})")
    print("Class probabilities:")
    for cls, prob in zip(class_names, prediction[0]):
        print(f"  {cls}: {prob:.3f}")