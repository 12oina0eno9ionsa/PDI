import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directorio de datos de validación
validation_data_dir = r"D:\PDI\annotatted\SAMPLES\dataset_4_12\data"

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

# Evaluar el modelo
results = model.evaluate(test_generator)
print(f"Test Loss: {results[0]}")
print(f"Test Accuracy: {results[1]}")