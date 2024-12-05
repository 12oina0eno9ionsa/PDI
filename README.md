# PDI


Dataset -  https://drive.google.com/drive/folders/1RIBPXS-eiM-_OeltnxCJOGiMaH23_FG6?usp=drive_link

Modelos -  https://drive.google.com/drive/folders/1Tj7p0rXY0lhN9jCzMiOeITaEoghN8TN2?usp=drive_link


El modelo ECA-CNN es ideal para clasificar fouling en tres niveles (High, Mid, Low) debido a su eficiencia y capacidad para enfocarse en características críticas mediante el módulo de atención eficiente (ECA). Este módulo prioriza detalles relevantes, como texturas y patrones, esenciales en la discriminación de niveles de fouling, mientras minimiza la carga computacional al evitar la reducción de dimensionalidad.

Dropout previene el sobreajuste al desactivar aleatoriamente neuronas durante el entrenamiento, mejorando la generalización del modelo, especialmente en un dataset con variabilidad, como las imágenes submarinas. Por su parte, la regularización L2 controla la magnitud de los pesos, reduciendo la sensibilidad al ruido y evitando que el modelo dependa excesivamente de características irrelevantes.

Combinado con técnicas de aumento de datos (rotaciones, zoom, desplazamientos), optimizadores avanzados como Adam, y callbacks como Early Stopping, el modelo es capaz de generalizar bien a nuevas imágenes. ECA-CNN, con su arquitectura eficiente y capacidad de atención, ofrece una solución robusta y escalable para monitorear y gestionar el fouling de mallas submarinas.
