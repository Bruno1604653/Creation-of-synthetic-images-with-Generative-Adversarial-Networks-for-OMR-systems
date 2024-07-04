# GANWriting para Generación de Símbolos Musicales

Este repositorio contiene todos los scripts y archivos necesarios para entrenar un modelo GAN para la generación de imágenes sintéticas de símbolos musicales manuscritos. El directorio principal de este proyecto es `GANWriting`, que alberga los scripts principales para el entrenamiento del modelo y la gestión de datos.

## Estructura del Directorio

- `main_run.py`: Este script es el punto de entrada para entrenar el modelo GAN. Configura los parámetros de la tasa de aprendizaje, el tamaño del lote y otros parámetros de entrenamiento.
- `read_classes.py`: Este script lee las imágenes y sus clases correspondientes desde los directorios especificados y genera un archivo `.txt` que contiene esta información.
- `load_data.py`: Este script carga los conjuntos de datos basados en el archivo `.txt` generado por `read_classes.py`. Es llamado desde `main_run.py` para crear conjuntos de datos para el entrenamiento.
- `network_tro.py`: Este script gestiona las fases de entrenamiento del modelo a través de la clase `ConTranModel`. Incluye las pasadas hacia adelante y las actualizaciones para el generador, el discriminador y el reconstructor.
- `modules_tro.py`: Este script contiene todos los componentes del modelo, incluidos codificadores, decodificadores y otros módulos de red. Llama a varios otros scripts que contienen diferentes capas de las transformaciones.
- `blocks.py`: Contiene varias estructuras de bloques utilizadas dentro del modelo.
- `loss_tro.py`: Contiene la implementación de las funciones de pérdida utilizadas durante el entrenamiento.
- `clean.py`: Script de utilidad para limpiar y preparar datos.

## Empezando

### Prerrequisitos

Asegúrate de tener las siguientes bibliotecas instaladas:

- PyTorch
- NumPy
- OpenCV
- Scikit-image
- Matplotlib

### Preparación del Conjunto de Datos

Antes de ejecutar el script de entrenamiento, debes preparar tu conjunto de datos:

1. Coloca tus conjuntos de datos en el directorio raíz.
2. Utiliza `read_classes.py` para leer las imágenes y generar un archivo `.txt` que contenga las rutas de las imágenes y sus etiquetas correspondientes.

### Entrenamiento del Modelo

Para entrenar el modelo, ejecuta el siguiente comando:

```bash
python main_run.py <start_epoch>
```

Reemplaza `<start_epoch>` con el número de época desde la que deseas comenzar. Si tienes un modelo guardado previamente, cargará los pesos del modelo desde esa época.

### Configuración

Asegúrate de configurar los directorios en los scripts para que apunten a las ubicaciones correctas de tus datos y donde deseas guardar los resultados. Las configuraciones clave se establecen en `main_run.py`:

- `BATCH_SIZE`: Establece el tamaño del lote para el entrenamiento.
- `lr_dis`, `lr_gen`, `lr_rec`: Tasas de aprendizaje para el discriminador, generador y reconstructor respectivamente.
- `MODEL_SAVE_EPOCH`: Define después de cuántas épocas se deben guardar los pesos del modelo.
- `EVAL_EPOCH`: Define después de cuántas épocas se debe evaluar el modelo.

### Ejemplo de Uso

```bash
python main_run.py 0
```

Esto iniciará el entrenamiento del modelo desde la época 0. Si tienes modelos preentrenados, puedes comenzar desde una época diferente especificando el número correspondiente.

### Importancia de la Configuración de Directorios

Es crucial configurar correctamente los directorios de los que se leerán los datos y donde se guardarán los resultados. Asegúrate de revisar y ajustar las rutas en los scripts antes de iniciar el entrenamiento.

## Integración con Mashcima

Las imágenes sintéticas generadas por el modelo GAN entrenado pueden integrarse con el sistema Mashcima para su uso en sistemas de Reconocimiento Óptico de Música (OMR). El proceso de integración implica utilizar las imágenes generadas para aumentar los conjuntos de datos de entrenamiento y evaluar el impacto en el rendimiento del OMR.

## MusicSymbolClassifier

En el directorio raíz también encontrarás varios archivos y ficheros relacionados con `MusicSymbolClassifier`. Este es otro repositorio que puedes utilizar para entrenar un modelo que clasifique los símbolos musicales. Este proyecto incluye clasificadores avanzados basados en arquitecturas como VGG y ResNet, y es una herramienta complementaria útil para mejorar el rendimiento de los sistemas de OMR.

El archivo `MusicSymbolClassifier_master.ipynb` proporciona un notebook de Jupyter que entrena clasificadores específicos para OMR utilizando los datos generados. Es recomendable realizar las siguientes pruebas:

1. Entrenar el clasificador con los datasets originales (sin datos sintéticos).
2. Entrenar el clasificador con los datos sintéticos generados.
3. Entrenar el clasificador con una combinación de datos originales y sintéticos.

Estas pruebas ayudarán a evaluar el impacto de los datos sintéticos en el rendimiento del sistema de OMR.