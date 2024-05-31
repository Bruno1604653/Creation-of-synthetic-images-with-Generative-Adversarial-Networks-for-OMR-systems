import os
from PIL import Image

# Directorios de los datasets
directories = [
    './dataset1',
    './dataset2',
    './data/open_omr_raw', 
    './data/images',
    './data/muscima_pp_raw'
]

# Función para listar todas las imágenes en los directorios especificados
def list_images_in_directories(directories):
    all_images = {}
    for data_dir in directories:
        if not os.path.exists(data_dir):
            print(f"Directorio no existe: {data_dir}")
            continue
        print(f"Procesando directorio: {data_dir}")
        for class_name in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                class_lower = class_name.lower()
                if class_lower not in all_images:
                    all_images[class_lower] = []
                png_count = 0
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith('.png'):
                        img_path = os.path.join(class_dir, img_file)
                        all_images[class_lower].append(img_path)
                        png_count += 1
                print(f"Archivos .png encontrados en {class_dir}: {png_count}")
            else:
                print(f"Directorio no encontrado para clase: {class_dir}")
    return all_images

# Ejecutar la función y obtener todas las imágenes
all_images = list_images_in_directories(directories)

# Mostrar el total de imágenes encontradas por clase
for class_name, images in all_images.items():
    print(f"Clase: {class_name}, Total de imágenes: {len(images)}")

# Opcional: Guardar la lista de imágenes en un archivo de texto
output_file = 'all_images.txt'
with open(output_file, 'w') as f:
    for class_name, images in all_images.items():
        for img_path in images:
            f.write(f"{class_name}\t{img_path}\n")
    print(f"Lista de imágenes guardada en {output_file}")
