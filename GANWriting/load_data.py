import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

file_path = './all_images.txt'

def load_musical_symbols(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no existe")

    musical_symbols = set()
    with open(file_path, 'r') as file:
        for line in file:
            symbol = line.strip().split('\t')[0]
            musical_symbols.add(symbol.lower())

    return sorted(musical_symbols)

# Crear la lista de símbolos musicales
MUSICAL_SYMBOLS = load_musical_symbols(file_path)

# Crear el diccionario de símbolos musicales
MUSICAL_SYMBOLS_DICT = {symbol: idx for idx, symbol in enumerate(MUSICAL_SYMBOLS)}

IMG_HEIGHT = 128
IMG_WIDTH = 128

tokens = {
    'PAD_TOKEN': 0,
    'SOS_TOKEN': 1,
    'EOS_TOKEN': 2,
    'UNK_TOKEN': 3
}

# Clases de interés
TARGET_CLASSES = {'sharp', 'quarter-rest', 'tie-slur', 'quarter-note'}

# Agregar las clases de interés a los tokens
for i, symbol in enumerate(TARGET_CLASSES):
    tokens[symbol] = i + 4

index2letter = {v: k for k, v in tokens.items()}
vocab_size = len(tokens)
num_tokens = 4
NUM_CHANNEL = 3

print(f"vocab_size: {vocab_size}")

class MusicSymbolDataset(Dataset):
    def __init__(self, data_dirs, target_classes, transform=None):
        global tokens
        self.data_dirs = data_dirs
        self.target_classes = target_classes
        self.transform = transform or transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
        self.data = []
        self.classes = []
        for data_dir in data_dirs:
            print(f"Procesando directorio: {data_dir}")
            if not os.path.exists(data_dir):
                print(f"Directorio no existe: {data_dir}")
                continue
            for symbol in os.listdir(data_dir):
                symbol_dir = os.path.join(data_dir, symbol)
                symbol_lower = symbol.lower()
                if symbol_lower in self.target_classes and os.path.isdir(symbol_dir):
                    if symbol_lower not in tokens:
                        tokens[symbol_lower] = len(tokens)
                    if symbol_lower not in self.classes:
                        self.classes.append(symbol_lower)
                    print(f"Existente: {symbol_dir}")
                    png_count = 0
                    for img_file in os.listdir(symbol_dir):
                        if img_file.lower().endswith('.png'):
                            png_count += 1
                            self.data.append((os.path.join(symbol_dir, img_file), tokens[symbol_lower]))
                            print(f"Añadido: {os.path.join(symbol_dir, img_file)}")
                    print(f"Archivos .png encontrados en {symbol_dir}: {png_count}")
                else:
                    print(f"Clase no encontrada o no es un directorio: {symbol_lower}")
        print(f"Ejemplo data: {self.data[0]}")
        print(f"tokens: {tokens}")
        print(f"Total de imágenes encontradas: {len(self.data)}")
        print(f"Clases encontradas: {self.classes}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('L')
        image = self.transform(image)
        return image, label

def loadData(oov, directories=None, batch_size=128, num_workers=0, test_split_ratio=0.2):
    if directories is None:
        directories = ['/data2fast/users/bfajardo/datasets/dataset1/dataset1', '/data2fast/users/bfajardo/datasets/dataset2/dataset2',
'/data2fast/users/bfajardo/datasets/data/images','/data2fast/users/bfajardo/datasets/data/muscima_pp_raw','/data2fast/users/bfajardo/datasets/data/open_omr_raw']


    dataset = MusicSymbolDataset(directories, TARGET_CLASSES)

    if len(dataset) == 0:
        raise ValueError("El dataset está vacío")

    # Dividir el dataset en entrenamiento y prueba
    test_size = int(test_split_ratio * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print(f"\nlen(train_dataset): {len(train_dataset)}")
    print(f"\nlen(test_dataset): {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
