import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

MUSICAL_SYMBOLS = ['1-8-time', '12-8-time', '2-4-time', '2-8-time', '3-4-time', '3-8-time', '4-2-time', '4-4-time', '4-8-time', '5-4-time', '5-8-time', '6-4-time', '6-8-time', '7-4-time', '8-8-time', '9-8-time', 'accent', 'barline', 'bass', 'beam', 'breve', 'c-clef', 'chord', 'common-time','crotchet', 'cut-time', 'demisemiquaver_line', 'dot', 'double-whole-rest', 'eighth-grace-note', 'eighth-note', 'eighth-rest', 'f-clef', 'fermata','flat', 'g-clef', 'glissando', 'half-note', 'marcato', 'minim', 'mordent', 'multiple-eighth-notes', 'multiple-half-notes', 'multiple-quarter-notes', 'multiple-sixteenth-notes', 'natural', 'other', 'quarter-note', 'quarter-rest', 'quaver_br', 'quaver_line', 'quaver_tr', 'repeat-measure', 'segno', 'semibreve', 'semiquaver_br', 'semiquaver_line', 'semiquaver_tr', 'sharp', 'sixteenth-note', 'sixteenth-rest', 'sixty-four-note', 'sixty-four-rest', 'staccatissimo', 'stopped', 'tenuto', 'test', 'thirty-two-note', 'thirty-two-rest', 'tie-slur', 'training', 'treble', 'trill', 'trill-wobble', 'tuplet', 'turn', 'v2.0', 'validation', 'volta', 'whole-half-rest', 'whole-note']

MUSICAL_SYMBOLS_DICT = {symbol: idx for idx, symbol in enumerate(MUSICAL_SYMBOLS)}

IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_SIZE = 128

tokens = {
    'PAD_TOKEN': 0,
    'SOS_TOKEN': 1,
    'EOS_TOKEN': 2,
    'UNK_TOKEN': 3
}

for i, symbol in enumerate(MUSICAL_SYMBOLS):
    tokens[symbol] = i + 4

index2letter = {v: k for k, v in tokens.items()}
vocab_size = len(tokens)
num_tokens = 4
NUM_CHANNEL = 3
print(f"vocab_size: {vocab_size}")
class MusicSymbolDataset(Dataset):
    def __init__(self, data_dirs, transform=None):
        global _tokens
        self.data_dirs = data_dirs
        self.transform = transform or transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
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
                if os.path.isdir(symbol_dir):
                    symbol_lower = symbol.lower()
                    if symbol_lower not in _tokens:
                        _tokens[symbol_lower] = len(_tokens)
                    if symbol_lower not in self.classes:
                        self.classes.append(symbol_lower)
                    print(f"Existente: {symbol_dir}")
                    png_count = 0
                    for img_file in os.listdir(symbol_dir):
                        if img_file.lower().endswith('.png'):
                            png_count += 1
                            label = _tokens[symbol_lower]
                            if label >= vocab_size:
                                print(f"Warning: Label {label} for {symbol_lower} exceeds vocab size {vocab_size}")
                                continue
                            self.data.append((os.path.join(symbol_dir, img_file), label))
                            print(f"Añadido: {os.path.join(symbol_dir, img_file)}")
                    print(f"Archivos .png encontrados en {symbol_dir}: {png_count}")
                else:
                    print(f"Directorio no encontrado para símbolo: {symbol_dir}")
        print(f"Total de imágenes encontradas: {len(self.data)}")
        print(f"Clases encontradas: {self.classes}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')  # Convertir a RGB
        image = self.transform(image)
        return image, label


def loadData(oov, directories=None, batch_size=128, num_workers=0):
    if directories is None:
        directories = directories = ['./dataset1/dataset1', './dataset2/dataset2', './data/open_omr_raw', './data/images', './data/muscima_pp_raw']

    train_dataset = MusicSymbolDataset(directories)
    test_dataset = MusicSymbolDataset(directories)

    if len(train_dataset) == 0:
        raise ValueError("El dataset de entrenamiento está vacío")

    if len(test_dataset) == 0:
        raise ValueError("El dataset de prueba está vacío")
    print(f"\nlen(train_dataset): {len(train_dataset)}")
    print(f"\nlen(test_dataset): {len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
