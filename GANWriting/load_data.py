import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

MUSICAL_SYMBOLS = ['Other', 'Quarter-Note', 'Quarter-Rest', 'Repeat-Measure', 'Segno', 'Sharp', 'Sixteenth-Note', 'Sixteenth-Rest', 'Sixty-Four-Note', 'Sixty-Four-Rest', 'Staccatissimo', 'Stopped', 'Tenuto', 'Thirty-Two-Note', 'Thirty-Two-Rest', 'Tie-Slur', 'Trill', 'Trill-Wobble', 'Tuplet', 'Turn', 'Volta', 'Whole-Half-Rest', 'Whole-Note', 'Test', 'Training', 'Validation', 'Accent', 'Barline', 'Beam', 'C-Clef', 'Common-Time', 'Cut-Time', 'Dot', 'Eighth-Grace-Note', 'Eighth-Note', 'Eighth-Rest', 'F-Clef', 'Flat', 'G-Clef', 'Half-Note', 'Multiple-Eighth-Notes', 'Multiple-Half-Notes', 'Multiple-Quarter-Notes', 'Multiple-Sixteenth-Notes', 'Natural', '1-8-Time', '12-8-Time', '2-4-Time', '2-8-Time', '3-4-Time', '3-8-Time', '4-2-Time', '4-4-Time', '4-8-Time', '5-4-Time', '5-8-Time', '6-4-Time', '6-8-Time', '7-4-Time', '8-8-Time', '9-8-Time', 'Breve', 'Chord', 'Double-Whole-Rest', 'Fermata', 'Glissando', 'Marcato', 'Mordent', 'Bass', 'Crotchet', 'Demisemiquaver_Line', 'Minim', 'Quaver_Br', 'Quaver_Line', 'Quaver_Tr', 'Semibreve', 'Semiquaver_Br', 'Semiquaver_Line', 'Semiquaver_Tr', 'Treble']

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
NUM_CHANNEL = 1

class MusicSymbolDataset(Dataset):
    def __init__(self, data_dirs, transform=None):
        self.data_dirs = data_dirs
        self.transform = transform or transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
        self.data = []
        for data_dir in data_dirs:
            print(f"Procesando directorio: {data_dir}")
            if not os.path.exists(data_dir):
                print(f"Directorio no existe: {data_dir}")
                continue
            for symbol in MUSICAL_SYMBOLS:
                symbol_dir = os.path.join(data_dir, symbol)
                if os.path.exists(symbol_dir):
                    print(f"Existente: {symbol_dir}")
                    png_count = 0
                    for img_file in os.listdir(symbol_dir):
                        print(f"Encontrado archivo: {img_file}")
                        if img_file.lower().endswith('.png'):
                            png_count += 1
                            self.data.append((os.path.join(symbol_dir, img_file), tokens[symbol]))
                            print(f"Añadido: {os.path.join(symbol_dir, img_file)}")
                    print(f"Archivos .png encontrados en {symbol_dir}: {png_count}")
                else:
                    print(f"Directorio no encontrado para símbolo: {symbol_dir}")
        print(f"Total de imágenes encontradas: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label

def loadData(oov, directories=None, batch_size=128, num_workers=0):
    if directories is None:
        directories = ['./dataset1/dataset1', './dataset2/dataset2','./data/open_omr_raw','./data/images', './data/open_pp_raw']
    
    train_dataset = MusicSymbolDataset(directories)
    test_dataset = MusicSymbolDataset(directories)

    if len(train_dataset) == 0:
        raise ValueError("El dataset de entrenamiento está vacío")

    if len(test_dataset) == 0:
        raise ValueError("El dataset de prueba está vacío")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

# Ejemplo de uso
train_loader, test_loader = loadData(oov=True)
