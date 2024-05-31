import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

MUSICAL_SYMBOLS = ['other', 'quarter-note', 'quarter-rest', 'repeat-measure', 'segno', 'sharp', 'sixteenth-note', 'sixteenth-rest', 'sixty-four-note', 'sixty-four-rest', 'staccatissimo', 'stopped', 'tenuto', 'thirty-two-note', 'thirty-two-rest', 'tie-slur', 'trill', 'trill-wobble', 'tuplet', 'turn', 'volta', 'whole-half-rest', 'whole-note', 'test', 'training', 'validation', 'accent', 'barline', 'beam', 'c-clef', 'common-time', 'cut-time', 'dot', 'eighth-grace-note', 'eighth-note', 'eighth-rest', 'f-clef', 'flat', 'g-clef', 'half-note', 'multiple-eighth-notes', 'multiple-half-notes', 'multiple-quarter-notes', 'multiple-sixteenth-notes', 'natural', '1-8-time', '12-8-time', '2-4-time', '2-8-time', '3-4-time', '3-8-time', '4-2-time', '4-4-time', '4-8-time', '5-4-time', '5-8-time', '6-4-time', '6-8-time', '7-4-time', '8-8-time', '9-8-time', 'breve', 'chord', 'double-whole-rest', 'fermata', 'glissando', 'marcato', 'mordent', 'bass', 'crotchet', 'demisemiquaver_line', 'minim', 'quaver_br', 'quaver_line', 'quaver_tr', 'semibreve', 'semiquaver_br', 'semiquaver_line', 'semiquaver_tr', 'treble']

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
            for symbol in MUSICAL_SYMBOLS:
                symbol_dir = os.path.join(data_dir, symbol)
                if os.path.exists(symbol_dir):
                    for img_file in os.listdir(symbol_dir):
                        if img_file.endswith('.png'):
                            self.data.append((os.path.join(symbol_dir, img_file), tokens[symbol]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        #print(f"Loaded image shape: {image.shape}")  # Add this line to print the shape of each loaded image
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
    # print(f"\nlen(train_dataset): {len(train_dataset)}")
    # print(f"\nlen(test_dataset): {len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
