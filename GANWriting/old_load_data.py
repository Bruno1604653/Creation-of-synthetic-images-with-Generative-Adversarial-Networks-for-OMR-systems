import os
import random
import numpy as np
import torch
import torch.utils.data as D
from PIL import Image
import torchvision.transforms as transforms

# Configuración de los parámetros
IMG_HEIGHT = 512
IMG_WIDTH = 512
IMG_SIZE = 512
CHANNELS = 1
N_CLASSES = 80
LATENT_DIM = 100
MAX_CHARS = 10
NUM_CHANNEL = 50
EXTRA_CHANNEL = NUM_CHANNEL + 1
NORMAL = True
OUTPUT_MAX_LEN = MAX_CHARS + 2
tokens = {'GO_TOKEN': 0, 'END_TOKEN': 1, 'PAD_TOKEN': 2}

# Definir la lista de símbolos musicales como nombres de clases
MUSICAL_SYMBOLS = ['other', 'quarter-note', 'quarter-rest', 'repeat-measure', 'segno', 'sharp',
                   'sixteenth-note', 'sixteenth-rest', 'sixty-four-note', 'sixty-four-rest',
                   'staccatissimo', 'stopped', 'tenuto', 'thirty-two-note', 'thirty-two-rest',
                   'tie-slur', 'trill', 'trill-wobble', 'tuplet', 'turn', 'volta', 'whole-half-rest',
                   'whole-note', 'test', 'training', 'validation', 'accent', 'barline', 'beam',
                   'c-clef', 'common-time', 'cut-time', 'dot', 'eighth-grace-note', 'eighth-note',
                   'eighth-rest', 'f-clef', 'flat', 'g-clef', 'half-note', 'multiple-eighth-notes',
                   'multiple-half-notes', 'multiple-quarter-notes', 'multiple-sixteenth-notes',
                   'natural', '1-8-time', '12-8-time', '2-4-time', '2-8-time', '3-4-time', '3-8-time',
                   '4-2-time', '4-4-time', '4-8-time', '5-4-time', '5-8-time', '6-4-time', '6-8-time',
                   '7-4-time', '8-8-time', '9-8-time', 'breve', 'chord', 'double-whole-rest',
                   'fermata', 'glissando', 'marcato', 'mordent', 'bass', 'crotchet', 'demisemiquaver_line',
                   'minim', 'quaver_br', 'quaver_line', 'quaver_tr', 'semibreve', 'semiquaver_br',
                   'semiquaver_line', 'semiquaver_tr', 'treble']

def labelDictionary(symbols):
    symbol2index = {symbol: idx for idx, symbol in enumerate(symbols)}
    index2letter = {v: k for k, v in symbol2index.items()}
    return len(symbols), symbol2index, index2letter

# Crear diccionarios para símbolos musicales
num_classes, SYMBOL2INDEX, INDEX2LETTER = labelDictionary(MUSICAL_SYMBOLS)

class MusicSymbolsDataset(D.Dataset):
    def __init__(self, directories, transform=None):
        self.transform = transform
        self.image_labels = []
        self.class_names = MUSICAL_SYMBOLS
        for directory in directories:
            if not os.path.exists(directory):
                print(f"Directory {directory} does not exist.")
                continue
            self.load_images_and_labels(directory)

    def load_images_and_labels(self, root_dir):
        for subdir in sorted(os.listdir(root_dir)):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                if subdir.lower() not in self.class_names:
                    self.class_names.append(subdir.lower())
                class_index = self.class_names.index(subdir.lower())
                for image_filename in os.listdir(subdir_path):
                    image_path = os.path.join(subdir_path, image_filename)
                    if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_labels.append((image_path, class_index))

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, index):
        img_path, label = self.image_labels[index]
        img = Image.open(img_path).convert('L')
        if self.transform:
            img = self.transform(img)
        img = torch.tensor(img, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        img = img.unsqueeze(0)  # Asegúrate de que img tenga la forma [1, 96, 96]
        return img, img.shape[-1], label

    def label_padding(self, labels, symbol2index):
        new_label_len = []
        ll = [symbol2index[i] for i in labels]
        new_label_len.append(len(ll) + 2)
        ll = np.array(ll) + len(tokens)
        ll = list(ll)
        ll = [tokens['GO_TOKEN']] + ll + [tokens['END_TOKEN']]
        num = OUTPUT_MAX_LEN - len(ll)
        if not num == 0:
            ll.extend([tokens['PAD_TOKEN']] * num)
        return ll

    def new_ed1(self, symbol_ori):
        symbol = symbol_ori.copy()
        start = symbol.index(tokens['GO_TOKEN'])
        fin = symbol.index(tokens['END_TOKEN'])
        symbol = ''.join([INDEX2LETTER[i - len(tokens)] for i in symbol[start + 1: fin]])
        new_symbol = edits1(symbol)
        label = np.array(self.label_padding(new_symbol, SYMBOL2INDEX))
        return label

def edits1(symbol, min_len=2, max_len=MAX_CHARS):
    "All edits that are one edit away from `symbol`."
    symbols = MUSICAL_SYMBOLS
    splits = [(symbol[:i], symbol[i:]) for i in range(len(symbol) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in symbols]
    inserts = [L + c + R for L, R in splits for c in symbols]
    if len(symbol) <= min_len:
        return random.choice(list(set(transposes + replaces + inserts)))
    elif len(symbol) >= max_len:
        return random.choice(list(set(deletes + transposes + replaces)))
    else:
        return random.choice(list(set(deletes + transposes + replaces + inserts)))

def loadData(directories, oov):
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = MusicSymbolsDataset(directories=directories, transform=transform)

    train_size = int(0.7 * len(dataset))
    valid_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size], generator=torch.Generator().manual_seed(123))

    batch_size = 256

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, valid_loader, test_loader
