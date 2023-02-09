import torch
from torch.utils.data import Dataset
import json
from PIL import Image
from PIL import ImageFile
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 2300000000

class Danbooru(Dataset):
    def __init__(self, root, annFile, num_class=53109, transform=None, file_ext=None):
        self.root = root
        self.transform = transform
        self.num_class = num_class
        self.file_ext = file_ext

        with open(annFile, 'r', encoding='utf8') as f:
            self.labels = json.loads(f.read())

    def __getitem__(self, index):
        item = self.labels[index]

        img_path = os.path.join(self.root, str(item[0] % 1000).zfill(4), f'{item[0]}.{self.file_ext if self.file_ext else item[1]}')

        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        target = torch.zeros(self.num_class, dtype=torch.float)
        target[item[2]]=1.

        return img, target

    def __len__(self):
        return len(self.labels)