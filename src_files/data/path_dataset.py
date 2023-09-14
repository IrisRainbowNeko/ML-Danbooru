from torch.utils.data import Dataset
from PIL import Image

class PathDataset(Dataset):
    def __init__(self, img_list, transform=None):
        self.img_list = img_list
        self.transform = transform

    def __getitem__(self, index):
        path = self.img_list[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, path

    def __len__(self):
        return len(self.img_list)