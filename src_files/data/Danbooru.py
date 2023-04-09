import torch
from torch.utils.data import Dataset
import json
from PIL import Image
from PIL import ImageFile
import cv2
import os
import numpy as np
from copy import deepcopy, copy

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
        self.data_len = len(self.labels)

        self.skip_num=0
        self.arb=None

    def make_arb(self, arb=None, bs=None):
        self.arb = arb
        self.bs = bs
        rs = np.random.RandomState(42)

        self.idx_imgid_map={item[0]:i for i,item in enumerate(self.labels)}

        with open(arb, 'r', encoding='utf8') as f:
            self.bucket_dict = json.loads(f.read())

        # make len(bucket)%bs==0
        self.data_len = 0
        self.bucket_list=[]
        for k,v in self.bucket_dict.items():
            bucket=copy(v)
            rest=len(v)%bs
            if rest>0:
                bucket.extend(rs.choice(v, bs-rest))
            self.data_len += len(bucket)
            self.bucket_list.append(np.array(bucket))

    def rest_arb(self, epoch):
        rs = np.random.RandomState(42+epoch)
        bucket_list = [copy(x) for x in self.bucket_list]
        #shuffle inter bucket
        for x in bucket_list:
            rs.shuffle(x)

        # shuffle of batches
        bucket_list=np.hstack(bucket_list).reshape(-1, self.bs)
        rs.shuffle(bucket_list)

        self.labels_arb = bucket_list.reshape(-1)

    def __getitem__(self, index):
        if index<self.skip_num:
            return torch.Tensor(0), torch.Tensor(0)

        if self.arb:
            item = self.labels[self.idx_imgid_map[self.labels_arb[index]]] # index -> imgid -> label_idx
        else:
            item = self.labels[index]

        img_path = self.get_path(item)
        img = self.get_image(img_path)

        target = torch.zeros(self.num_class, dtype=torch.float)
        target[item[2]]=1.

        return img, target

    def set_skip_imgs(self, skip_num):
        self.skip_num=skip_num

    def get_path(self, item):
        return os.path.join(self.root, str(item[0] % 1000).zfill(4), f'{item[0]}.{self.file_ext if self.file_ext else item[1]}')

    def get_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return self.data_len