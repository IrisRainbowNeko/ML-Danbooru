import os
import random
import time
from copy import deepcopy

import numpy as np
from PIL import Image
import torch
from torch import nn
from PIL import ImageDraw
import json
import torch.utils.data as data
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args(parser):
    # parsing args
    args = parser.parse_args()
    return args


def average_precision(output, target, total_count_):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    total_count_ = np.cumsum(np.ones((preds.shape[0], 1)))
    # compute average precision for each class
    for k in tqdm(range(preds.shape[1])):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets, total_count_)
    return 100 * ap.mean()


class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.sum = None
        self.cnt = None
        self.avg = None
        self.ema = None
        self.initialized = False

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def initialize(self, val, n):
        self.val = val
        self.sum = val * n
        self.cnt = n
        self.avg = val
        self.ema = val
        self.initialized = True

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.ema = self.ema * 0.99 + self.val * 0.01

class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model:nn.Module, update_fn):
        with torch.no_grad():
            for p_ema, p_model in zip(self.module.parameters(), model.parameters()):
                if self.device is not None:
                    p_model = p_model.to(device=self.device)
                p_ema.copy_(update_fn(p_ema, p_model))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x


def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def add_weight_decay_lr(model, weight_decay=1e-4, lr_backbone=1e-5, backbone_name='backbone', skip_list=()):
    decay = [[],[]]
    no_decay = [[],[]]
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            if backbone_name in name and param.requires_grad:
                no_decay[0].append(param)
            else:
                no_decay[1].append(param)
        else:
            if backbone_name in name and param.requires_grad:
                decay[0].append(param)
            else:
                decay[1].append(param)
    return [
        {'params': no_decay[0], 'weight_decay': 0., "lr": lr_backbone},
        {'params': no_decay[1], 'weight_decay': 0.},
        {'params': decay[0], 'weight_decay': weight_decay, "lr": lr_backbone},
        {'params': decay[1], 'weight_decay': weight_decay},
    ]


def get_class_ids_split(json_path, classes_dict):
    with open(json_path) as fp:
        split_dict = json.load(fp)
    if 'train class' in split_dict:
        only_test_classes = False
    else:
        only_test_classes = True

    train_cls_ids = set()
    val_cls_ids = set()
    test_cls_ids = set()

    # classes_dict = self.learn.dbunch.dataset.classes
    for idx, (i, current_class) in enumerate(classes_dict.items()):
        if only_test_classes:  # base the division only on test classes
            if current_class in split_dict['test class']:
                test_cls_ids.add(idx)
            else:
                val_cls_ids.add(idx)
                train_cls_ids.add(idx)
        else:  # per set classes are provided
            if current_class in split_dict['train class']:
                train_cls_ids.add(idx)
            # if current_class in split_dict['validation class']:
            #     val_cls_ids.add(i)
            if current_class in split_dict['test class']:
                test_cls_ids.add(idx)

    train_cls_ids = np.fromiter(train_cls_ids, np.int32)
    val_cls_ids = np.fromiter(val_cls_ids, np.int32)
    test_cls_ids = np.fromiter(test_cls_ids, np.int32)
    return train_cls_ids, val_cls_ids, test_cls_ids


def update_wordvecs(model, train_wordvecs=None, test_wordvecs=None):
    if hasattr(model, 'fc'):
        if train_wordvecs is not None:
            model.fc.decoder.query_embed = train_wordvecs.transpose(0, 1).to(device)
        else:
            model.fc.decoder.query_embed = test_wordvecs.transpose(0, 1).to(device)
    elif hasattr(model, 'head'):
        if train_wordvecs is not None:
            model.head.decoder.query_embed = train_wordvecs.transpose(0, 1).to(device)
        else:
            model.head.decoder.query_embed = test_wordvecs.transpose(0, 1).to(device)
    else:
        print("model is not suited for ml-decoder")
        exit(-1)

def parse_csv_data(dataset_local_path, metadata_local_path):
    try:
        df = pd.read_csv(os.path.join(metadata_local_path, "data.csv"))
    except FileNotFoundError:
        # No data.csv in metadata_path. Try dataset_local_path:
        metadata_local_path = dataset_local_path
        df = pd.read_csv(os.path.join(metadata_local_path, "data.csv"))
    images_path_list = df.values[:, 0]
    # images_path_list = [os.path.join(dataset_local_path, images_path_list[i]) for i in range(len(images_path_list))]
    labels = df.values[:, 1]
    image_labels_list = [labels.replace('[', "").replace(']', "").split(', ') for labels in
                             labels]

    if df.values.shape[1] == 3:  # split provided
        valid_idx = [i for i in range(len(df.values[:, 2])) if df.values[i, 2] == 'val']
        train_idx = [i for i in range(len(df.values[:, 2])) if df.values[i, 2] == 'train']
    else:
        valid_idx = None
        train_idx = None

    # logger.info("em: end parsr_csv_data: num_labeles: %d " % len(image_labels_list))
    # logger.info("em: end parsr_csv_data: : %d " % len(image_labels_list))

    return images_path_list, image_labels_list, train_idx, valid_idx


def multilabel2numeric(multilabels):
    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit(multilabels)
    classes = multilabel_binarizer.classes_
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    multilabels_numeric = []
    for multilabel in multilabels:
        labels = [class_to_idx[label] for label in multilabel]
        multilabels_numeric.append(labels)
    return multilabels_numeric, class_to_idx, idx_to_class


def get_datasets_from_csv(dataset_local_path, metadata_local_path, train_transform,
                          val_transform, json_path):

    images_path_list, image_labels_list, train_idx, valid_idx = parse_csv_data(dataset_local_path, metadata_local_path)
    labels, class_to_idx, idx_to_class = multilabel2numeric(image_labels_list)

    images_path_list_train = [images_path_list[idx] for idx in train_idx]
    image_labels_list_train = [labels[idx] for idx in train_idx]

    images_path_list_val = [images_path_list[idx] for idx in valid_idx]
    image_labels_list_val = [labels[idx] for idx in valid_idx]

    train_cls_ids, _, test_cls_ids = get_class_ids_split(json_path, idx_to_class)

    train_dl = DatasetFromList(dataset_local_path, images_path_list_train, image_labels_list_train,
                               idx_to_class,
                               transform=train_transform, class_ids=train_cls_ids)

    val_dl = DatasetFromList(dataset_local_path, images_path_list_val, image_labels_list_val, idx_to_class,
                             transform=val_transform, class_ids=test_cls_ids)

    return train_dl, val_dl, train_cls_ids, test_cls_ids

def crop_fix(img: Image):
    w,h=img.size
    w=(w//4)*4
    h=(h//4)*4
    return img.crop((0,0,w,h))