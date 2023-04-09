from typing import Tuple, Dict
import torch
import torch.utils.data
import torchvision
from torchvision import transforms
import os
import cv2
import random
import math

import albumentations as A

from albumentations.pytorch import ToTensorV2

def resize(image, size, max_size=None, interpolation=cv2.INTER_LINEAR):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.shape[1::-1], size, max_size)
    rescaled_image = cv2.resize(image, size, interpolation=interpolation)

    return rescaled_image


class RandomResize(A.DualTransform):
    def __init__(self, sizes, max_size=None, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1):
        super(RandomResize, self).__init__(always_apply, p)
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size
        self.interpolation = interpolation

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        size = random.choice(self.sizes)
        return resize(img, size, self.max_size, interpolation=interpolation)

    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

    def get_transform_init_args_names(self):
        return ("sizes", "max_size", "interpolation")


class CropFix(A.DualTransform):
    def __init__(self, item_size, always_apply=False, p=1):
        super(CropFix, self).__init__(always_apply, p)
        self.item_size = item_size

    def apply(self, img, **params):
        h, w = img.shape[:2]
        w = (w // self.item_size) * self.item_size
        h = (h // self.item_size) * self.item_size
        return img[:h,:w,:]

    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

    def get_transform_init_args_names(self):
        return ("item_size",)

def make_dn_transforms(image_set, fix_size=False, strong_aug=False, args=None):
    normalize = A.Compose([
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # config the params for data aug
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1280

    if image_set == 'train':
        if fix_size:
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                #RandomResize([(max_size, max(scales))]),
                CropFix(4),
                normalize,
            ])

        if strong_aug:
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                #RandomResize(scales, max_size=max_size),
                CropFix(4),

                A.GaussNoise(var_limit=(3, 30), p=0.4),
                A.RandomBrightnessContrast(0.3,0.3,p=0.4),
                A.RandomGamma(p=0.2),
                normalize,
            ])

        return A.Compose([
            A.HorizontalFlip(p=0.5),
            #A.RandomResize(scales, max_size=max_size),
            CropFix(4),
            normalize,
        ])

    if image_set in ['val', 'test']:

        if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == 'INFO':
            print("Under debug mode for flops calculation only!!!!!!!!!!!!!!!!")
            return A.Compose([
                A.Resize(640, 640),
                CropFix(4),
                normalize,
            ])

        return A.Compose([
            A.Resize(640, 640),
            #RandomResize([max(scales)], max_size=max_size),
            CropFix(4),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

class ResizeArea:
    def __init__(self, target_area=512*512):
        self.target_area=target_area

    def __call__(self, img):
        w, h = img.size
        s = math.sqrt(self.target_area/(w*h))
        w = int(w*s)
        h = int(h*s)
        img = img.resize((w, h))
        w = int(w/8)*8
        h = int(h/8)*8
        img = img.crop((0, 0, w, h))
        return img

class WeakRandAugment(transforms.RandAugment):
    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[torch.Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.1, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.1, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 0.1 * image_size[1], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 0.1 * image_size[0], num_bins), True),
            "Rotate": (torch.linspace(0.0, 8.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.1, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.1, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.2, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }