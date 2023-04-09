import os
import argparse
import time

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms

from src_files.helper_functions.bn_fusion import fuse_bn_recursively
from src_files.helper_functions.helper_functions import crop_fix
from src_files.models.tresnet.tresnet import InplacABN_to_ABN
from src_files.models import create_model

import json
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def make_args():
    parser = argparse.ArgumentParser(description='ML-Danbooru Demo')
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--class_map', type=str, default='./class.json')
    parser.add_argument('--model_name', default='caformer_m36')
    parser.add_argument('--num_classes', default=12547)
    parser.add_argument('--image_size', default=448, type=int,
                        metavar='N', help='input image size')
    parser.add_argument('--thr', default=0.75, type=float,
                        metavar='N', help='threshold value')
    parser.add_argument('--keep_ratio', type=str2bool, default=False)

    # ML-Decoder
    parser.add_argument('--use_ml_decoder', default=0, type=int)
    parser.add_argument('--fp16', action="store_true", default=False)
    parser.add_argument('--ema', action="store_true", default=False)

    parser.add_argument('--frelu', type=str2bool, default=True)
    parser.add_argument('--xformers', type=str2bool, default=False)

    # CAFormer
    parser.add_argument('--decoder_embedding', default=384, type=int)
    parser.add_argument('--num_layers_decoder', default=4, type=int)
    parser.add_argument('--num_head_decoder', default=8, type=int)
    parser.add_argument('--num_queries', default=80, type=int)
    parser.add_argument('--scale_skip', default=1, type=int)

    args = parser.parse_args()
    return args

class Demo:
    def __init__(self, args):
        self.args=args

        print('creating model {}...'.format(args.model_name))
        args.model_path = None
        model = create_model(args, load_head=True).to(device)
        state = torch.load(args.ckpt, map_location='cpu')
        if args.ema:
            state = state['ema']
        elif 'model' in state:
            state=state['model']
        model.load_state_dict(state, strict=True)

        self.model = model.to(device).eval()
        if args.fp16:
            self.model = self.model.half()
        #######################################################
        print('done')

        if args.keep_ratio:
            self.trans = transforms.Compose([
                transforms.Resize(args.image_size),
                crop_fix,
                transforms.ToTensor(),
            ])
        else:
            self.trans = transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                ])

        self.load_class_map()

    def load_class_map(self):
        with open(self.args.class_map, 'r') as f:
            self.class_map = json.loads(f.read())

    def load_data(self, path):
        img = Image.open(path).convert('RGB')
        img = self.trans(img)
        return img

    def infer_one(self, img):
        if self.args.fp16:
            img = img.half()
        img = img.unsqueeze(0)
        output = torch.sigmoid(self.model(img)).cpu().view(-1)
        pred = torch.where(output > self.args.thr)[0].numpy()

        cls_list = [(self.class_map[str(i)], output[i]) for i in pred]
        return cls_list

    @torch.no_grad()
    def infer(self, path):
        if os.path.isfile(path):
            img = self.load_data(path).to(device)
            cls_list = self.infer_one(img)
            return cls_list
        else:
            tag_dict = {}
            img_list = [os.path.join(path, x) for x in os.listdir(path) if x[x.rfind('.'):].lower() in IMAGE_EXTENSIONS]
            for item in tqdm(img_list):
                img = self.load_data(item).to(device)
                cls_list = self.infer_one(img)
                cls_list.sort(reverse=True, key=lambda x: x[1])
                if self.args.out_type == 'txt':
                    with open(item[:item.rfind('.')] + '.txt', 'w', encoding='utf8') as f:
                        f.write(', '.join([name.replace('_', ' ') for name, prob in cls_list]))
                elif self.args.out_type == 'json':
                    tag_dict[os.path.basename(item)] = ', '.join([name.replace('_', ' ') for name, prob in cls_list])

            if self.args.out_type == 'json':
                with open(os.path.join(path, 'image_captions.json'), 'w', encoding='utf8') as f:
                    f.write(json.dumps(tag_dict))

            return None

#python demo_ca.py --data imgs/t1.jpg --model_name caformer_m36 --ckpt ckpt/caformer_m36-2-20000.ckpt --thr 0.7 --image_size 448
if __name__ == '__main__':
    args = make_args()
    demo = Demo(args)
    cls_list = demo.infer(args.data)

    cls_list.sort(reverse=True, key=lambda x:x[1])
    print(', '.join([f'{name}:{prob:.3}' for name, prob in cls_list]))
    print(', '.join([name for name, prob in cls_list]))