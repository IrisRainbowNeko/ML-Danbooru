import os
import argparse

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from src_files.helper_functions.helper_functions import add_weight_decay_lr
from src_files.data.Danbooru import Danbooru_DN
from src_files.models import create_model
from src_files.loss_functions.losses import AsymmetricLoss
from torch.cuda.amp import GradScaler, autocast

from src_files import dist as Adist
from loguru import logger
import time

from accelerate import Accelerator
from accelerate.utils import set_seed
from train_ac import Trainer

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import warnings
warnings.filterwarnings('ignore')

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')

parser.add_argument('--imgs_train', type=str, default='/dataset/dzy/danbooru2021/px640')
parser.add_argument('--imgs_val', type=str, default='/dataset/dzy/danbooru2021/px640')
parser.add_argument('--label_train', type=str, default='/data3/dzy/datas/danbooru2021/danbooru2021/data_train.json')
parser.add_argument('--label_val', type=str, default='/data3/dzy/datas/danbooru2021/danbooru2021/data_val.json')
parser.add_argument('--arb', type=str, default=None)
parser.add_argument('--out_dir', type=str, default='models/')

parser.add_argument('--adam_8bit', action="store_true", default=False)
parser.add_argument('--log_step', type=int, default=20)
parser.add_argument('--save_step', type=int, default=2000)
parser.add_argument('--ema_step', default=1, type=int)
parser.add_argument('--log_dir', type=str, default='logs/')

parser.add_argument('--ckpt', default=None, type=str)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--start_step', default=0, type=int)

parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--ema', default=0.997, type=float)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lr_backbone', default=5e-5, type=float,
                    help='learning rate for backbone')
parser.add_argument('--weight_decay', default=1e-2, type=float)
parser.add_argument('--model_name', default='caformer_m36')
parser.add_argument('--model_path', default=None, type=str)
parser.add_argument('--num_classes', default=12547)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--image_size', default=448, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--batch_size', default=56, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
parser.add_argument('--max_grad_norm', default=1.0, type=float)

parser.add_argument('--frelu', default=False, type=str2bool)
parser.add_argument('--use_ml_decoder', default=False, type=str2bool)

# CAFormer
parser.add_argument('--decoder_embedding', default=512, type=int)
parser.add_argument('--num_layers_decoder', default=4, type=int)
parser.add_argument('--num_head_decoder', default=8, type=int)
parser.add_argument('--num_queries', default=80, type=int)
parser.add_argument('--scale_skip', default=1, type=int)

parser.add_argument('--drop_rate', default=0.0, type=float)
parser.add_argument('--drop_path_rate', default=0.4, type=float)

parser.add_argument('--base_ckpt', default=None, type=str)


class TrainerCA(Trainer):
    def __init__(self, args):
        super().__init__(args)

    def make_lr(self):
        self.args.lr = (self.args.batch_size / 56) * self.args.lr * self.world_size * self.accelerator.gradient_accumulation_steps
        self.args.lr_backbone = (self.args.batch_size / 56) * self.args.lr_backbone * self.world_size * self.accelerator.gradient_accumulation_steps

    def cal_loss(self, inputData, target):
        outputs = self.model(inputData)
        #outputs = (outputs * torch.softmax(outputs, dim=1)).sum(dim=1)
        loss = self.criterion(outputs, target)

        return loss, (outputs, )

    def get_parameter_group(self):
        return add_weight_decay_lr(self.model, self.args.weight_decay, lr_backbone=self.args.lr_backbone, backbone_name='encoder')

    def forward_val(self, model, input):
        outputs = self.model(input)
        #outputs = (outputs * torch.softmax(outputs, dim=1)).sum(dim=1)
        return torch.sigmoid(outputs).cpu()

def main():
    args = parser.parse_args()
    trainer = TrainerCA(args)
    trainer.train()

if __name__ == '__main__':
    main()