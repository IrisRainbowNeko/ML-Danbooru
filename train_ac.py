import os
import argparse

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from src_files.helper_functions.helper_functions import mAP, CutoutPIL, ModelEma, \
    add_weight_decay
from src_files.data.Danbooru import Danbooru
from src_files.models import create_model
from src_files.loss_functions.losses import AsymmetricLoss
from torch.cuda.amp import GradScaler, autocast

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from src_files import dist as Adist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
import numpy as np
import random
from loguru import logger
import time

from accelerate import Accelerator
from accelerate.utils import set_seed

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

parser.add_argument('--adam_8bit', action="store_true", default=False)
parser.add_argument('--log_step', type=int, default=20)
parser.add_argument('--save_step', type=int, default=2000)
parser.add_argument('--log_dir', type=str, default='logs/')

parser.add_argument('--ckpt', default=None, type=str)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--start_step', default=0, type=int)

parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--model_name', default='tresnet_l')
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

# ML-Decoder
parser.add_argument('--use_ml_decoder', default=1, type=int)
parser.add_argument('--num_of_groups', default=512, type=int)  # full-decoding
parser.add_argument('--decoder_embedding', default=1024, type=int)
parser.add_argument('--zsl', default=0, type=int)

parser.add_argument('--frelu', type=str2bool, default=True)
parser.add_argument('--xformers', type=str2bool, default=True)
parser.add_argument('--learn_query', type=str2bool, default=False)

def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False

def main():
    args = parser.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision='fp16',
        step_scheduler_with_optimizer=False,
    )

    if accelerator.is_local_main_process:
        os.makedirs(args.log_dir, exist_ok=True)
        logger.add(os.path.join(args.log_dir, f'{time.strftime("%Y-%m-%d-%H-%M-%S")}.log'))

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = accelerator.num_processes

    logger.info(f'rank: {local_rank}')
    if accelerator.is_local_main_process:
        logger.info(f'world_size: {world_size}')
        logger.info(f'accumulation: {accelerator.gradient_accumulation_steps}')

    args.lr = (args.batch_size / 56) * args.lr * world_size * accelerator.gradient_accumulation_steps

    set_seed(41 + local_rank)

    # Setup model
    if accelerator.is_local_main_process:
        logger.info('creating model {}...'.format(args.model_name))
    model = create_model(args).cuda()

    if accelerator.is_local_main_process:
        logger.info('done')
        logger.info(f'lr_max: {args.lr}')

    val_dataset = Danbooru(args.imgs_val,
                           args.label_val,
                           num_class=args.num_classes,
                           file_ext='jpg',
                           transform=transforms.Compose([
                               transforms.Resize((args.image_size, args.image_size)),
                               transforms.ToTensor(),
                               # normalize, # no need, toTensor does normalization
                           ]))
    train_dataset = Danbooru(args.imgs_train,
                            args.label_train,
                             num_class=args.num_classes,
                             file_ext='jpg',
                             transform=transforms.Compose([
                                   transforms.Resize((args.image_size, args.image_size)),
                                   #CutoutPIL(cutout_factor=0.5),
                                   #RandAugment(),
                                   #transforms.RandomApply(
                                   #    transforms=[
                                   #        transforms.RandomAffine(degrees=(0, 10), translate=(0.0, 0.1),
                                   #                                scale=(0.9, 1.0), shear=8, fill=(0, 0, 0))
                                   #    ], p=0.25
                                   #),
                                   transforms.RandomHorizontalFlip(p=0.25),
                                   transforms.ToTensor(),
                                   # normalize,
                               ]))
    logger.info(f"len(val_dataset)): {len(val_dataset)}")
    logger.info(f"len(train_dataset)): {len(train_dataset)}")

    # Pytorch Data loader
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size,
                                                                   rank=local_rank, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=False, sampler=val_sampler)

    # Actuall Training
    train_multi_label(accelerator, model, train_loader, val_loader, args.lr, local_rank, args)

def save_model(model_name, model, ema: ModelEma, local_rank, epoch, step):
    if local_rank == 0:
        try:
            torch.save({'model':model.module.state_dict(), 'ema':ema.module.state_dict()}, os.path.join(
                'models/', f'{model_name}-{epoch + 1}-{step + 1}.ckpt'))
        except:
            pass

def train_multi_label(accelerator, model, train_loader, val_loader, lr, local_rank, args):
    ema = ModelEma(model, 0.997)  # 0.9997^641=0.82

    if args.ckpt:
        state = torch.load(args.ckpt, map_location='cpu')
        if 'model' in state:
            model.load_state_dict(state['model'], strict=True)
            ema.module.load_state_dict(state['ema'], strict=True)
        else:
            model.load_state_dict(state, strict=True)

    #ema.to(accelerator.device)

    # set optimizer
    Epochs = args.epochs
    weight_decay = 1e-4
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    parameters = add_weight_decay(model, weight_decay)

    if args.adam_8bit:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(params=parameters, lr=lr, weight_decay=0)
    elif accelerator.state.deepspeed_plugin is not None:
        from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
        optimizer = FusedAdam(params=parameters, lr=lr, weight_decay=0)
    else:
        optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn

    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.2)

    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16

    highest_mAP = 0
    trainInfoList = []
    loss_sum = 0

    start_step = args.start_step
    scheduler.step(args.start_epoch*steps_per_epoch+start_step)

    for epoch in range(args.start_epoch, Epochs):
        for i, (inputData, target) in enumerate(train_loader):
            with accelerator.accumulate(model):
                inputData = inputData.to(accelerator.device, dtype=weight_dtype)
                target = target.to(accelerator.device, dtype=weight_dtype)

                output = model(inputData)  # sigmoid will be done in loss !
                loss = criterion(output, target)
                loss_sum += loss.item()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                ema.update(model)
                # store information
                if accelerator.is_local_main_process:
                    if (i+1) % args.log_step == 0:
                        trainInfoList.append([epoch, i, loss.item()])
                        logger.info('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                              .format(epoch, Epochs, str(i+1).zfill(3), str(steps_per_epoch).zfill(3),
                                      scheduler.get_last_lr()[0],
                                      loss_sum/args.log_step))
                        loss_sum = 0

                    if (i + 1) % args.save_step == 0:
                        save_model(args.model_name, model, ema, local_rank, epoch, i)

                if start_step+i >= steps_per_epoch:
                    start_step=-1
                    break

        if accelerator.is_local_main_process:
            save_model(args.model_name, model, ema, local_rank, epoch, i)

        model.eval()

        mAP_score = validate_multi(val_loader, model, ema, local_rank)
        model.train()
        if local_rank in [-1, 0]:
            if mAP_score > highest_mAP:
                highest_mAP = mAP_score
                save_model(args.model_name, model, ema, local_rank, epoch, i)
            logger.info('current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(mAP_score, highest_mAP))


def validate_multi(val_loader, model, ema_model, local_rank):
    logger.info("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    for i, (input, target) in enumerate(val_loader):
        target = target
        # compute output
        with torch.no_grad():
            with autocast():
                output_regular = Sig(model(input.cuda())).cpu()
                output_ema = Sig(ema_model.module(input.cuda())).cpu()

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        targets.append(target.cpu().detach())

    targets_cat = torch.cat(targets)
    preds_regular_cat = torch.cat(preds_regular)
    preds_ema_cat = torch.cat(preds_ema)

    if local_rank>-1:
        targets_all=Adist.gather(targets_cat, dst=0)
        preds_regular_all=Adist.gather(preds_regular_cat, dst=0)
        preds_ema_all=Adist.gather(preds_ema_cat, dst=0)

    if local_rank in [-1, 0]:
        mAP_score_regular = mAP(torch.cat(targets_all).numpy(), torch.cat(preds_regular_all).numpy())
        mAP_score_ema = mAP(torch.cat(targets_all).numpy(), torch.cat(preds_ema_all).numpy())
        logger.info("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
        return max(mAP_score_regular, mAP_score_ema)
    else:
        return 0


if __name__ == '__main__':
    main()
