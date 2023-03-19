#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import simsiam.loader
import simsiam.builder

from pygit2 import Repository
import socket
from datetime import datetime
from pathlib import Path
from datetime import timedelta
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.elastic.utils.data import ElasticDistributedSampler

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimSiam Training')
parser.add_argument('--data', default='/mnt/aitrics_ext/ext02/chris/imagenet',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial (base) learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# simsiam specific configs:
parser.add_argument('--dim', default=2048, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred-dim', default=512, type=int,
                    help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='Fix learning rate for the predictor')


parser.add_argument('--checkpoint-dir', default='/nfs/thena/chris/equi/ckpts', type=Path, help='path to checkpoint directory')
parser.add_argument( "--dataset", default="IMAGENET", choices=["IMAGENET", "CIFAR100", "CIFAR10"], type=str, help="Dataset")
parser.add_argument('--equiv-mode', default='False', choices=['contrastive', 'lp', 'cosine', 'equiv_only', 'False'], type=str, help='loss type to learn equivariance')
parser.add_argument('--pushtoken', default='o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95', help='Push Bullet token')

def main():
    print(f'Training: {Repository(".").head.shorthand}')
    t_overall = time.time()

    args = parser.parse_args()
    if args.equiv_mode == 'False':
        args.equiv_mode = False
    args.multiprocessing_distributed = True

    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset)
    if not(os.path.isdir(args.checkpoint_dir)):
        args.checkpoint_dir = args.checkpoint_dir.replace('thena', 'thena/ext01')
        if not(os.path.isdir(args.checkpoint_dir)):
            tempdir = '/mnt/aitrics_ext/ext01/chris/temp'
            if not(os.path.isdir(tempdir)):
                os.mkdir(tempdir)
            args.checkpoint_dir = tempdir

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # tr=''
    # for tt in args.transform_types:
    #     tr+=tt
    args.exp_name = f'{datetime.today().strftime("%m%d")}_{socket.gethostname()}_{Repository(".").head.shorthand}_SimSiam_{args.dataset}_lr{args.learning_rate}_{args.equiv_mode}_p'
    torch.backends.cudnn.benchmark = True
    
    device_id = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device_id)
    dist.init_process_group(backend=args.dist_backend, init_method="env://", timeout=timedelta(seconds=10))
    args.world_size = dist.get_world_size()
    print(f"=> set cuda device = {device_id}")
    print(f'args.world_size: {args.world_size}')
    print(f'master: {os.environ.get("MASTER_ADDR")}')
    

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = simsiam.builder.SimSiam(args, args.dim, args.pred_dim).cuda(device_id)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DistributedDataParallel(model, device_ids=[device_id])

    # infer learning rate before changing batch size
    init_lr = args.learning_rate * args.batch_size / 256
    
    criterion = nn.CosineSimilarity(dim=1).cuda(device_id)

    if args.fix_pred_lr:
        optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    train_dataset = datasets.ImageFolder(
        traindir,
        simsiam.loader.TwoCropsTransform(transforms.Compose(augmentation)))

    sampler = ElasticDistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=int(args.batch_size / float(args.world_size)), shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=sampler, drop_last=True)

    if device_id == 0:
        t_epoch = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        torch.cuda.empty_cache()
        train_loader.batch_sampler.sampler.set_epoch(epoch)

        if device_id == 0:
            print(f'Epoch: {epoch+1}, Time: {round(time.time() - t_epoch, 3)}')
            t_epoch = time.time()

        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, device_id)

    if device_id == 0:
        torch.save(dict(encoder=model.module.encoder.state_dict(),
                        projector_inv=model.module.projector_inv.state_dict() if (args.equiv_mode != 'equiv_only') else None,
                        projector_equiv=model.module.projector_equiv.state_dict() if args.equiv_mode else None,
                        args=args),
                    os.path.join(args.checkpoint_dir, f'{args.exp_name}.pth'))
        
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'arch': args.arch,
        #     'state_dict': model.state_dict(),
        #     'optimizer' : optimizer.state_dict(),
        # }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))

    print(f'{args.exp_name} training took {round(time.time()-t_overall, 3)} sec')

    if args.pushtoken:
        from pushbullet import API
        pb = API()
        pb.set_token(args.pushtoken)
        push = pb.send_note('SimSiam train finished', f'{socket.gethostname()}')

def train(train_loader, model, criterion, optimizer, epoch, args, device_id):
    # batch_time = AverageMeter('Time', ':6.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    # losses = AverageMeter('Loss', ':.4f')
    # progress = ProgressMeter(
    #     len(train_loader),
    #     [batch_time, data_time, losses],
    #     prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    # end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        # data_time.update(time.time() - end)

        images[0] = images[0].cuda(device_id, non_blocking=True)
        images[1] = images[1].cuda(device_id, non_blocking=True)

        # compute output and loss
        p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        # losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


if __name__ == '__main__':
    main()
