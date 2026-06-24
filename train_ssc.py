import math
import torch
import torch.nn as nn
import torch.nn.functional as F
_seed_ = 2020
import random
random.seed(2020)
import numpy as np
import torchvision
from torchvision import transforms
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from spikingjelly.activation_based import functional, surrogate, layer
from spikingjelly.activation_based import neuron as ner
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp
from torch.optim.lr_scheduler import StepLR
from utils import *
import os
import time
import json
import argparse
import logging
import h5py
from datetime import datetime
import torch.nn as nn
import torch.backends.cudnn as cudnn
from pathlib import Path
from functools import partial
import sys
from ssc_dataset import create_spikingjelly_frame_dataloader,create_fixed_time_h5_dataloader
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode
from module import dendrite,dend_compartment,soma,neuron,wiring
from spikingjelly.activation_based import functional, surrogate, layer
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from module.dend_compartment import ChannelPreservingTrunkDistalDendCompartment,SparseChannelPreservingTrunkDistalDendCompartment,CoupledSparseChannelPreservingTrunkDistalDendCompartment

def setup_train_logging(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def scalar_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()
    return float(value)

#device = 'cuda'
class ff_SHD(nn.Module):
    def __init__(self, in_dim=700, hidden=[128,128], out_dim=35,drop=0.0,T=250):
        super().__init__()
        layers = []
        layers += [layer.Conv1d(in_dim, hidden[0],kernel_size=1),
                   #layer.BatchNorm1d(hidden[0]),
                   nn.Dropout(drop),
                   SparseChannelPreservingTrunkDistalDendCompartment(channels=hidden[0],num_branches=16,compartments_per_branch=8,branch_degree=4,shared_tau_parallel=False), #,branch_readout_mode="linear"),#,learn_edge_gain=False,learn_comp_gain=False),
                   #nn.Identity(),
                   #soma.MaskedSlidingPSN(order=T,surrogate_function=surrogate.Sigmoid())]
                   #soma.IntergerSoma_ssf(decay_input=False)]
                   #soma.PSNIntergerSoma_ssf(psn_order=T)]
                   #soma.SelectiveAstroPSNIntergerSoma_ssf(psn_order=T,astro_update_interval=25,astro_pool_kernel=5)] 
                   soma.AstroPSNIntergerSoma_ssf(psn_order=T,astro_update_interval=5,astro_pool_kernel=5)]  #需要试astro_event_write=True
        layers += [layer.Conv1d(hidden[0], hidden[1],kernel_size=1),
                   #layer.BatchNorm1d(hidden[1]),
                   nn.Dropout(drop),
                   SparseChannelPreservingTrunkDistalDendCompartment(channels=hidden[1],num_branches=16,compartments_per_branch=8,branch_degree=4,shared_tau_parallel=False),  #,branch_readout_mode="linear"), #,learn_edge_gain=False,learn_comp_gain=False),
                   #nn.Identity(),
                   #soma.MaskedSlidingPSN(order=T,surrogate_function=surrogate.Sigmoid())]
                   #soma.IntergerSoma_ssf(decay_input=False)]
                   #soma.PSNIntergerSoma_ssf(psn_order=T)]
                   #soma.SelectiveAstroPSNIntergerSoma_ssf(psn_order=T,astro_update_interval=25,astro_pool_kernel=5)]
                   soma.AstroPSNIntergerSoma_ssf(psn_order=T,astro_update_interval=5,astro_pool_kernel=5)]
        layers += [layer.Conv1d(hidden[1], out_dim,kernel_size=1)]
        self.features = nn.Sequential(*layers)

        functional.set_step_mode(self, 'm')

    def forward(self, x): # x [N, T, F]
        x = x.transpose(0, 1)
        x = x.unsqueeze(-1)
        assert x.dim() == 4, "dimension of x is not correct!"

        x = self.features(x).squeeze(-1)
        return x.mean(0)

def train(train_loader, model, criterion, optimizer, epoch, args, gpu):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(gpu, dtype=torch.float32, non_blocking=True)
        target = target.to(gpu, non_blocking=True)

        functional.reset_net(model)
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.print_freq > 0 and i % args.print_freq == 0:
            progress.display(i)
    logging.info(
        'Train Epoch: [{}/{}], lr: {:.6f}, top1: {:.4f}'.format(epoch, args.epochs, optimizer.param_groups[0]['lr'],
                                                                scalar_value(top1.avg)))
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args, gpu):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(gpu, dtype=torch.float32, non_blocking=True)
            target = target.to(gpu, non_blocking=True)

            functional.reset_net(model)
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # torch.cuda.synchronize()

            if args.print_freq > 0 and i % args.print_freq == 0:
                progress.display(i)

        logging.info(
            ' * Loss {:.4e} Acc@1 {:.3f} Acc@5 {:.3f}'.format(
                losses.avg,
                scalar_value(top1.avg),
                scalar_value(top5.avg),
            )
        )

    return top1.avg, top5.avg, losses.avg


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# python train_ssc.py --task SSC --device cuda:3     --lr 0.0005 --epochs 100 --schedule 40 80 --batch-size 64        --epochs 400 --workers 16 --cos   --optim sgd    --lr 0.1

parser = argparse.ArgumentParser(description='Sequential SHD/SSC')
parser.add_argument('--task', default='SSC', type=str, help='SHD, SSC')
parser.add_argument('--optim', default='adam', type=str, help='optimizer (default: adam)')
parser.add_argument('--results-dir', default='', type=str, metavar='PATH', help='path to cache (default: none)')
parser.add_argument('-p', '--print-freq', default=512, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--seed', default=0, type=int, metavar='N', help='seed')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')
parser.add_argument('--schedule', default=[], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
parser.add_argument('--batch-size', default=32, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--wd', default=0.0, type=float, metavar='W', help='weight decay')
parser.add_argument("--workers", type=int, default=8)
parser.add_argument('--cos', action='store_true', default=False, help='use cosine lr schedule')
parser.add_argument('--data-root', default='/data/hyx/ViT-dend/data/ssc', type=str,
                    help='path to extract/ or frames_number_250_split_by_number/')
parser.add_argument('--device', default=None, type=str, help="device, e.g. 'cuda', 'cuda:5', or 'cpu'")
parser.add_argument('--no-pin-memory', action='store_true', default=False, help='disable DataLoader pinned memory')

args = parser.parse_args()
if args.results_dir == '':
    args.results_dir = './exp/'+args.task+'ff'+'-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

Path(args.results_dir).mkdir(parents=True, exist_ok=True)
logger = setup_train_logging(os.path.join(args.results_dir, "log-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".txt"))

if args.device is not None:
    gpu = torch.device(args.device)
elif torch.cuda.is_available():
    gpu = torch.device('cuda')
    print('GPU is available')
else:
    gpu = torch.device('cpu')
    print('GPU is not available')
seed_everything(seed=args.seed, is_cuda=True)

torch.backends.cudnn.benchmark = True

if args.task.upper() == 'SSC' or args.task.upper() == 'SHD':
    T = 250
    in_dim = 700
    pin_memory = (not args.no_pin_memory) and gpu.type == 'cuda'
    train_loader = create_fixed_time_h5_dataloader(
    split="train",
    batch_size=args.batch_size,
    root_path=args.data_root,
    nb_steps=250,
    nb_units=700,
    shuffle=True,
    num_workers=args.workers,
    pin_memory=pin_memory,
)

    test_loader = create_fixed_time_h5_dataloader(
    split="test",
    batch_size=args.batch_size,
    root_path=args.data_root,
    nb_steps=250,
    nb_units=700,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=pin_memory,
)
else:
    raise NotImplementedError

model = ff_SHD(in_dim=in_dim, out_dim=35,T=T).to(gpu)
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"number of params: {n_parameters}")

logging.info(str(model))

criterion = nn.CrossEntropyLoss().to(gpu)

if args.optim == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
elif args.optim == 'adam':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
else:
    raise NotImplementedError
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# dump args
with open(args.results_dir + '/args.json', 'w') as fid:
    json.dump(args.__dict__, fid, indent=2)
logging.info(str(args))
start_epoch = 0
writer = SummaryWriter(args.results_dir, purge_step=start_epoch)

if args.print_freq > len(train_loader):
    args.print_freq = max(1, math.ceil(len(train_loader) / 2))

best_acc = argparse.Namespace(top1=0, top5=0)
for epoch in range(start_epoch, args.epochs):
    epoch_start_time = time.time()
    adjust_learning_rate(optimizer, epoch, args)

    train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch, args,gpu=gpu)
    #train_loader.reset()
    acc1, acc5, test_loss = validate(test_loader, model, criterion, args,gpu=gpu)
    #test_loader.reset()
    #scheduler.step()

    train_acc_value = scalar_value(train_acc)
    test_acc_value = scalar_value(acc1)
    test_acc5_value = scalar_value(acc5)
    lr = optimizer.param_groups[0]['lr']

    writer.add_scalar('train_loss', train_loss, epoch)
    writer.add_scalar('train_acc', train_acc_value, epoch)
    writer.add_scalar('test_loss', test_loss, epoch)
    writer.add_scalar('test_acc', test_acc_value, epoch)
    writer.add_scalar('test_acc5', test_acc5_value, epoch)
    writer.add_scalar('lr', lr, epoch)

    is_best = test_acc_value > best_acc.top1
    best_acc.top1 = max(best_acc.top1, test_acc_value)
    best_acc.top5 = max(best_acc.top5, test_acc5_value)

    logging.info(
        'epoch = {}, train_loss = {:.4f}, train_acc = {:.4f}, '
        'test_loss = {:.4f}, test_acc = {:.4f}, best_test_acc = {:.4f}, '
        'lr = {:.6f}, time = {:.2f}s'.format(
            epoch,
            train_loss,
            train_acc_value,
            test_loss,
            test_acc_value,
            best_acc.top1,
            lr,
            time.time() - epoch_start_time,
        )
    )
    #print('current beta values:', model)

    save_checkpoint({
        'epoch': epoch + 1,
        'best_acc': best_acc,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, is_best=is_best, dirname=args.results_dir, filename='checkpoint.pth.tar')

writer.close()