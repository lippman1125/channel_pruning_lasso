import os
import argparse

from prune import LassoPruner
from data import Imagenet1kCalib
from config import LassoPruneConfig

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from vgg16_pruning_policy import vgg16_pruning_policy, vgg16_ratios

def get_config():
    print('=> Building model..')
    if args.model == 'vgg16_bn':
        from models.vgg import vgg16_bn
        net, ratios, policy = vgg16_bn(), vgg16_ratios, vgg16_pruning_policy
    elif args.model == 'vgg11_bn':
        from models.vgg import vgg11_bn
        net, ratios, policy = vgg11_bn(), {20: 0.5}, vgg16_pruning_policy
    else:
        print("Not support model {}".format(args.model))
        raise NotImplementedError
    return net, ratios, policy

def check_args(args):
    print("=> Checking Parameter")
    ret = 0
    if not os.path.exists(args.calib_dir):
        print("calib dir {} not exists".format(args.calib_dir))
        ret = -1
    if not os.path.exists(args.valid_dir):
        print("valid dir {} not exists".format(args.valid_dir))
        ret = -1
    if not os.path.exists(args.ckpt):
        print("checkpoint {} not exists".format(args.ckpt))
        ret = -1
    return ret

parser = argparse.ArgumentParser(description='Channel pruning')
parser.add_argument('--model', default='vgg16_bn', type=str, help='name of the model to train')
parser.add_argument('--batch_size', default=50, type=int, help='batch size')
parser.add_argument('--calib_batch', default=None, type=int, help='how many batches used to calib')
parser.add_argument('--n_worker', default=1, type=int, help='number of data loader worker')
parser.add_argument('--seed', default=None, type=int, help='random seed to set')
parser.add_argument('--ckpt', default=None, type=str, help='checkpoint path to resume from')
parser.add_argument('--calib_dir', default=None, type=str, help='calib dataset path')
parser.add_argument('--valid_dir', default=None, type=str, help='valid dataset path')
parser.add_argument('--pruner', default='lasso', type=str, help='channel pruner (lasso/l1norm)')
parser.add_argument('--fmap', default=None, type=str, help='feature map file')
parser.add_argument('--fmap_save', action='store_true', help='save feature map')
parser.add_argument('--fmap_save_path', default='./', type=str, help='feature map save path')
args = parser.parse_args()
print(args)

if check_args(args) < 0:
    print("paramters check fail")
    exit(0)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
input_size = 224
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

img1k_calib = Imagenet1kCalib(args.calib_dir, train_transform)
# for img in img1k_calib:
#     print(img.size())

train_loader = torch.utils.data.DataLoader(img1k_calib,
                                           batch_size=args.batch_size,
                                           num_workers=args.n_worker,
                                           pin_memory=True)

val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.valid_dir,
        transforms.Compose([
            transforms.Resize(int(input_size / 0.875)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_worker,
        pin_memory=True)

net, sparsity_ratios, pruning_policy = get_config()

lassopruner_config = LassoPruneConfig(args.model,
                                      net,
                                      args.ckpt,
                                      train_dataloader=train_loader,
                                      pruner=args.pruner,
                                      val_dataloader=val_loader,
                                      criterion=nn.CrossEntropyLoss(),
                                      policy=pruning_policy,
                                      fmap_path=args.fmap)
lassopruner_config.calib_batch = args.calib_batch
lassopruner_config.fmap_save = args.fmap_save
lassopruner_config.fmap_save_path = args.fmap_save_path

lassopruner = LassoPruner(lassopruner_config)

# vgg16_ratios = {5: 17/64, 9: 37/128, 12: 47/128, 16: 83/256, 19: 89/256, 22: 106/256,
#                 26: 175/512, 29: 192/512, 32: 227/512, 36: 398/512, 39: 390/512, 42: 379/512}

# vgg16_ratios = {5: 17/64, 9: 37/128, 12: 47/128, 16: 83/256, 19: 89/256, 22: 106/256,
#                  26: 175/512, 29: 192/512, 32: 227/512}
# config 1, top1 63.536%
'''
vgg16_ratios = { 5: 0.5,
                 9: 0.5, # conv2
                 12: 0.5,
                 16: 0.4, # conv3
                 19: 0.4,
                 22: 0.4,
                 26: 0.2, # conv4
                 29: 0.2,
                 32: 0.2
                 }
'''

# lassopruner.prune_layer(26, 0.5)
lassopruner.prune(sparsity_ratios)
lassopruner.metric()
lassopruner.save_pruned_model('./')



