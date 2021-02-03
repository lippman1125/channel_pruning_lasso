from prune import LassoPruner
from models.vgg import vgg16_bn
from data import Imagenet1kCalib
from config import LassoPruneConfig
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from vgg16_pruning_policy import vgg16_pruning_policy

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
input_size = 224
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

img1k_calib = Imagenet1kCalib("/data/imagenet1k_calib", train_transform)
# for img in img1k_calib:
#     print(img.size())

train_loader = torch.utils.data.DataLoader(img1k_calib,
                                           batch_size=100,
                                           num_workers=1,
                                           pin_memory=True)

val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder("/data/ILSVRC2012/val",
        transforms.Compose([
            transforms.Resize(int(input_size / 0.875)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=50,
        shuffle=False,
        num_workers=1,
        pin_memory=True)

vgg16 = vgg16_bn()

lassopruner_config = LassoPruneConfig(vgg16,
                                      "vgg16_bn-6c64b313.pth",
                                      train_dataloader=train_loader,
                                      pruner='lasso',
                                      val_dataloader=val_loader,
                                      criterion=nn.CrossEntropyLoss(),
                                      policy=vgg16_pruning_policy,
                                      fmap_path='fmap.pkl')
lassopruner = LassoPruner(lassopruner_config)

# vgg16_ratios = {5: 17/64, 9: 37/128, 12: 47/128, 16: 83/256, 19: 89/256, 22: 106/256,
#                 26: 175/512, 29: 192/512, 32: 227/512, 36: 398/512, 39: 390/512, 42: 379/512}

# vgg16_ratios = {5: 17/64, 9: 37/128, 12: 47/128, 16: 83/256, 19: 89/256, 22: 106/256,
#                  26: 175/512, 29: 192/512, 32: 227/512}
vgg16_ratios = { 5: 0.5,
                 9: 0.5, # conv2
                 12: 0.5,
                 16: 0.5, # conv3
                 19: 0.5,
                 22: 0.4,
                 26: 0.2, # conv4
                 29: 0.2,
                 32: 0.2
                 }

# vgg16_ratios = {5: 0.5, 9: 0.5, 16: 0.4, 19: 0.5, 22: 0.4, 29: 0.4}
# vgg16_ratios = {5: 0.5, 9: 0.5, 16: 0.4}

# lassopruner.prune_layer(26, 0.5)
lassopruner.prune(vgg16_ratios)

lassopruner.metric()



