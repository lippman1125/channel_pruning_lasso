from prune import LassoPruner
from models.vgg import vgg11_bn
from data import Imagenet1kCalib
from config import LassoPruneConfig
import torch
import torchvision.transforms as transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
input_size = 224
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

img1k_calib = Imagenet1kCalib("/home/lqy/data/imagenet1k_calib", train_transform)
# for img in img1k_calib:
#     print(img.size())

train_loader = torch.utils.data.DataLoader(img1k_calib,
                                           batch_size=50,
                                           num_workers=1,
                                           pin_memory=True)

vgg11 = vgg11_bn()

lassopruner_config = LassoPruneConfig(vgg11,
                                      "/home/lqy/workshop/amc/checkpoints/vgg11_bn-6002323d.pth",
                                      train_dataloader=train_loader)
lassopruner = LassoPruner(lassopruner_config)

lassopruner.prune_layer(27, 0.5)



