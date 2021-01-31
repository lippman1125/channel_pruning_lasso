from PIL import Image
import os
import glob
import random
import numpy as np

import torch
import torchvision.transforms as transforms

class MyDataSet(object):
    """
    Underlying dataset to support uniform api for different data loader.
    """

    def __init__(self):
        pass

    def __len__(self):
        assert False

    def __getitem__(self, item):
        assert False

class Imagenet1kCalib(MyDataSet):
    def __init__(self, path, transform=None):
        super(Imagenet1kCalib, self).__init__()
        self.path = path
        self.transform = transform
        self.path_list = glob.glob(os.path.join(self.path, "*.*"))
        random.shuffle(self.path_list)
        print("total files : {}".format(len(self.path_list)))

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, item):
        img = Image.open(self.path_list[item])
        img_rgb = img.convert('RGB')
        if self.transform is not None:
            img_rgb = self.transform(img_rgb)
        return img_rgb, np.ones((1), dtype=np.int32)


if __name__ == "__main__":

    # img1k_calib = Imagenet1kCalib("/home/lqy/data/imagenet1k_calib")
    # for img in img1k_calib:
    #     img.show("pic", img)

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
                                               batch_size=500,
                                               num_workers=1,
                                               pin_memory=True)

    for i, (data, label) in enumerate(train_loader):
        print(i, data.size())