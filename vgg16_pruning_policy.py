import torch
import torch.nn as nn
from torchvision import models
import cv2
import sys
import numpy as np

# after pruning, top1 = 63.536%
vgg16_ratios = { 5: 0.5,
                 9: 0.5, # conv2
                12: 0.5,
                16: 0.5, # conv3
                19: 0.5,
                22: 0.5,
                26: 0.2, # conv4
                29: 0.2,
                32: 0.2
                }

# after pruning, top1 = 63.536%
# vgg16_ratios = { 5: 0.5,
#                  9: 0.5, # conv2
#                  12: 0.5,
#                  16: 0.4, # conv3
#                  19: 0.4,
#                  22: 0.4,
#                  26: 0.2, # conv4
#                  29: 0.2,
#                  32: 0.2
#                  }

def vgg16_pruning_policy(model, layer_index, weights, filter_index, device):
    prev_op = None
    offset = -1
    # assign new weight to pruned model
    op = list(model.modules())[layer_index]
    # print(op.weight.data.size(), weights.shape)
    op.weight.data = torch.from_numpy(weights).to(device)
    # we do not need to modify bias, because output channels number is not modified
    # if op.bias is not None:
    #     op.bias.data = torch.zeros_like(op.bias.data)
    # print(op.bias.data.cpu().numpy())

    # find prev conv, because we prune channel of present conv and prune filters of prev conv simutaneously
    while layer_index + offset >= 0:
        prev_op = list(model.modules())[layer_index + offset]
        # print(prev_op)
        if type(prev_op) == nn.Conv2d or type(prev_op) == nn.Linear:
            prev_op.weight.data = torch.from_numpy(prev_op.weight.data.cpu().numpy()[filter_index]).to(device)
            if prev_op.bias is not None:
                prev_op.bias.data = torch.from_numpy(prev_op.bias.data.cpu().numpy()[filter_index]).to(device)
            break
        # select bn
        elif type(prev_op) == nn.BatchNorm2d:
            prev_op.weight.data = torch.from_numpy(prev_op.weight.data.cpu().numpy()[filter_index]).to(device)
            prev_op.bias.data = torch.from_numpy(prev_op.bias.data.cpu().numpy()[filter_index]).to(device)
            prev_op.running_mean.data = torch.from_numpy(prev_op.running_mean.data.cpu().numpy()[filter_index]).to(device)
            prev_op.running_var.data = torch.from_numpy(prev_op.running_var.data.cpu().numpy()[filter_index]).to(device)
        offset -= 1
