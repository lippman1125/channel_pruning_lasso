import torch
import numpy as np
from torch import nn

def feature_sample(input_feat, output_feat, kernel_size, padding, stride, batch_size, sample_num):
    unfold = nn.Unfold(kernel_size=kernel_size,
                       padding=padding,
                       stride=stride)
    # extract image patches correspond to each output point
    # shape=[B, C x K x K, H x W]
    patches = unfold(torch.from_numpy(input_feat).cpu()).numpy()

    # input:  [N, C, H, W]
    # output: [N*sample_num, C]
    b, o_c, o_h, o_w = output_feat.shape
    rand_inds = np.random.randint(0, o_w * o_h, sample_num)
    # reshape to [N, C, H*W] and select to [N, C, sample_num]
    output_feat = output_feat.reshape(b, o_c, -1)[:,:,rand_inds]
    # transpose to [N, sample_num, C]
    # reshape to [N*sample_num, C]
    f_out2save = output_feat.transpose(0, 2, 1).reshape(-1, o_c)

    i_c = input_feat.shape[1]
    # extract patches correspond to the sampled points
    # shape=[B, C x K x K, sample_num]
    # and permute to [B, sample_num, C x K x K]
    # adn reshape to [B*sample_num, C, K x K]
    if isinstance(kernel_size, tuple):
        k_h, k_w = kernel_size
        f_in2save = patches[:, :, rand_inds].transpose(0,2,1).reshape(-1, i_c, k_h*k_w)
    else:
        k = kernel_size
        f_in2save = patches[:, :, rand_inds].transpose(0,2,1).reshape(-1, i_c, k*k)

    return f_in2save, f_out2save