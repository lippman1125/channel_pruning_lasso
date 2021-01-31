import os
import torch
import numpy as np
import torch.nn as nn
from utils import feature_sample
from lasso import channel_pruning, weight_reconstruction
from abc import ABCMeta, abstractclassmethod

class Pruner(metaclass=ABCMeta):
    @abstractclassmethod
    def prune_layer(self, idx, ratio):
        pass
    @abstractclassmethod
    def prune(self, ratio):
        pass

class LassoPruner(Pruner):
    def __init__(self, config):
        super(LassoPruner, self).__init__()
        self.device = config.device
        self.model = config.model.to(self.device)
        self.ckpt = config.ckpt
        self.train_dataloader = config.train_dataloader
        self.val_dataloader = config.val_dataloader
        self.n_points_per_layer = config.n_points_per_layer
        self.prunable_layer_types = config.prunable_layer_types
        self.calib_batch = config.calib_batch
        self._load_checkpoint()
        self._build_index()
        self._extract_layer_info()

    def set_method(self):
        pass

    def _load_checkpoint(self):
        assert os.path.exists(self.ckpt)
        checkpoint = torch.load(self.ckpt)
        if 'state_dict' in checkpoint:
           checkpoint = checkpoint['state_dict']
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        self.model.load_state_dict(checkpoint)

    def _build_index(self):
        self.prunable_idx = []
        self.prunable_ops = []
        self.layer_type_dict = {}

        # build index and the min strategy dict
        for i, m in enumerate(self.model.modules()):
            if type(m) in self.prunable_layer_types:
                # we do not prune depthwise conv
                if type(m) == nn.Conv2d or type(m) == nn.Linear:
                    # really prunable
                    # for example:
                    # mobilenet  depthconv2d 3x3 + conv2d 1x1
                    self.prunable_idx.append(i)
                    self.prunable_ops.append(m)
                    self.layer_type_dict[i] = type(m)

        for i in range(len(self.prunable_idx)):
            print('=> Prunable layer idx: {} op type: {}'.format(self.prunable_idx[i], self.prunable_ops[i]))

    def _extract_layer_info(self):
        m_list = list(self.model.modules())

        self.layer_info_dict = dict()
        for idx in self.prunable_idx:
            self.layer_info_dict[idx] = dict()

        # extend the forward fn to record layer info
        def new_forward(m):
            def lambda_forward(x):
                m.input_feat = x.clone()
                y = m.old_forward(x)
                m.output_feat = y.clone()
                return y

            return lambda_forward

        for idx in self.prunable_idx:  # get all
            m = m_list[idx]
            m.old_forward = m.forward
            m.forward = new_forward(m)

        # now let the image flow
        print('=> Extracting information...')
        with torch.no_grad():
            for i_b, (input, target) in enumerate(self.train_dataloader):  # use image from train set
                if i_b > self.calib_batch:
                    break
                input_var = torch.autograd.Variable(input).to(self.device)

                # inference and collect stats
                _ = self.model(input_var)

                # first conv exclude, because we do not prune input channel
                for idx in self.prunable_idx:
                    f_in_np = m_list[idx].input_feat.data.cpu().numpy()
                    f_out_np = m_list[idx].output_feat.data.cpu().numpy()
                    # conv
                    if len(f_in_np.shape) == 4:
                        # we do not prune depthwise-conv
                        if m_list[idx].groups == 1:
                            # normal conv: 1x1, 3x3,5x5
                            # f_in2save shape is [B*sample_num, C_in, KxK]
                            # f_out2save shape is [B*samle_num, C_out]
                            f_in2save, f_out2save = feature_sample(f_in_np,
                                                                   f_out_np,
                                                                   m_list[idx].kernel_size,
                                                                   m_list[idx].padding,
                                                                   m_list[idx].stride,
                                                                   # batch size
                                                                   input.size(0),
                                                                   # sample point number
                                                                   self.n_points_per_layer)
                    # fc
                    else:
                        # f_in2save shape is [B*sample_num, C_in]
                        # f_out2save shape is [B*samle_num, C_out]
                        assert len(f_in_np.shape) == 2
                        f_in2save = f_in_np.copy()
                        f_out2save = f_out_np.copy()

                    if 'input_feat' not in self.layer_info_dict[idx]:
                        self.layer_info_dict[idx]['input_feat'] = f_in2save
                        self.layer_info_dict[idx]['output_feat'] = f_out2save
                    else:
                        self.layer_info_dict[idx]['input_feat'] = np.vstack(
                            (self.layer_info_dict[idx]['input_feat'], f_in2save))
                        self.layer_info_dict[idx]['output_feat'] = np.vstack(
                            (self.layer_info_dict[idx]['output_feat'], f_out2save))

        for idx in self.prunable_idx:
            print('Layer NO.{} {}'.format(idx, m_list[idx].__class__.__name__))
            print('\tinput_feat shape : {}'.format(self.layer_info_dict[idx]['input_feat'].shape))
            print('\toutput_feat shape : {}'.format(self.layer_info_dict[idx]['output_feat'].shape))


    def prune_layer(self, idx, sparsity_ratio):
        if sparsity_ratio == 0 or sparsity_ratio == 1:
            return

        if idx in self.prunable_idx:
            X = self.layer_info_dict[idx]['input_feat']
            Y = self.layer_info_dict[idx]['output_feat']
            op = list(self.model.modules())[idx]
            W = op.weight.data.cpu().numpy()
            n, c = W.shape[0], W.shape[1]
            c_new = int(c*(1-sparsity_ratio))
            keep_inds, keep_num = channel_pruning(X, Y, W, c_new, debug=True)
            W_rec = weight_reconstruction(X, Y, W, keep_inds, debug=True)

    def prune(self, ratio):
        pass
