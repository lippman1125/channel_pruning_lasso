import torch
class PruneConfig(object):
    def __init__(self):
        self.n_points_per_layer = 1
        self.prunable_layer_types = [torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear]
        self.calib_batch = 10
        self.device = 'cuda'


class LassoPruneConfig(PruneConfig):
    def __init__(self, model, ckpt, train_dataloader, val_dataloader=None):
        super(LassoPruneConfig, self).__init__()
        self.model = model
        self.ckpt = ckpt
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

