import torch
import torch.nn as nn


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):

    def __init__(self, features, linears, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, linears[0]),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(linears[0], linears[1]),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(linears[1], num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'P': [17, 17, 'M', 37, 47, 'M', 83, 89, 106, 'M', 175, 192, 227, 'M', 512, 512, 512, 'M'],
}
cfgs_linear = {
    'A': [4096, 4096],
    'B': [4096, 4096],
    'D': [4096, 4096],
    'E': [4096, 4096],
    'P': [4096, 4096],
}


def _vgg(arch, cfg, batch_norm):
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), cfgs_linear[cfg])
    return model


def vgg11():
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg('vgg11', 'A', False)


def vgg11_bn():
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg('vgg11_bn', 'A', True)


def vgg13():
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg('vgg13', 'B', False)


def vgg13_bn():
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg('vgg13_bn', 'B', True)


def vgg16():
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg('vgg16', 'D', False)


def vgg16_bn():
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg('vgg16_bn', 'D', True)

def vgg16_bn_p():
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg('vgg16_bn', 'P', True)


def vgg19():
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg('vgg19', 'E', False)


def vgg19_bn():
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg('vgg19_bn', 'E', True)


if __name__ == '__main__':
    from thop import profile
    net = vgg16_bn()
    for i, m in enumerate(net.modules()):
        if type(m) == torch.nn.Conv2d:
            print(i, m)
    # # print(net(torch.randn((1,3,224,224))))
    # input = torch.randn((1,3,224,224))
    # flops, params = profile(net, (input,), verbose=False)
    # print("=>PARAMS: {:.3f} M, FLOPS: {:.3f} M".format(params / 1e6, flops / 1e6))


