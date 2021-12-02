# ------------------------------------------------------------------------------
# Copyright (c) by contributors
# Licensed under the MIT License.
# Written by Haiping Wu
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torchvision
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.resnet import model_urls
from torchvision.models.resnet import BasicBlock, Bottleneck

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50']


class ResNet(torchvision.models.resnet.ResNet):
    def __init__(self, block, layers, **kwargs):
        if 'dropout_rate' in kwargs:
            dropout_rate = kwargs['dropout_rate']
            del kwargs['dropout_rate']
        else:
            dropout_rate = 0.0
        if 'return_inter' in kwargs:
            self.return_inter = kwargs['return_inter']
            del kwargs['return_inter']
        else:
            self.return_inter = False
        if 'return_res4' in kwargs:
            self.return_res4 = kwargs['return_res4']
            del kwargs['return_res4']
        else:
            self.return_res4 = False
        super(ResNet, self).__init__(block, layers, **kwargs)
        self.dropout = nn.Dropout(p=dropout_rate)

        if self.return_res4:
            del self.layer4
            self.fc = nn.Linear(256, self.fc.weight.shape[0])

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if not self.return_res4:
            x = self.layer4(x)

        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        x = self.dropout(feat)
        x = self.fc(x)

        if self.return_inter:
            return x, feat
        else:
            return x

    def forward(self, x):
        return self._forward_impl(x)


class VideoClassifier(nn.Module):
    def __init__(self, base, num_classes, **kwargs):
        super(VideoClassifier, self).__init__()
        self.base = base
        # self.fc = nn.Linear(128 * 20, num_classes)
        self.fc = nn.Linear(128, num_classes)

        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()

    def forward(self, x):
        # x: batch_size x num_frames x c x h x w
        bs, num_frames, c, h, w = x.shape
        # print(x.shape)
        x = x.view(bs * num_frames, c, h, w)
        x = self.base(x)

        x = x.view(bs, num_frames, -1)
        x = x.mean(dim=1)

        # x = x.view(bs, -1)


        x = self.fc(x)
        return x

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)
