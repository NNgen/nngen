from __future__ import absolute_import
from __future__ import print_function

import time
import copy
import datetime

import torch
import torchvision
import torchvision.transforms as transforms


# --------------------------------------
# Model definition and customization
# --------------------------------------

import torch.nn as nn
import torch.nn.functional as F

net = torchvision.models.vgg11_bn()

net.avgpool = nn.Identity()

#net.classifier[0] = nn.Linear(512, 4096)
#net.classifier[6] = nn.Linear(4096, 10)

net.classifier = nn.Sequential(
    nn.Linear(in_features=512, out_features=1024, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(in_features=1024, out_features=1024, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(in_features=1024, out_features=10, bias=True),
)

print(net)


# --------------------------------------
# to ONNX
# --------------------------------------

batch_size = 1
dummy_input = torch.randn(batch_size, 3, 32, 32)

#torch.onnx.export(net, dummy_input, 'vgg11_cifar10.onnx', verbose=True)
torch.onnx.export(net, dummy_input, 'vgg11_cifar10.onnx')
