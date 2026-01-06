#!/usr/bin/env python
# -*- coding: utf-8 -*-

# CUDA TEST
import torch
from platform import node

print('Hostname:', node())
print('PyTorch:', torch.__version__)
print(torch.cuda.get_device_name())

x = torch.Tensor([1.0])
x_cuda = x.cuda()
print(x_cuda)

# CUDNN TEST
from torch.backends import cudnn
print('cudnn:',cudnn.is_acceptable(x_cuda))