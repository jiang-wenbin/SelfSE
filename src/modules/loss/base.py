import logging
import torch
from abc import ABC, abstractmethod

class BaseLoss(ABC, torch.nn.Module):
    def __init__(self, loss_conf):
        super().__init__()
        self.loss_conf = loss_conf
        
    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError