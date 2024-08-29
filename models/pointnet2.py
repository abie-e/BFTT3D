# based on: https://github.com/fxia22/pointnet.pytorch/blob/master/utils/train_classification.py
from __future__ import print_function
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from pointnet2_pyt.pointnet2.models.pointnet2_msg_cls import Pointnet2MSG

DATASET_NUM_CLASS = {
    'mn40': 40,
    'modelnet40_c': 40,
    'scan': 15,
}
class PointNet2(nn.Module):

    def __init__(self, dataset, version_cls=1.0):
        super().__init__()
        num_class = DATASET_NUM_CLASS[dataset]
        self.model = Pointnet2MSG(num_classes=num_class, input_channels=0, use_xyz=True, version=version_cls)


    def forward(self, pc, normal=None, cls=None):
        logit = self.model(pc)
        return logit
