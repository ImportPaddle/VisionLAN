# coding:utf-8
import sys

import torch
import torch.optim as optim
import os
from torchvision import transforms

DIR = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(DIR, '../'))
from VisionLAN import VisionLAN

global_cfgs = {
    'state': 'Train',
    'epoch': 8,
    'show_interval': 200,
    'test_interval': 2000,
    'step': 'LA',
}
dataset_cfgs = {
    'case_sensitive': False,
    'dict_dir': '/app/wht/lc/VisionLAN/pytorch_VisionLAN/dict/dic_36.txt'
}
net_cfgs = {
    'VisualLAN': VisionLAN,
    'args': {
        'strides': [(1, 1), (2, 2), (2, 2), (2, 2), (1, 1), (1, 1)],
        'input_shape': [3, 64, 256],  # C x H x W
    },

    'init_state_dict': None,
    # 'init_state_dict': './output/LF_2/LF_2.pth',
}
optimizer_cfgs = {
    'optimizer_0': optim.Adam,
    'optimizer_0_args': {
        'lr': 0.0001,
    },
    'optimizer_0_scheduler': optim.lr_scheduler.MultiStepLR,
    'optimizer_0_scheduler_args': {
        'milestones': [6],
        'gamma': 0.1,
    },
}
saving_cfgs = {
    'saving_epoch_interval': 1,
    'saving_path': './output/LA/',

}
