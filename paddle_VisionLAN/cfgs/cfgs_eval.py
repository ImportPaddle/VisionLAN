# coding:utf-8
import paddle
import paddle.optimizer as optim
import os

from data.dataset_scene import *
from paddle.vision import transforms
from VisionLAN import *

from VisionLAN import VisionLAN

global_cfgs = {
    'state': 'Test',
    'epoch': 8,
    'show_interval': 200,
    'test_interval': 2000,
    'step': 'LA', # 'LF_1' 'LF_2' 'LA'
}

dataset_cfgs = {
    'dataset_train': lmdbDataset,
    'dataset_train_args': {
        'roots': [
            '/app/wht/lc/VisionLAN/datasets/training/ST',
            '/app/wht/lc/VisionLAN/datasets/training/MJ/MJ_test',
            '/app/wht/lc/VisionLAN/datasets/training/MJ/MJ_train',
            '/app/wht/lc/VisionLAN/datasets/training/MJ/MJ_valid',
        ],
        'img_height': 64,
        'img_width': 256,
        'transform': transforms.Compose([transforms.ToTensor()]),
        'global_state': 'Train',
    },
    'dataloader_train': {
        'batch_size': 16,
        'shuffle': True,
        'num_workers': 32,
        'pin_memory': True,
    },

    'dataset_test': lmdbDataset,
    'dataset_test_all': {
        'roots': [
            '/app/wht/lc/VisionLAN/datasets/evaluation/Sumof6benchmarks'
        ],
        'img_height': 64,
        'img_width': 256,
        'transform': transforms.Compose([transforms.ToTensor()]),
        'global_state': 'Test',
    },

    'dataset_test_args': {
        'roots': [
            '/app/wht/lc/VisionLAN/datasets/evaluation/IIIT5K',

        ],
        'img_height': 64,
        'img_width': 256,
        'transform': transforms.Compose([transforms.ToTensor()]),
        'global_state': 'Test',
    },

    'dataset_test_argsIC13': {
        'roots': [
            '/app/wht/lc/VisionLAN/datasets/evaluation/IC13',
        ],
        'img_height': 64,
        'img_width': 256,
        'transform': transforms.Compose([transforms.ToTensor()]),
        'global_state': 'Test',
    },

    'dataset_test_argsIC15': {
        'roots': [
            '/app/wht/lc/VisionLAN/datasets/evaluation/IC15',
        ],
        'img_height': 64,
        'img_width': 256,
        'transform': transforms.Compose([transforms.ToTensor()]),
        'global_state': 'Test',
    },

    'dataset_test_argsSVT': {
        'roots': [
            '/app/wht/lc/VisionLAN/datasets/evaluation/SVT',
        ],
        'img_height': 64,
        'img_width': 256,
        'transform': transforms.Compose([transforms.ToTensor()]),
        'global_state': 'Test',
    },
    'dataset_test_argsSVTP': {
        'roots': [
                  '/app/wht/lc/VisionLAN/datasets/evaluation/SVTP'
                  ],
        'img_height': 64,
        'img_width': 256,
        'transform': transforms.Compose([transforms.ToTensor()]),
        'global_state': 'Test',
    },

    'dataset_test_argsCUTE': {
        'roots': [
                '/app/wht/lc/VisionLAN/datasets/evaluation/CUTE'
                  ],
        'img_height': 64,
        'img_width': 256,
        'transform': transforms.Compose([transforms.ToTensor()]),
        'global_state': 'Test',
    },

    'dataloader_test': {
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 16,
        'pin_memory': True,
    },
    'case_sensitive': False,
    'dict_dir' : './dict/dic_36.txt'
}

net_cfgs = {
    'VisualLAN': VisionLAN,
    'args': {
        'strides': [(1,1), (2,2), (2,2), (2,2), (1,1), (1,1)],
        'input_shape': [3, 64, 256], # C x H x W
    },

    'init_state_dict': './output/LA/final.pth',
}


