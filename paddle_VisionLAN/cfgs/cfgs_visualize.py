# coding:utf-8
import paddle
import paddle.optimizer as optim
import os
from data.dataset_scene_vis import *
from paddle.vision import transforms


global_cfgs = {
    'state': 'Train',
    'epoch': 8,
    'show_interval': 200,
    'test_interval': 2000,
    'step': 'LA',
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
        'mask_id': 6 # 1 2 or n
    },
    'dataloader_train': {
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 32,
        'pin_memory': True,
    },

    'dataset_test': lmdbDataset,
    'dataset_test_args': {
        'roots': [
            '/app/wht/lc/VisionLAN/datasets/evaluation/Sumof6benchmarks'
        ],
        'img_height': 64,
        'img_width': 256,
        'transform': transforms.Compose([transforms.ToTensor()]),
        'global_state': 'Test',
    },

    'dataloader_test': {
        'batch_size': 64,
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
    'init_state_dict': './output/LA/final.pth'
}

