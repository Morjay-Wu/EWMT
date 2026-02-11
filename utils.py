import sys
import logging
import csv
from pathlib import Path
import rasterio
import os

import torch
import torch.nn as nn


def make_tuple(x):
    if isinstance(x, int):
        return x, x
    if isinstance(x, list) and len(x) == 1:
        return x[0], x[0]
    return x


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_logger(logpath=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        if logpath is not None:
            file_handler = logging.FileHandler(logpath)
            file_handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
    return logger


def save_checkpoint(model, optimizer, path):
    if path.exists():
        path.unlink()
    model = model.module if isinstance(model, nn.DataParallel) else model
    state = {'state_dict': model.state_dict()}
    if optimizer:
        state = {'state_dict': model.state_dict(),
                 'optim_dict': optimizer.state_dict()}
    if isinstance(path, Path):
        torch.save(state, str(path.resolve()))
    else:
        torch.save(state, str(path.resolve()))


def load_checkpoint(checkpoint_path, model=None, optimizer=None):
    if not os.path.exists(checkpoint_path):
        print(f"警告: 检查点文件 {checkpoint_path} 不存在")
        return 0, 0

    print(f"加载检查点: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location='cpu')
    
    if model is not None:
        # 获取当前模型的状态字典
        current_state_dict = model.state_dict()
        # 获取预训练模型的状态字典
        pretrained_state_dict = state['state_dict']
        
        # 创建一个新的状态字典，只包含匹配的键
        new_state_dict = {}
        for k, v in pretrained_state_dict.items():
            if k in current_state_dict and v.shape == current_state_dict[k].shape:
                new_state_dict[k] = v
            else:
                print(f"跳过不匹配的参数: {k}")
        
        # 加载匹配的参数
        model.load_state_dict(new_state_dict, strict=False)
        print("模型参数加载完成")
    
    if optimizer is not None and 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])
        print("优化器状态加载完成")
    
    start_epoch = state.get('epoch', 0)
    best_psnr = state.get('best_psnr', 0)
    
    return start_epoch, best_psnr


def log_csv(filepath, values, header=None, multirows=False):
    empty = False
    if not filepath.exists():
        filepath.touch()
        empty = True

    with open(filepath, 'a') as file:
        writer = csv.writer(file)
        if empty and header:
            writer.writerow(header)
        if multirows:
            writer.writerows(values)
        else:
            writer.writerow(values)


def load_pretrained(model, pretrained, requires_grad=False):
    if isinstance(model, nn.DataParallel):
        model = model.module
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained)['state_dict']
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    if not requires_grad:
        for param in model.parameters():
            param.requires_grad = False


def save_array_as_tif(matrix, path, profile=None, prototype=None):
    assert matrix.ndim == 2 or matrix.ndim == 3
    if prototype:
        with rasterio.open(str(prototype)) as src:
            profile = src.profile
    with rasterio.open(path, mode='w', **profile) as dst:
        if matrix.ndim == 3:
            for i in range(matrix.shape[0]):
                dst.write(matrix[i], i + 1)
        else:
            dst.write(matrix, 1)