import torch
from torch.nn import init
from torch import nn
import os
from scipy.io import loadmat
import random
import numpy as np


def weight_init(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()


def train_test_split(gt: np.ndarray, size: int = None, ratio: int = None):
    """
    :param gt: 高光谱图像的groundtruth
    :param size: 每类样本的大小
    :return: train_gt(训练样本标签), test_gt(测试样本标签)
    """
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    test_gt[:] = gt[:]
    random.seed(666)
    for i in np.unique(gt):
        if i != 0:
            indices = list(zip(*np.nonzero(gt == i)))
            num = size if size else int(len(indices) * ratio)
            samples = random.sample(indices, min(num, len(indices)))
            x = tuple(zip(*samples))
            train_gt[x] = gt[x]
            test_gt[x] = 0
    return train_gt, test_gt


def pad(d_in, kernel, stride):
    r = (d_in - kernel) % stride
    if r == 0:
        padding=  0
    else:
        m = stride - r
        padding = m // 2 if m % 2 == 0 else (m + stride) // 2
    return padding


def cross_evaluation_split(gt, n):
    gt = np.copy(gt)
    tr_index, te_index = [], []
    n_per_class = np.sum(gt!=0) // gt.max() // n
    split = []
    for i in range(n):
        current = []
        for j in range(1, gt.max() + 1):
            indices = tuple(zip(*np.nonzero(gt == j)))
            if len(indices) >= 2 * n_per_class:
                sample = random.sample(indices, n_per_class)
            else:
                sample = random.sample(indices, len(indices))
            gt[tuple(zip(*sample))] = 0
            current += sample
        te_index.append(current)
        indices = list(zip(*np.nonzero(gt)))
        tr_index.append(split + indices)
        split += current
    return np.array(tr_index), np.array(te_index)








