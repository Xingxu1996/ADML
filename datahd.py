# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 14:13:12 2018

@author: yxx_h
"""
import random

import torch.utils.data as data
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import imread
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import os.path
import glob
import numpy as np
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, root, datatxt, train=True, transform=None, target_transform=None):
        self.train = train
        fh = open(root + datatxt, 'r')
        lines = fh.readlines()
        # if train:
        #     random.shuffle(lines)
        imgs = []
        train_labels = []
        train_data = []
        for line in lines:
            line = line.rstrip()
            words = line.split()
            train_data.append((words[0]))
            imgs.append((words[0], int(words[1])))
            train_labels.append((int(words[1])))
        self.train_data = train_data
        self.train_labels = torch.IntTensor(train_labels)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        root = '/home/ubuntu5/yxx/Benchmark_split/Fi/fi/'
        img = Image.open(root + fn).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        temp = np.array((1,))
        temp[0] = label
        label = torch.LongTensor(temp)[0]
        return img, label

    def __len__(self):
        return len(self.imgs)

    def __labels__(self):
        imgs = np.array(self.imgs)
        return imgs[:, 1]
