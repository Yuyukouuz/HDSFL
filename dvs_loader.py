# -*- coding: utf-8 -*-
"""
@author: QgZhan
@contact: zhanqg@foxmail.com
@file: cifar.py
@time: 2022/4/19 11:19
"""

import os
import torch
import random
from torchvision import datasets, transforms
from torch.utils.data import random_split, Dataset
from typing import Any, Callable, Optional, Tuple
#from prefetch_generator import BackgroundGenerator


# your own data dir
USER_NAME = 'zhan'
DIR = {'CIFAR10': f'/data/{USER_NAME}/Event_Camera_Datasets/CIFAR10/cifar10',
       'CIFAR10DVS': f'/data/{USER_NAME}/Event_Camera_Datasets/CIFAR10/CIFAR10DVS/temporal_effecient_training_0.84_mat',
       'CIFAR10DVS_CATCH': f'/data/{USER_NAME}/Event_Camera_Datasets/CIFAR10/CIFAR10DVS_dst_cache',
       }


'''class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())'''


def get_cifar10_DVS(batch_size=64, train_set_ratio=1, size=32):
    """
    get the train loader and test loader of cifar10.
    :param batch_size:
    :param train_set_ratio: the real used train set ratio
    :param size:
    :return: train_loader, test_loader
    """
    dvs_path = DIR['CIFAR10DVS']
    train_path = DIR['CIFAR10DVS'] + '/train'
    test_path = DIR['CIFAR10DVS'] + '/test'

    dvs_trans = transforms.Compose([transforms.Resize(size),
                                    # transforms.RandomCrop(48, padding=4),
                                    # transforms.RandomHorizontalFlip(),  # 随机水平翻转
                                    transforms.ToTensor(),
                                   ])

    train_set = DVSCIFAR10(DIR['CIFAR10DVS'], train=True, dvs_transform=dvs_trans)
    test_set= DVSCIFAR10(DIR['CIFAR10DVS'], train=False, dvs_transform=dvs_trans)

    # take train set by train_set_ratio
    if train_set_ratio < 1.0:
        n_train = len(train_set)  # 60000
        split = int(n_train * train_set_ratio)  # 60000*0.9 = 54000
        train_set, _ = random_split(train_set, [split, n_train-split], generator=torch.Generator().manual_seed(1000))

    return train_set, test_set


class DVSCifar10(Dataset):
    # This code is form https://github.com/Gus-Lab/temporal_efficient_training
    def __init__(self, root, train=True, shape=48, transform=True, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize = transforms.Resize(size=(shape, shape))  # 48 48
        self.tensorx = transforms.ToTensor()
        self.imgx = transforms.ToPILImage()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        #data, target = torch.load(self.root + '/{}.pt'.format(index))
        data, target = torch.load(self.root + '/{}_mat.pt'.format(index))
        # if self.train:
        new_data = []
        for t in range(data.size(0)):
            new_data.append(self.tensorx(self.resize(self.imgx(data[t, ...]))))
        data = torch.stack(new_data, dim=0)

        if self.transform:
            flip = random.random() > 0.5
            if flip:
                data = torch.flip(data, dims=(3,))
            off1 = random.randint(-5, 5)
            off2 = random.randint(-5, 5)
            data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target.long().squeeze(-1)

    def __len__(self):
        #return len(os.listdir(self.root))
        return 8330


class DVSCIFAR10(Dataset):
    def __init__(
            self,
            dvs_root: str,
            train: bool = True,
            dvs_train_set_ratio: float = 1.0,
            dvs_transform: Optional[Callable] = None,
    ) -> None:

        self.train = train  # training set or test set
        self.dvs_train_set_ratio = dvs_train_set_ratio
        self.dvs_transform = dvs_transform
        self.imgx = transforms.ToPILImage()
        if self.train:
            self.dvs_root = os.path.join(dvs_root, 'train')
        else:
            self.dvs_root = os.path.join(dvs_root, 'test')

        """
        准备DVS数据
        """
        dvs_class_list = sorted(os.listdir(self.dvs_root))
        self.dvs_data = []
        self.dvs_targets = []
        for i, dvs_class in enumerate(dvs_class_list):
            dvs_class_path = os.path.join(self.dvs_root, dvs_class)
            file_list = sorted(os.listdir(dvs_class_path))
            for file_name in file_list:
                self.dvs_data.append(os.path.join(dvs_class_path, file_name))
                self.dvs_targets.append(i)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # dvs图像的transform
        dvs_img = torch.load(self.dvs_data[index])
        if self.dvs_transform is not None:
            dvs_img = self.dvs_trans(dvs_img)
        target = self.dvs_targets[index]  # 输入索引对应dvs图像的类别

        return dvs_img, target

    def __len__(self) -> int:
        return len(self.dvs_data)

    def dvs_trans(self, dvs_img):
        transformed_dvs_img = []
        for t in range(dvs_img.size(0)):
            data = self.imgx(dvs_img[t, ...])
            transformed_dvs_img.append(self.dvs_transform(data))
        dvs_img = torch.stack(transformed_dvs_img, dim=0)

        if self.train:
            flip = random.random() > 0.5
            if flip:
                dvs_img = torch.flip(dvs_img, dims=(3,))
            off1 = random.randint(-5, 5)
            off2 = random.randint(-5, 5)
            dvs_img = torch.roll(dvs_img, shifts=(off1, off2), dims=(2, 3))
        return dvs_img
