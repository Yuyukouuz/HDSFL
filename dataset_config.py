import torchvision
import torchvision.transforms as transforms
import torch.utils.data
from torch.utils.data import Subset
import random

from global_configs import *
from dvs_loader import get_cifar10_DVS

data_path = '/data/lzt/datasets/'

used = {'mnist':0, 'fmnist':0, 'emnist':0, 'cifar10':0, 'cifar100':0, 'cinic10':0, 'cinic10_val':0, 'cifar10_dvs':0}
if used.get(shareset_name) is not None:
    used[shareset_name] += share_clip


def reshuffle(dataset):  #数据集打乱
    inds = list(range(len(dataset)))
    random.shuffle(inds)
    return Subset(dataset, inds)


def sub(dataset, num_subset=1000):  #顺序打乱并取子集
    sum_dataset = len(dataset)
    if num_subset >= sum_dataset:
        return dataset
    indices = list(range(sum_dataset))
    random.shuffle(indices)
    dataset_ind = indices[:num_subset]
    subs = Subset(dataset, dataset_ind)
    return subs


if dataset_name == 'cifar100':
    data_path_cifar100 = data_path + 'cifar100'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trains_cifar100 = shares_cifar100 = torchvision.datasets.CIFAR100(
        root=data_path_cifar100, train=True, download=False, transform=transform)  #训练集指针，下同
    tests_cifar100 = torchvision.datasets.CIFAR100(
        root=data_path_cifar100, train=False, download=False, transform=transform)  #测试集指针，下同


if dataset_name == 'cifar10':
    data_path_cifar10 = data_path + 'cifar10'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trains_cifar10 = shares_cifar10 = torchvision.datasets.CIFAR10(
        root=data_path_cifar10, train=True, download=False, transform=transform)
    tests_cifar10 = torchvision.datasets.CIFAR10(
        root=data_path_cifar10, train=False, download=False, transform=transform)


if dataset_name == 'mnist':
    data_path_mnist = data_path + 'mnist'
    trains_mnist = shares_mnist = torchvision.datasets.MNIST(
        root=data_path_mnist, train=True, download=False, transform=transforms.ToTensor())
    tests_mnist = torchvision.datasets.MNIST(
        root=data_path_mnist, train=False, download=False, transform=transforms.ToTensor())


if dataset_name == 'fmnist':
    data_path_fmnist = data_path + 'fmnist'
    trains_fmnist = shares_fmnist = torchvision.datasets.FashionMNIST(
        root=data_path_fmnist, train=True, download=True, transform=transforms.ToTensor())
    tests_fmnist = torchvision.datasets.FashionMNIST(
        root=data_path_fmnist, train=False, download=True, transform=transforms.ToTensor())


if dataset_name == 'cinic10':
    data_path_cinic = data_path + 'cinic10'
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    trains_cinic10 = shares_train_cinic10 = torchvision.datasets.ImageFolder(data_path_cinic + '/train',
            transform=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean,std=cinic_std)]))
    tests_cinic10 = torchvision.datasets.ImageFolder(data_path_cinic + '/test',
            transform=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean,std=cinic_std)]))
    tests_cinic10 = reshuffle(tests_cinic10)
    shares_cinic10 = torchvision.datasets.ImageFolder(data_path_cinic + '/valid',
            transform=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean,std=cinic_std)]))  #公共数据集指针


if dataset_name == 'cifar10_dvs':
    trains_dvs, tests_dvs = get_cifar10_DVS(size=32)
    shares_dvs = trains_dvs


if dataset_name == 'emnist':
    data_path_emnist = data_path + 'emnist'
    trains_emnist = torchvision.datasets.EMNIST(
    root=data_path_emnist, split='bymerge', train=True, download=False, transform=transforms.ToTensor())
        #root=data_path_emnist, split='balanced', train=True, download=True, transform=transforms.ToTensor())
    tests_emnist = torchvision.datasets.EMNIST(
        root=data_path_emnist, split='balanced', train=False, download=False, transform=transforms.ToTensor())
    shares_emnist = torchvision.datasets.EMNIST(
    root=data_path_emnist, split='balanced', train=True, download=True, transform=transforms.ToTensor())


if dataset_name == 'fmnist':  #参数处理
    dataset_train = trains_fmnist
    dataset_test = tests_fmnist
elif dataset_name == 'emnist':
    dataset_train = trains_emnist
    dataset_test = tests_emnist
elif dataset_name == 'mnist':
    dataset_train = trains_mnist
    dataset_test = tests_mnist
elif dataset_name == 'cifar10':
    dataset_train = trains_cifar10
    dataset_test = tests_cifar10
elif dataset_name == 'cifar100':
    dataset_train = trains_cifar100
    dataset_test = tests_cifar100
elif dataset_name == 'cinic10':
    dataset_train = trains_cinic10
    dataset_test = tests_cinic10
elif dataset_name == 'cifar10_dvs':
    dataset_train = trains_dvs
    dataset_test = tests_dvs
if n_testDataset > 0:
    dataset_test = sub(dataset_test, n_testDataset)

if shareset_name == 'fmnist':
    dataset_shares = shares_fmnist
elif shareset_name == 'emnist':
    dataset_shares = shares_emnist
elif shareset_name == 'mnist':
    dataset_shares = shares_mnist
elif shareset_name == 'cifar10':
    dataset_shares = shares_cifar10
elif shareset_name == 'cifar100':
    dataset_shares = shares_cifar100
elif shareset_name == 'cinic10':
    dataset_shares = shares_train_cinic10
elif shareset_name == 'cinic10_val':
    dataset_shares = shares_cinic10
elif shareset_name == 'cifar10_dvs':
    dataset_shares = shares_dvs
batch_share = 50  #公共数据集batch size


def get_share_set_byinds(inds): #根据下标取公共数据集
    return Subset(dataset_shares, inds)


def get_train_set_byinds(inds):  #根据下标取训练集
    return Subset(dataset_train, inds)


class Distributer:  #数据集下标分配器
    def __init__(self):
        #self.set = dataset_train
        self.length = len(dataset_train)
        self.indices = list(range(self.length))
        random.shuffle(self.indices)
        if dataset_train == dataset_shares:
            self.share_ind = self.indices[0 : share_clip]
            self.next = share_clip
        else:
            share_length = len(dataset_shares)
            share_indices = list(range(share_length))
            random.shuffle(share_indices)
            self.share_ind = share_indices[0: share_clip]
            self.next = 0

    def get_train_set(self, num):
        last_ind = (self.next + num) % self.length
        sub_ind = self.indices[self.next : last_ind]
        subset = Subset(dataset_train, sub_ind)
        self.next = last_ind
        return subset

    def get_train_inds(self, num):
        print(num)
        print(self.length)
        if self.next + num == self.length:
            last_ind = self.length
        else:
            last_ind = (self.next + num) % self.length
        print(self.next)
        print(last_ind)
        sub_ind = self.indices[self.next : last_ind]
        self.next = last_ind
        return sub_ind

    def get_share_inds(self):
        return self.share_ind




def fine_to_coarse(fine_labels):  #cifar100用

    reflect = [4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9,
               7, 11, 6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12,
               14, 16, 9, 11, 5, 5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16,
               4, 17, 4, 2, 0, 17, 4, 18, 17, 10, 3, 2, 12, 12, 16, 12, 1,
               9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13, 16, 19, 2, 4, 6,
               19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13]
    coarse_labels = torch.zeros_like(fine_labels)
    for i in range(len(coarse_labels)):
        coarse_labels[i] = reflect[fine_labels[i]]
    return coarse_labels