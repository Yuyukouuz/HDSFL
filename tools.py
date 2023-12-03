import socket
import pickle
import json
import os
import random
import threading
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from collections import OrderedDict
from spikingjelly.clock_driven import functional

from dataset_config import *
from compress import spike_tensor_compress, spike_tensor_depress
from CKA_loss import linear_CKA

from dataset_config import *
if net_name == 'ANN':
    from CNN import SCNN
elif net_name == 'SNN':
    from SCNN import SCNN
elif net_name == 'SVGG':
    from SVGG import SCNN
elif net_name == 'newSVGG':
    from newVGG import SCNN

class ConfusionMatrixSave: #保存混淆矩阵类
    def __init__(self, save_name):
        self.confusion_matrix_list = []
        self.path = f'./running_save/confusion-matrix_{save_name}'

    def add_matrix(self, confusion_matrix):
        self.confusion_matrix_list.append(confusion_matrix)

    def save_matrix(self):
        if not os.path.isdir('running_save'):
            os.mkdir('running_save')
        state = {'confusion_matrix_list': self.confusion_matrix_list,
                 'frame': frame,
                 'dataset_name': dataset_name}
        torch.save(state, self.path)


class SimulateSave:  #未使用
    def __init__(self, save_name):
        self.record_list = []
        self.path = f'./running_save/simulation_{save_name}'

    def add_record(self, record):
        self.record_list.append(record)

    def save_record_list(self):
        if not os.path.isdir('running_save'):
            os.mkdir('running_save')
        state = {'confusion_matrix_list': self.confusion_matrix_list,
                 'frame': frame,
                 'dataset_name': dataset_name}
        torch.save(state, self.path)


def saves(best_acc, best_epoch, recorder, names='spiking_model', model=None, pre_name='s',):  #保存网络参数和其他数据

    if model is not None:
        state = {
            'net': model,
            'best_acc': best_acc,
            'best_epoch': best_epoch,
            'record': recorder,
            'configs':get_configs(),
        }
    else:
        state = {
            'best_acc': best_acc,
            'best_epoch':best_epoch,
            'record': recorder,
            'configs': get_configs(),
        }

    if not os.path.isdir('running_save'):
        os.mkdir('running_save')
    if net_name == 'SNN':
        save_name = './running_save/' + pre_name + dataset_name + '_' + shareset_name + '_' + names
    else:
        save_name = './running_save/' + pre_name + dataset_name + '_' + shareset_name + '_' + net_name + '_' + frame + '_' + names
    # while os.path.exists(save_name + '.t7'):
    #    save_name += '_new'
    torch.save(state, save_name)



def crossentropy(outputs:torch.Tensor, target:torch.Tensor):  #未使用
    tensor_size = outputs.size() #batch size * class
    loss_sum = torch.tensor([0.0])
    for b in range(tensor_size[0]):
        local_sum = 0
        for c in range(tensor_size[1]):
            local_sum += target[b][c] * torch.log(outputs[b][c])
        local_sum = -local_sum
        loss_sum += local_sum
    return loss_sum / tensor_size[0]


def my_loss(outputs:torch.Tensor, target:torch.Tensor): #输入脉冲规格：(b, T, classes/*) 蒸馏损失函数
    logit_target = torch.mean(target, dim=0)
    logit_outputs = torch.mean(outputs, dim=0) #(b, classes/*)
    cross_loss = - torch.sum(F.log_softmax(logit_target, dim=1) * F.softmax(logit_outputs, dim=1))
    #return (1 - linear_CKA(outputs, target)) + 0.1 * cross_loss
    return torch.mean(torch.pow((outputs - target), 2)) + 1 * cross_loss
    ##return torch.mean(torch.pow((logit_outputs - logit_target), 2))


def my_loss_ann(outputs:torch.Tensor, target:torch.Tensor):  #未使用
    cross_loss = - torch.sum(F.softmax(target, dim=1) * F.log_softmax(outputs, dim=1))
    return crossentropy(outputs, target)


def getloader(n_sample, datasets):  # 获取随机采样样本
    global batch_size
    indices = list(range(len(datasets)))
    random.shuffle(indices)
    dataset_ind = indices[:n_sample]
    train_loader = DataLoader(datasets, batch_size=batch_size,
                            sampler=SubsetRandomSampler(dataset_ind),
                            pin_memory=False)
    return train_loader


def list_to_tensor(l:list):
    s = l[0].size()
    s = torch.tensor(s)
    s = s.tolist()
    h = len(l)
    s.insert(0, h)
    ten = torch.zeros(s)
    for i in range(h):
        ten[i] = l[i]
    return ten


def print_net_state(state, avg=None):  #debug用
    key = 'static_conv.0.weight'
    #for key in state.keys():
    #    print(key)
    print(state[key].size())
    print(state[key][1])
    if avg is not None:
        print(avg[key][1])
    #exit()


def train0(net, dataset, epochs=1, lossf=my_loss_ann if net_name == 'ANN' else my_loss, onehot=False, num_classes=10,
          opti='Adam', lr=1e-3, bs=batch_share,
          spikes=None, temperature=1, device='cuda'):  #备用，未使用

    #optimizer = torch.optim.SGD(snn_net.parameters(), lr=learning_rate_s, momentum=0.9)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, )
    if opti == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, )

    running_loss = 0
    train_loader = DataLoader(dataset, batch_size=bs, pin_memory=False)
    #loss_function = torch.nn.CrossEntropyLoss()

    loss_list = []

    net.to(device)
    net.train()
    for e in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            net.zero_grad()
            optimizer.zero_grad()
            # print('获取训练数据')
            images = images.float().to(device)
            # print('前向传播')
            if spikes is not None:
                _, outputs = net(images, temperature)
            else:
                if frame == 'fedAvg':
                    outputs = net(images)
                elif frame == 'fedDis' or frame == 'fedAd1':
                    outputs, _ = net(images)

            # print('标签处理')
            if onehot: #是否需要预处理成one hot编码；使用交叉熵损失时不需要处理
                labels = F.one_hot(labels, num_classes).float()
            # print('损失函数计算')
            #_, label = spike.max(1)
            if spikes is not None: #蒸馏训练或普通训练
                spike = spikes[i]
                loss = lossf(outputs.cpu(), spike)  # cpu? 这里使用自定义损失函数进行蒸馏
            else:
                loss = lossf(outputs.cpu(), labels)
            loss_list.append(loss)
            running_loss += loss.item()
            # print('反向传播')
            loss.backward()
            # print('参数优化')
            optimizer.step()
            # 使用spikingjelly必须加这句
            functional.reset_net(net)
            torch.cuda.empty_cache()
        print('--已完成第%d轮训练，当前loss=%.4f--' % (e + 1, loss))

    del train_loader
    torch.cuda.empty_cache()
    return loss_list


def train(net, dataset, epochs=1, lossf=my_loss_ann if net_name == 'ANN' else my_loss, onehot=False,
          num_classes=20 if dataset_name == 'cifar100' else 10,
          opti='Adam', lr=1e-3, bs=batch_share,
          spikes=None, temperature=1, layer = 'all', device='cuda'):  #网络训练函数。蒸馏训练和本地训练都可以用这个函数

    #optimizer = torch.optim.SGD(snn_net.parameters(), lr=learning_rate_s, momentum=0.9)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, )
    if opti == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, )

    running_loss = 0
    train_loader = DataLoader(dataset, batch_size=bs, pin_memory=False)
    #loss_function = torch.nn.CrossEntropyLoss()

    loss_list = []

    net.to(device)
    net.train()
    for e in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            if dataset_name == 'cifar100':
                labels = fine_to_coarse(labels)

            net.zero_grad()
            optimizer.zero_grad()
            # print('获取训练数据')
            images = images.float().to(device)
            # print('前向传播')
            if spikes is not None:
                if layer == 'all':
                    _, outputs = net(images, temperature)
                elif layer == 'hint':
                    _, outputs = net.feature_forward(images)
            else:
                if frame == 'fedAvg':
                    outputs = net(images)
                elif frame == 'fedDis' or frame == 'fedAd1':
                    outputs, _ = net(images)

            # print('标签处理')
            if onehot: #是否需要预处理成one hot编码；使用交叉熵损失时不需要处理
                labels = F.one_hot(labels, num_classes).float()
            # print('损失函数计算')
            #_, label = spike.max(1)
            if spikes is not None: #蒸馏训练或普通训练
                spike = spikes[i]
                loss = lossf(outputs.cpu(), spike)  # cpu? 这里使用自定义损失函数进行蒸馏
            else:
                loss = lossf(outputs.cpu(), labels)
            loss_list.append(loss)
            running_loss += loss.item()
            # print('反向传播')
            loss.backward()
            # print('参数优化')
            optimizer.step()
            torch.cuda.empty_cache()
        print('--已完成第%d轮训练，当前loss=%.4f--' % (e + 1, loss))

    del train_loader
    torch.cuda.empty_cache()
    return loss_list


def train_distillation(net, spikes_all, spikes_hint, dataset, epochs=1, onehot=False,
        lossf_all=my_loss_ann if net_name == 'ANN' else my_loss, lossf_hint=torch.nn.MSELoss(), k=1,
          opti='Adam', lr=1e-3,
          num_classes=20 if dataset_name == 'cifar100' else 10,
          bs=batch_share,
          temperature=1, device='cuda'):  #复合蒸馏训练函数，同时完成提示层蒸馏和输出蒸馏

    #optimizer = torch.optim.SGD(snn_net.parameters(), lr=learning_rate_s, momentum=0.9)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, )
    if opti == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, )

    running_loss = 0
    train_loader = DataLoader(dataset, batch_size=bs, pin_memory=False)
    #loss_function = torch.nn.CrossEntropyLoss()

    loss_list = []

    net.to(device)
    net.train()
    for e in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            if dataset_name == 'cifar100':
                labels = fine_to_coarse(labels)

            net.zero_grad()
            optimizer.zero_grad()
            # print('获取训练数据')
            images = images.float().to(device)
            # print('前向传播')
            outputs_all, outputs_hint = net.forward_merge(images)

            # print('标签处理')
            if onehot: #是否需要预处理成one hot编码；使用交叉熵损失时不需要处理
                labels = F.one_hot(labels, num_classes).float()
            # print('损失函数计算')
            #_, label = spike.max(1)
            spike_all = spikes_all[i]
            spike_hint = spikes_hint[i]
            loss_all = lossf_all(outputs_all.cpu(), spike_all)  # cpu? 这里使用自定义损失函数进行蒸馏
            loss_hint = lossf_hint(outputs_hint.cpu(), spike_hint)
            loss = loss_all + k * loss_hint

            loss_list.append(loss)
            running_loss += loss.item()
            # print('反向传播')
            loss.backward()
            # print('参数优化')
            optimizer.step()
            torch.cuda.empty_cache()
        print('--已完成第%d轮训练，当前loss=%.4f--' % (e + 1, loss))

    del train_loader
    torch.cuda.empty_cache()
    return loss_list



def quick_test(net, dataset, use=0, bs=batch_share,
               tips='Test Accuracy of the model:', device='cuda'):  #快捷测试 仅输出准确率不保存
    correct = 0
    total = 0

    if use>0:
        test_loader = DataLoader(sub(dataset, use), batch_size=bs, shuffle=True, num_workers=2)
    else:
        test_loader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=2)

    net.to(device)
    functional.reset_net(net)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if dataset_name == 'cifar100':
                targets = fine_to_coarse(targets)
            inputs = inputs.to(device)
            # optimizer.zero_grad()
            if frame == 'fedAvg':
                outputs = net(inputs)
            elif frame == 'fedDis' or frame == 'fedAd1':
                outputs, _ = net(inputs)
            # labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
            # loss = criterion(outputs.cpu(), labels_)
            _, predicted = outputs.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
            functional.reset_net(net)
            if batch_idx % 100 == 0:
                acc = 100. * float(correct) / float(total)
                print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)

    acc = correct / total
    print(tips + ' %.3f' % (100 * acc))
    return acc


def compute_spikes(snn_net, dataset_share, batch=batch_share, layer='all', device='cuda'):
    #计算脉冲张量
    test_loader = DataLoader(dataset_share, batch_size=batch, num_workers=2)
    snn_net.to(device)
    #functional.reset_net(snn_net)

    '''if net == 'SNN':
        spike_tensor = torch.zeros([len(dataset_share)//batch, 8, batch, num_classes])
    if net == 'ANN':
        spike_tensor = torch.zeros([len(dataset_share) // batch, batch, num_classes])'''

    spike_tensor_list = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader)):
            inputs = inputs.to(device)
            if layer == 'hint':
                _, outputs = snn_net.feature_forward(inputs)
            elif layer == 'all':
                _, outputs = snn_net(inputs)
            #spike_tensor[batch_idx] = outputs
            spike_tensor_list.append(outputs)

    spike_size = [len(spike_tensor_list)]
    spike_size.extend(spike_tensor_list[0].size())
    spike_tensor = torch.zeros(spike_size)

    for i in range(len(spike_tensor_list)):
        spike_tensor[i] = spike_tensor_list[i]

    return spike_tensor


def get_confusion_matrix(net, dataset, classes=10, bs=64):  #获取混淆矩阵数据
    confusion_matrix = torch.zeros([classes, classes], dtype=torch.int)
    test_loader = DataLoader(dataset, batch_size=bs, num_workers=2)
    net = net.cuda()
    functional.reset_net(net)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if dataset_name == 'cifar100':
                targets = fine_to_coarse(targets)
            inputs = inputs.cuda()
            # optimizer.zero_grad()
            if frame == 'fedAvg':
                outputs = net(inputs)
            elif frame == 'fedDis' or frame == 'fedAd1':
                outputs, _ = net(inputs)
            _, predicted = outputs.cpu().max(1)
            for i in range(len(predicted)):
                confusion_matrix[targets[i]][predicted[i]] += 1
            functional.reset_net(net)

    return confusion_matrix