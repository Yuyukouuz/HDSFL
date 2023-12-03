import torch.nn as nn
import torch
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, layer

from global_configs import *

thresh = 0.25 # 阈值θ
lens = 0.5 # hyper-parameters of approximate function
decay = 0.1 #膜电位衰减
#batch_size  = 20

if dataset_name == 'emnist':
    num_classes = 47
elif dataset_name == 'cifar100':
    num_classes = 20
else:
    num_classes = 10

if sub_layer:
    cfg_cnn_mnist = [(1, 128, 1, 1, 3),
               (128, 256, 1, 1, 3),
               (256, 512, 1, 1, 3),
               (512, 256, 1, 1, 3),
               (256, 64, 1, 1, 3),
               (64, 16, 1, 1, 3), ]###
else:
    cfg_cnn_mnist = [(1, 128, 1, 1, 3),
                     (128, 256, 1, 1, 3),
                     (256, 512, 1, 1, 3),
                     (512, 256, 1, 1, 3),
                     (256, 64, 1, 1, 3),]
cfg_kernel_mnist = [28, 28, 14, 7, 7, 7, 7]
cfg_fc_mnist = [1024, 512, num_classes]


cfg_cnn_cifar = [(3, 128, 1, 1, 3),
           (128, 256, 1, 1, 3),
           (256, 512, 1, 1, 3),
           (512, 1024, 1, 1, 3),
           (1024, 512, 1, 1, 3),
            (512, 32, 1, 1, 3), ]###
cfg_kernel_cifar = [32, 32, 16, 8, 8, 8]
cfg_fc_cifar = [1024, 512, num_classes]


# define approximate firing function
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()  # gt：great than，大于

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()


act_fun = ActFun.apply


# 膜电位更新，核心函数
def mem_update(ops, x, mem, spike):
    # 第一项：时序相关，自身状态累积（包括decay衰减和不应期的清空）；第二项：输入相关，与人工神经网络计算方式相同
    # print(mem.size(),spike.size(),x.size())
    outx = ops(x)
    last_state = 1. - spike
    mem = mem * decay * last_state + outx
    # 激活函数，理解为直接与阈值比较大小
    spike = act_fun(mem)  # act_fun : approximation firing function
    return mem, spike


# 卷积层(in_planes, out_planes, stride, padding, kernel_size)
if dataset_name == 'cifar10':
    cfg_cnn = cfg_cnn_cifar
    cfg_kernel = cfg_kernel_cifar
    cfg_fc = cfg_fc_cifar
else:
    cfg_cnn = cfg_cnn_mnist
    cfg_kernel = cfg_kernel_mnist
    cfg_fc = cfg_fc_mnist


# 学习率衰减，每lr_decay_epoch轮学习率衰减一次
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer


# SNN网络定义
class SCNN(nn.Module):
    def __init__(self, cuda_id=0):
        super(SCNN, self).__init__()
        # 卷积层参数，与人工神经网络的卷积层相同
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[2]
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[3]
        self.conv4 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[4]
        self.conv5 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        ###
        if sub_layer:
            in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[5]
            self.conv6 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        ###

        # 全连接层，与人工神经网络的卷积层相同
        self.fc1 = nn.Linear(cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1], cfg_fc[0])  # a[-1]表示倒数第一个元素
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.fc3 = nn.Linear(cfg_fc[1], cfg_fc[2])

        self.cuda_id = cuda_id

    def forward(self, input, T=1):
        # 初始化
        # c：卷积层，h：全连接层，mem：膜电位（实数，神经元状态，用于判定放电），spike：脉冲状态（0,1，即神经元输出）
        # cfg_cnn = [(3, 32, 1, 1, 3),(32, 32, 1, 1, 3),]
        # cfg_kernel = [32, 16, 8]
        # cfg_fc = [128, 10]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu",self.cuda_id)
        batch_size = input.size(0)
        time_window = time_steps

        c1_mem = c1_spike = torch.zeros(
            batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device)
        c2_mem = c2_spike = torch.zeros(
            batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device)
        c3_mem = c3_spike = torch.zeros(
            batch_size, cfg_cnn[2][1], cfg_kernel[2], cfg_kernel[2], device=device)
        c4_mem = c4_spike = torch.zeros(
            batch_size, cfg_cnn[3][1], cfg_kernel[3], cfg_kernel[3], device=device)
        c5_mem = c5_spike = torch.zeros(
            batch_size, cfg_cnn[4][1], cfg_kernel[4], cfg_kernel[4], device=device)

        if sub_layer:
            ###
            c6_mem = c6_spike = torch.zeros(
                batch_size, cfg_cnn[5][1], cfg_kernel[5], cfg_kernel[5], device=device)
            ###

        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)
        h3_mem = h3_spike = h3_sumspike = torch.zeros(batch_size, cfg_fc[2], device=device)
        output_spike = torch.zeros(time_window, batch_size, cfg_fc[2])

        # print('===进入时间窗===')
        for step in range(time_window):  # 时间轴模拟
            # 与随机矩阵比较，得到初始输入的脉冲状态
            x = input > torch.rand(input.size(), device=device)  # 放电

            # 更新参数，扩展的卷积操作（mem_update是核心函数，作用是在脉冲神经网络视角下更新神经元的状态以确定输出）
            c1_mem, c1_spike = mem_update(self.conv1, x.float(), c1_mem, c1_spike)
            x = c1_spike

            c2_mem, c2_spike = mem_update(self.conv2, x, c2_mem, c2_spike)
            # 平均池化（对上层输出即脉冲）
            x = F.avg_pool2d(c2_spike, 2)

            c3_mem, c3_spike = mem_update(self.conv3, x, c3_mem, c3_spike)
            x = F.avg_pool2d(c3_spike, 2)

            c4_mem, c4_spike = mem_update(self.conv4, x, c4_mem, c4_spike)
            x = c4_spike

            c5_mem, c5_spike = mem_update(self.conv5, x, c5_mem, c5_spike)
            x = c5_spike

            if sub_layer:
                ###
                c6_mem, c6_spike = mem_update(self.conv6, x, c6_mem, c6_spike)
                x = c6_spike
                ###

            # print('卷积层输出：')
            # print(x[0][256])
            # 拉长输出矩阵用于全连接层输入
            x = x.view(batch_size, -1)

            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike)
            h1_sumspike += h1_spike
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike)
            # 用于计算时间段内的平均值作为最终输出
            h2_sumspike += h2_spike
            h3_mem, h3_spike = mem_update(self.fc3, h2_spike, h3_mem, h3_spike)
            h3_sumspike += h3_spike
            output_spike[step] = h3_spike

        outputs = h3_sumspike / time_window

        if frame == 'fedDis':
            return outputs, output_spike.transpose(0, 1)
        else:
            return outputs

    def feature_forward(self, input):
        # 初始化
        # c：卷积层，h：全连接层，mem：膜电位（实数，神经元状态，用于判定放电），spike：脉冲状态（0,1，即神经元输出）
        # cfg_cnn = [(3, 32, 1, 1, 3),(32, 32, 1, 1, 3),]
        # cfg_kernel = [32, 16, 8]
        # cfg_fc = [128, 10]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu",self.cuda_id)
        batch_size = input.size(0)
        time_window = time_steps

        c1_mem = c1_spike = torch.zeros(
            batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device)
        c2_mem = c2_spike = torch.zeros(
            batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device)
        c3_mem = c3_spike = torch.zeros(
            batch_size, cfg_cnn[2][1], cfg_kernel[2], cfg_kernel[2], device=device)
        c4_mem = c4_spike = torch.zeros(
            batch_size, cfg_cnn[3][1], cfg_kernel[3], cfg_kernel[3], device=device)
        c5_mem = c5_spike = torch.zeros(
            batch_size, cfg_cnn[4][1], cfg_kernel[4], cfg_kernel[4], device=device)
        ###
        if sub_layer:
            c6_mem = c6_spike = torch.zeros(
                batch_size, cfg_cnn[5][1], cfg_kernel[5], cfg_kernel[5], device=device)
            c_sumspike = torch.zeros(
                batch_size, cfg_cnn[5][1], cfg_kernel[5], cfg_kernel[5], device=device)
            output_spike = torch.zeros(time_window,
                                       batch_size, cfg_cnn[5][1], cfg_kernel[5], cfg_kernel[5], device=device)
        else:
            c_sumspike = torch.zeros(
                batch_size, cfg_cnn[4][1], cfg_kernel[4], cfg_kernel[4], device=device)
            output_spike = torch.zeros(time_window,
                                       batch_size, cfg_cnn[4][1], cfg_kernel[4], cfg_kernel[4], device=device)
        ###

        # print('===进入时间窗===')
        for step in range(time_window):  # 时间轴模拟
            # 与随机矩阵比较，得到初始输入的脉冲状态
            x = input > torch.rand(input.size(), device=device)  # 放电

            # 更新参数，扩展的卷积操作（mem_update是核心函数，作用是在脉冲神经网络视角下更新神经元的状态以确定输出）
            c1_mem, c1_spike = mem_update(self.conv1, x.float(), c1_mem, c1_spike)
            x = c1_spike

            c2_mem, c2_spike = mem_update(self.conv2, x, c2_mem, c2_spike)
            # 平均池化（对上层输出即脉冲）
            x = F.avg_pool2d(c2_spike, 2)

            c3_mem, c3_spike = mem_update(self.conv3, x, c3_mem, c3_spike)
            x = F.avg_pool2d(c3_spike, 2)

            c4_mem, c4_spike = mem_update(self.conv4, x, c4_mem, c4_spike)
            x = c4_spike

            c5_mem, c5_spike = mem_update(self.conv5, x, c5_mem, c5_spike)
            x = c5_spike

            if sub_layer:
                ###
                c6_mem, c6_spike = mem_update(self.conv6, x, c6_mem, c6_spike)
                x = c6_spike
                ###

            output_spike[step] = x
            c_sumspike += x

        outputs = c_sumspike / time_window
        output_spike = torch.flatten(output_spike, 2)  # x维度：(T, b, 32*)
        output_spike = output_spike.transpose(0, 1)
        return outputs, output_spike

    def forward_merge(self, input, T=1):
        # 初始化
        # c：卷积层，h：全连接层，mem：膜电位（实数，神经元状态，用于判定放电），spike：脉冲状态（0,1，即神经元输出）
        # cfg_cnn = [(3, 32, 1, 1, 3),(32, 32, 1, 1, 3),]
        # cfg_kernel = [32, 16, 8]
        # cfg_fc = [128, 10]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu",self.cuda_id)
        batch_size = input.size(0)
        time_window = time_steps

        c1_mem = c1_spike = torch.zeros(
            batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device)
        c2_mem = c2_spike = torch.zeros(
            batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device)
        c3_mem = c3_spike = torch.zeros(
            batch_size, cfg_cnn[2][1], cfg_kernel[2], cfg_kernel[2], device=device)
        c4_mem = c4_spike = torch.zeros(
            batch_size, cfg_cnn[3][1], cfg_kernel[3], cfg_kernel[3], device=device)
        c5_mem = c5_spike = torch.zeros(
            batch_size, cfg_cnn[4][1], cfg_kernel[4], cfg_kernel[4], device=device)

        if sub_layer:
            ###
            c6_mem = c6_spike = torch.zeros(
                batch_size, cfg_cnn[5][1], cfg_kernel[5], cfg_kernel[5], device=device)
            output_spike_hint = torch.zeros(time_window,
                                       batch_size, cfg_cnn[5][1], cfg_kernel[5], cfg_kernel[5], device=device)
            ###
        else:
            output_spike_hint = torch.zeros(time_window,
                                       batch_size, cfg_cnn[4][1], cfg_kernel[4], cfg_kernel[4], device=device)

        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)
        h3_mem = h3_spike = h3_sumspike = torch.zeros(batch_size, cfg_fc[2], device=device)
        output_spike_all = torch.zeros(time_window, batch_size, cfg_fc[2])

        # print('===进入时间窗===')
        for step in range(time_window):  # 时间轴模拟
            # 与随机矩阵比较，得到初始输入的脉冲状态
            x = input > torch.rand(input.size(), device=device)  # 放电

            # 更新参数，扩展的卷积操作（mem_update是核心函数，作用是在脉冲神经网络视角下更新神经元的状态以确定输出）
            c1_mem, c1_spike = mem_update(self.conv1, x.float(), c1_mem, c1_spike)
            x = c1_spike

            c2_mem, c2_spike = mem_update(self.conv2, x, c2_mem, c2_spike)
            # 平均池化（对上层输出即脉冲）
            x = F.avg_pool2d(c2_spike, 2)

            c3_mem, c3_spike = mem_update(self.conv3, x, c3_mem, c3_spike)
            x = F.avg_pool2d(c3_spike, 2)

            c4_mem, c4_spike = mem_update(self.conv4, x, c4_mem, c4_spike)
            x = c4_spike

            c5_mem, c5_spike = mem_update(self.conv5, x, c5_mem, c5_spike)
            x = c5_spike

            if sub_layer:
                ###
                c6_mem, c6_spike = mem_update(self.conv6, x, c6_mem, c6_spike)
                x = c6_spike
                ###
            output_spike_hint[step] = x

            # print('卷积层输出：')
            # print(x[0][256])
            # 拉长输出矩阵用于全连接层输入
            x = x.view(batch_size, -1)

            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike)
            h1_sumspike += h1_spike
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike)
            # 用于计算时间段内的平均值作为最终输出
            h2_sumspike += h2_spike
            h3_mem, h3_spike = mem_update(self.fc3, h2_spike, h3_mem, h3_spike)
            h3_sumspike += h3_spike
            output_spike_all[step] = h3_spike

        output_spike_hint = torch.flatten(output_spike_hint, 2)  # x维度：(T, b, 32*)

        return output_spike_all.transpose(0, 1), output_spike_hint.transpose(0, 1)

