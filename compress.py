import torch
from tqdm import tqdm
from global_configs import *


def spike_compress(spike:torch.Tensor):
    comp = 0
    for i in range(len(spike)):
        comp = comp * 2 + spike[i]
    return int(comp)


def spike_depress(comp:int, step):
    spike = torch.zeros(step, dtype=torch.float)
    for i in range(step-1, -1, -1):
        spike[i] = comp % 2
        comp = comp // 2
    return spike


def spike_tensor_compress(spikes:torch.Tensor, dim_time_window=2): #维度：(batch数, T, batch size, classes/**) (batch数, batch size, T, classes/**)
    if net_name == 'ANN':
        return spikes
    if not compression:
        return spikes
    device_old = spikes.device
    '''spikes_comp = torch.zeros([spikes.size(0), spikes.size(1) ,spikes.size(3), 1], dtype=torch.int16, device='cuda')
    for it in range(spikes.size(0)):
        for b in range(spikes.size(1)):
            for c in range(spikes.size(3)):
                for t in range(spikes.size(2)):
                    spikes_comp[it][b][c] = spikes_comp[it][b][c] * 2 + spikes[it][b][t][c]
    spikes_comp.to(device_old)
    spikes.to(device_old)'''

    spike_tensor = spikes.transpose(0, dim_time_window).cuda() #(T, batch size, batch数, classes/**)
    time_window = spike_tensor.size(0)
    spikes_comp = torch.zeros_like(spike_tensor[0], device='cuda')
    for i in tqdm(range(time_window)):
        spikes_comp += spike_tensor[i]<<(time_window - i)

    spikes_comp = spikes_comp.clone().type(torch.int16)

    spikes_comp = spikes_comp.to(device_old)
    del spike_tensor

    return spikes_comp #(batch数, batch size, classes/, 1)


def spike_tensor_depress(spikes_comp:torch.Tensor, dim_time_window=2, time_window=time_steps): #(batch数, batch size, classes/, 1)
    if net_name == 'ANN':
        return spikes_comp
    if not compression:
        return spikes_comp
    device_old = spikes_comp.device

    '''spikes = torch.zeros([spikes_comp.size(0), spikes_comp.size(1), time_step, spikes_comp.size(2)],
                         dtype=torch.float, device='cuda')  #维度：(batch数, batch size, T, classes/**)
    for it in range(spikes.size(0)):
        for b in range(spikes.size(1)):
            for c in range(spikes.size(3)):
                spike = spikes_comp[it][b][c]
                for t in range(time_step-1, -1, -1):
                    spikes[it][b][t][c] = spike % 2
                    spike = spike // 2
    spikes = spikes.to(device_old)'''

    spikes_comp = spikes_comp.clone().type(torch.int16).cuda()

    size_spike_tensor = [time_window]
    size_spike_tensor.extend(spikes_comp.size()) #(T, batch size, batch数, classes/**)
    spike_tensor = torch.zeros(size_spike_tensor, dtype=torch.float, device='cuda')
    for i in tqdm(range(time_window)):
        spike_tensor[i] = (spikes_comp>>(time_window - i)) % 2
    spikes = spike_tensor.transpose(0, dim_time_window)

    spikes = spikes.to(device_old)
    del spike_tensor

    return spikes
