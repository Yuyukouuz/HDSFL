import torch.optim

from tools import *

max_devices = 10

class DeviceControl: #设备控制块，用于服务器记录各个连接设备的状态
    def __init__(self, device_id, sock:socket.socket, addr):
        self.device_id = device_id
        self.sock = sock
        self.addr = addr
        self.net_state = None
        self.spikes = None
        self.spikes_hint = None
        self.acc = 0.0
        self.datasets_size:int = 1
        self.net = SCNN()

    def set_information(self, device_information):
        self.datasets_size = device_information['datasets_size']

    def update_net_state(self, net_state): #网络参数值，或输出的张量
        self.net_state = net_state

    def update_spikes(self, spike=None, spike_hint=None):
        if spike is not None:
            self.spikes = spike
        if spike_hint is not None:
            self.spikes_hint = spike_hint

    def update_acc(self, acc_new):
        self.acc = acc_new

    def init_net(self):
        self.net = SCNN()

    #def distillation_net(self, epochs=4, lr=learning_rate_s):
    #    train(net=self.net, epochs=epochs, dataset=dataset_share, bs=batch_share,
    #          spikes=self.net_state, lr=lr)

    def test(self, dataset, use):
        quick_test(self.net, dataset=dataset, use=use, bs=batch_size, tips='设备' + str(self.device_id) + '测试准确率：')

    def get_net_state(self):
        return self.net_state

    def get_spikes(self):
        return self.spikes

    def get_spikes_hint(self):
        return self.spikes_hint

    def get_sets_size(self):
        return self.datasets_size

    def get_acc_r(self):
        return self.acc


class ServerControl(threading.Thread): #通信控制模块，实现服务器通信进程
    def __init__(self, controler:DeviceControl, action:str):
        threading.Thread.__init__(self)
        self.controler = controler
        self.action = action

    def run(self):
        sock = self.controler.sock
        global federal_net_state
        global pcount
        global global_epoch
        if simulation:
            global single_upstream_size
            global single_downstream_size
            global upstream_size
            global downstream_size
        if self.action == 'init':
            sock.send('\1'.encode())#控制码，初始化协商
            #sock.recv(16)
            information = receive_and_load(sock)
            #sock.send('\0'.encode())
            self.controler.set_information(information)
            dataset_inds = distributer.get_train_inds(information['datasets_size'])
            #dump_and_send(sock, dataset_inds)
            send_list(sock, dataset_inds)
            sock.recv(1)
            #dump_and_send(sock, distributer.get_share_inds())
            send_list(sock, distributer.get_share_inds())
            sock.recv(1)
            if frame == 'fedAvg':
                #下发网络参数，这里应该用读者-写者同步，即加入计数，允许同时读、不允许同时写或读写。
                count_lock.acquire()
                if pcount == 0:
                    fns_lock.acquire()
                pcount += 1
                count_lock.release()
                dump_and_send_pickle(sock, federal_net_state)
                count_lock.acquire()
                pcount -= 1
                if pcount == 0:
                    fns_lock.release()
                count_lock.release()
                sock.recv(1)
            elif frame == 'fedDis' or frame == 'fedAd1':
                #下发当前蒸馏状态
                if global_epoch == 0:
                    dump_and_send(sock, 'first')
                else:
                    dump_and_send(sock, 'wait')
                    sock.recv(1)
                    count_lock.acquire()
                    if pcount == 0:
                        fns_lock.acquire()
                    pcount += 1
                    count_lock.release()
                    spikes_compress = spike_tensor_compress(federal_net_state)
                    spikes_compress_list = spikes_compress.tolist()
                    #dump_and_send(sock, spikes_compress_list)
                    send_list(sock, spikes_compress_list)
                    count_lock.acquire()
                    pcount -= 1
                    if pcount == 0:
                        fns_lock.release()
                    count_lock.release()
                sock.recv(1)

        elif self.action == 'train':
            sock.send('\2'.encode())#控制码，训练
            sock.recv(1)
            if frame == 'fedAvg':
                parameters = trans_parameter
                dump_and_send(sock, parameters)
                net_state = receive_and_load_pickle(sock)
                #sock.send('\0'.encode())
                self.controler.update_net_state(net_state)
            elif frame == 'fedDis' or frame == 'fedAd1':
                parameters = trans_parameter
                parameters['temperature'] = temperature
                if global_epoch == 0:
                    parameters['state_of_distillation'] = 'first'
                else:
                    parameters['state_of_distillation'] = 'wait'
                dump_and_send(sock, parameters)

                #spikes_compress_list = receive_and_load(sock)
                spikes_compress_list = receive_list(sock)
                sock.send('\0'.encode())

                ###
                spikes_hint_compress_list = None
                if hint_layer:
                    #spikes_hint_compress_list = receive_and_load(sock)
                    spikes_hint_compress_list = receive_list(sock)
                    sock.send('\0'.encode())
                ###

                acc_r = receive_and_load(sock)

                if acc_r is not None:
                    self.controler.update_acc(acc_r)

                if spikes_compress_list is not None:
                    spikes_compress = torch.tensor(spikes_compress_list)
                    spikes = spike_tensor_depress(spikes_compress)

                ###改成解压
                if hint_layer and spikes_hint_compress_list is not None:
                    spikes_compress_hint = torch.tensor(spikes_hint_compress_list)
                    spikes_hint = spike_tensor_depress(spikes_compress_hint)
                    #spikes_hint = spikes_hint
                ###

                if hint_layer:
                    self.controler.update_spikes(spike=spikes, spike_hint=spikes_hint)
                self.controler.update_net_state(net_state=spikes)

        elif self.action == 'distribute':
            sock.send('\3'.encode())#控制码，下发新的网络参数
            sock.recv(1)
            count_lock.acquire()
            if pcount == 0:
                fns_lock.acquire()
            pcount += 1
            count_lock.release()
            if frame == 'fedDis' or frame == 'fedAd1':
                spikes_compress = spike_tensor_compress(federal_net_state)
                spikes_compress_list = spikes_compress.tolist()
                #dump_and_send(sock, spikes_compress_list)
                send_list(sock, spikes_compress_list)
            elif frame == 'fedAvg':
                dump_and_send_pickle(sock, federal_net_state)
                #federal_net_state_list = federal_net_state.tolist()
                #dump_and_send(sock, federal_net_state_list)

            count_lock.acquire()
            pcount -= 1
            if pcount == 0:
                fns_lock.release()
            count_lock.release()
            sock.recv(1)

        elif self.action == 'finish':
            sock.send('\4'.encode())  # 控制码，下发新的网络参数
            sock.recv(1)
            sock.send('\0'.encode())

trans_parameter = {'epochs':2, 'epochs_distillation':8, 'samples_size':160, 'num_classes':20 if dataset_name == 'cifar100' else 10,
                   'learning_rate':lr_train, 'batch_size':64, 'state_of_distillation':'wait',}


class ServerListener(threading.Thread): #连接监听器，监听申请接入服务器的设备
    def __init__(self, ss:socket.socket):
        threading.Thread.__init__(self)
        self.ss = ss

    def run(self):
        global connected_devices
        global devices_list
        self.ss.listen(max_devices)
        while connected_devices <= max_devices:
            sock, addr = self.ss.accept()
            print('客户端',connected_devices,'连接成功')
            controler = DeviceControl(connected_devices, sock, addr) #客户端设备控制块，记录每个客户端的状态
            #thread_device = server_device(controler, sock, addr)
            thread_control = ServerControl(controler=controler, action='init')  #通信进程：初始化
            thread_control.start()
            thread_control.join()
            devices_list.append(controler)
            cd_lock.acquire()
            connected_devices += 1
            if connected_devices >= min_device_number:
                event_first_device.set()
            cd_lock.release()


#globle
connected_devices = 0
cd_lock = threading.Lock() #设备数的互斥锁
devices_list = list([]) #设备控制块列表
event_first_device = threading.Event()
upstream_size = 0
downstream_size = 0
single_upstream_size = 0
single_downstream_size = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_net = SCNN()
federal_net_state = None
federal_spike_hint = None
fns_lock = threading.Lock() #网络参数互斥锁
pcount = 0
count_lock = threading.Lock() #读者写者同步计数互斥
batch_size = 50
#test_loader = DataLoader(emnist_test, batch_size=batch_size, shuffle=True, num_workers=2)
max_acc = 0
epoch_max_acc = 0
global_epoch = 0
temperature = 100

distributer = Distributer()


def server_start():  #服务端启动接口
    host = socket.gethostname()
    port = tcp_port
    s = socket.socket()
    s.bind((host, port))
    print('建立监听线程')
    thread_listen = ServerListener(s)
    thread_listen.start()

    print('建立完成')

    global connected_devices
    global devices_list
    global test_net
    global federal_net_state
    global trans_parameter
    global single_upstream_size
    global single_downstream_size

    global distributer

    dataset_share = get_share_set_byinds(distributer.get_share_inds())
    save_net = SCNN() #用于保存的网络

    if frame == 'fedAvg':
        federal_net_state = test_net.state_dict()
    print('等待客户端连接')
    event_first_device.wait()  #等待首个客户端进入训练
    global global_epoch

    start_time = time.strftime('%d-%H',time.localtime(time.time()))
    recorder = []

    if use_confusion_matrix:
        saver = ConfusionMatrixSave(f'server_{start_time}')
    while global_epoch <= total_epochs: #开始训练
        cd_lock.acquire()
        current_devices = connected_devices
        cd_lock.release()
        threads = list([])

        if global_epoch == total_epochs:
            for i in range(current_devices):
                thread_control = ServerControl(controler=devices_list[i], action='finish')  # 通信进程：训练结束
                thread_control.start()
                threads.append(thread_control)
            for i in range(current_devices):
                threads[i].join()

            break

        print('客户端训练并上传网络参数...')
        for i in range(current_devices):
            thread_control = ServerControl(controler=devices_list[i], action='train')#训练上传
            thread_control.start()
            threads.append(thread_control)
        for i in range(current_devices):
            threads[i].join()
            print('客户端',i,'训练完成')
        del threads

        #联合训练
        print('联合训练...')
        if frame == 'fedDis':
            ###
            if hint_layer:
                if simulation:
                    print('--模拟模式：含提示层的聚合--')
                    print('--模拟模式：含提示层的蒸馏训练--')
                    loss_list = []
                    fl_cortrain_hint_distillation(current_devices, skip=True)
                else:
                    fl_cortrain_hint_distillation(current_devices)
                    if loss_f_distillation == 'cka':
                        loss_hint = linear_CKA
                    else:
                        loss_hint = torch.nn.MSELoss()
                    loss_list = train_distillation(net=test_net, spikes_hint=federal_spike_hint, spikes_all=federal_net_state,
                                                   lossf_hint=loss_hint,
                                                   epochs=8, dataset=dataset_share, bs=batch_share, lr=learning_rate_s) #提示层蒸馏
                    '''loss_list = train(net=test_net, epochs=8, dataset=dataset_share, bs=batch_share,
                          spikes=federal_spike_hint, lr=learning_rate_s, layer='hint', lossf=loss_hint)'''
                    print('--完成提示层蒸馏--')
                    quick_test(test_net, dataset=dataset_test, use=2000, bs=20)
                #额外蒸馏
                '''train(net=test_net, epochs=8, dataset=dataset_share, bs=batch_share,
                      spikes=federal_net_state, lr=learning_rate_s)
                # loss_list = distillation(4)
                quick_test(test_net, dataset=dataset_test, use=2000, bs=20)'''
            ###
            else:
                if simulation:
                    print('--模拟模式：联邦聚合--')
                    print('--模拟模式：蒸馏训练--')
                    loss_list = []
                    fl_cortrain_distillation(current_devices, skip=True)
                else:
                    fl_cortrain_distillation(current_devices)
                    loss_list = train(net=test_net, epochs=8, dataset=dataset_share, bs=batch_share,
                                      spikes=federal_net_state, lr=learning_rate_s)  #蒸馏
            federal_net_state = torch.round(federal_net_state)
            #compute_spikes(batch_share) #二次蒸馏，或直接比较
            # federal_net_state > 0.5
        elif frame == 'fedAvg':
            fl_cortrain(current_devices)
        elif frame == 'fedAd1':
            fl_dcortrain_2(current_devices)

        threads = list([])
        print('分发新网络参数...')
        for i in range(current_devices):
            thread_control = ServerControl(controler=devices_list[i], action='distribute')#回发
            thread_control.start()
            threads.append(thread_control)
        if use_confusion_matrix:
            cur_a, max_a, max_epoch, confusion_matrix = test(s=-1, epoch=global_epoch, conf=True, classes=10)
            saver.add_matrix(confusion_matrix)
            saver.save_matrix()
        else:
            if simulation:
                print('--模拟模式：测试准确率，当前轮数：%d--' %(global_epoch,))
                cur_a = max_a = max_epoch = 0
            else:
                cur_a, max_a, max_epoch = test(epoch=global_epoch)
        if cur_a == max_a:
            save_net.load_state_dict(test_net.state_dict())
        for i in range(current_devices):
            threads[i].join()

        print('当前下行通信总量：', downstream_size, '上行通信总量：', upstream_size)
        if simulation:
            print('--模拟模式：保存通信字节数--')
            recorder.append({'upstream': upstream_size, 'downstream': downstream_size,
                             'single_upstream': single_upstream_size, 'single_downstream': single_downstream_size})
            saves(best_acc=max_a, best_epoch=max_epoch, recorder=recorder, names=start_time, pre_name='simulation_')
        else:
            if frame == 'fedDis':
                recorder.append({'acc': cur_a, 'upstream': upstream_size, 'downstream': downstream_size, 'loss': loss_list})
            elif frame == 'fedAvg' or frame == 'fedAd1':
                recorder.append({'acc': cur_a, 'upstream': upstream_size, 'downstream': downstream_size, 'loss': []})
            if do_save:
                if global_epoch % 5 == 0 or (global_epoch > 10 and global_epoch % 2 == 0):
                    saves(best_acc=max_a, best_epoch=max_epoch, recorder=recorder, names=start_time, model=save_net, pre_name='s')

        single_upstream_size = single_downstream_size = 0
        global_epoch += 1

        #if global_epoch % 16 == 0:
        #    #trans_parameter['learning_rate'] = trans_parameter['learning_rate'] / 2 #学习率调整
        #    trans_parameter['epochs_distillation'] += 2
        #    trans_parameter['epochs'] = max(trans_parameter['epochs'] - 2, 1)

        del threads

    return


def fl_cortrain(used_devices): #联邦平均的聚合主程序
    print('---开始联合训练---')
    #sum_sets, dvs_sets = get_data_cfg()
    global devices_list
    global federal_net_state
    if used_devices == 1:
        print('只有单个设备，跳过联邦平均')
        federal_net_state = devices_list[0].get_net_state()
        return federal_net_state

    n_devices = used_devices
    state_dict = np.empty(n_devices, OrderedDict)

    sum_sets = 0 #这个总和只能提前获取，否则容易出错（被多次叠加）
    for i in range(n_devices):
        state_dict[i] = devices_list[i].get_net_state()
        sum_sets += devices_list[i].get_sets_size()

    state_dict_avg = OrderedDict()
    # key：网络层；i：设备号
    for key in federal_net_state.keys(): #修改之后应改为本地的网络参数为准，即改为federal_net_state
        state_dict_avg[key] = torch.zeros(federal_net_state[key].size())
        for i in range(n_devices):
            sets_size = devices_list[i].get_sets_size()
            #print(state_dict_avg[key].device, state_dict[i][key].device) #cpu cuda:0  #cpu cpu
            #input()
            state_dict_avg[key] += (state_dict[i][key].cpu() * sets_size)
        state_dict_avg[key] = state_dict_avg[key] / sum_sets

    for i in range(n_devices):
        devices_list[i].update_net_state(state_dict_avg)

    fns_lock.acquire()
    federal_net_state = state_dict_avg
    fns_lock.release()

    return state_dict_avg


def fl_cortrain_distillation(used_devices, t=1, skip=False): #经典联邦蒸馏聚合
    print('---开始联合训练：联邦蒸馏---')
    #sum_sets, dvs_sets = get_data_cfg()
    global devices_list
    global federal_net_state
    if used_devices == 1 or skip:
        print('只有单个设备，跳过联邦平均')
        federal_net_state = devices_list[0].get_net_state()
        return federal_net_state

    n_devices = used_devices
    spike_list = list([])
    acc_list = list([])
    for i in range(n_devices):
        spike_list.append(devices_list[i].get_net_state())
        acc_list.append(devices_list[i].get_acc_r())

    acc_softmax = torch.tensor(acc_list)
    print(acc_softmax)
    acc_softmax = F.softmax(acc_softmax * 1., dim=0)
    #sum_spike = spike_list[0]
    state_dict_avg = spike_list[0] * acc_softmax[0]

    for i in range(1, n_devices):
        state_dict_avg += spike_list[i] * acc_softmax[i]
    #    sum_spike += spike_list[i]

    #state_dict_avg = sum_spike / n_devices #算术平均，后续考虑改为加权平均

    #for i in range(n_devices):
    #    devices_list[i].update_net_state(state_dict_avg)

    fns_lock.acquire()
    federal_net_state = state_dict_avg
    fns_lock.release()

    return state_dict_avg


def fl_cortrain_hint_distillation(used_devices, t=1, skip=False): #经典联邦蒸馏聚合
    print('---开始联合训练：联邦蒸馏---')
    #sum_sets, dvs_sets = get_data_cfg()
    global devices_list
    global federal_net_state
    global federal_spike_hint
    if used_devices == 1 or skip:
        print('跳过联邦平均')
        federal_net_state = devices_list[0].get_spikes()
        federal_spike_hint = devices_list[0].get_spikes_hint()
        return federal_net_state, federal_spike_hint

    n_devices = used_devices
    spike_list = list([])
    spike_hint_list = list([])
    acc_list = list([])
    for i in range(n_devices):
        spike_list.append(devices_list[i].get_spikes())
        spike_hint_list.append(devices_list[i].get_spikes_hint())
        acc_list.append(devices_list[i].get_acc_r())

    acc_softmax = torch.tensor(acc_list)
    print(acc_softmax)
    acc_softmax = F.softmax(acc_softmax * 1., dim=0)
    #sum_spike = spike_list[0]
    state_dict_avg = spike_list[0] * acc_softmax[0]
    state_dict_hint_avg = spike_hint_list[0] * acc_softmax[0]

    for i in range(1, n_devices):
        state_dict_avg += spike_list[i] * acc_softmax[i]
        state_dict_hint_avg += spike_hint_list[i] * acc_softmax[i]
    #    sum_spike += spike_list[i]

    #state_dict_avg = sum_spike / n_devices #算术平均，后续考虑改为加权平均

    #for i in range(n_devices):
    #    devices_list[i].update_net_state(state_dict_avg)

    fns_lock.acquire()
    federal_net_state = state_dict_avg
    federal_spike_hint = state_dict_hint_avg
    fns_lock.release()

    return federal_net_state, federal_spike_hint


def fl_dcortrain_2(used_devices, t=1): #改进联邦蒸馏聚合
    print('---开始联合训练：蒸馏平均---')
    #sum_sets, dvs_sets = get_data_cfg()
    global devices_list
    global federal_net_state
    global test_net
    if federal_net_state is None:
        print('初始化')
        federal_net_state = devices_list[0].get_net_state()

    n_devices = used_devices
    net_state_list = list([])
    acc_list = list([])

    for i in range(n_devices): #网络初始化
        devices_list[i].init_net()

    for i in range(n_devices): #每个设备网络分别蒸馏，然后对网络参数进行平均
        devices_list[i].distillation_net(epochs=4, lr=learning_rate_s)
        net_state_list.append(devices_list[i].net.state_dict())
        acc_list.append(devices_list[i].get_acc_r())

    for i in range(n_devices): #分别测试准确率
        devices_list[i].test(dataset=dataset_test, use=2000)

    acc_softmax = torch.tensor(acc_list)
    acc_softmax = F.softmax(acc_softmax, dim=0)
    #sum_spike = spike_list[0]
    state_dict_avg = OrderedDict()
    # key：网络层；i：设备号
    for key in net_state_list[0].keys():  # 修改之后应改为本地的网络参数为准，即改为federal_net_state
        state_dict_avg[key] = torch.zeros(net_state_list[0][key].size())
        for i in range(n_devices):
            state_dict_avg[key] += (net_state_list[i][key].cpu() * acc_softmax[i])

    test_net.load_state_dict(state_dict_avg)

    fns_lock.acquire()
    compute_spikes(batch_share)
    fns_lock.release()

    return state_dict_avg


'''def distillation(epochs_distillation=1):
    global test_net
    global federal_net_state
    spikes = federal_net_state
    snn_net = test_net
    #optimizer = torch.optim.SGD(snn_net.parameters(), lr=learning_rate_s, momentum=0.9)
    optimizer = torch.optim.Adam(snn_net.parameters(), lr=learning_rate_s, )
    running_loss = 0
    train_loader = DataLoader(dataset_share, batch_size=batch_share, pin_memory=False)
    loss_function = my_loss_ann if net_name == 'ANN' else my_loss
    #loss_function = torch.nn.CrossEntropyLoss()

    loss_list = []

    snn_net.to(device)
    snn_net.train()
    for e in range(epochs_distillation):
        for i, (images, labels) in enumerate(train_loader):
            snn_net.zero_grad()
            optimizer.zero_grad()
            # print('获取训练数据')
            images = images.float().to(device)
            # print('前向传播')
            _, outputs = snn_net(images, temperature)
            # print('标签处理')
            #labels_ = F.one_hot(labels, num_classes).float()
            # print('损失函数计算')
            spike = spikes[i]
            #_, label = spike.max(1)
            loss = loss_function(outputs.cpu(), spike)  # cpu? 这里使用自定义损失函数进行蒸馏
            loss_list.append(loss)
            running_loss += loss.item()
            # print('反向传播')
            loss.backward()
            # print('参数优化')
            optimizer.step()
            # 使用spikingjelly必须加这句
            functional.reset_net(snn_net)
            torch.cuda.empty_cache()
        print('--已完成第%d轮蒸馏，当前loss=%.4f--' % (e + 1, loss))
    del train_loader
    torch.cuda.empty_cache()
    test_net = snn_net
    # 1225
    print('--服务器完成二次蒸馏--')
    return loss_list'''


def test(s = 0, epoch = 0, conf=False, classes=10):
    correct = 0
    total = 0
    global test_net
    global federal_net_state
    global max_acc
    global epoch_max_acc
    global pcount

    if s > 0:
        test_loader = DataLoader(sub(dataset_test, s), batch_size=batch_size, shuffle=True, num_workers=2)
    else:
        test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=2)

    count_lock.acquire()
    if pcount == 0:
        fns_lock.acquire()
    pcount += 1
    count_lock.release()
    if frame == 'fedAvg':
        test_net.load_state_dict(federal_net_state)
    count_lock.acquire()
    pcount -= 1
    if pcount == 0:
        fns_lock.release()
    count_lock.release()

    if conf:
        confusion_matrix = torch.zeros([classes, classes], dtype=torch.int)

    test_net.to(device)
    functional.reset_net(test_net)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if dataset_name == 'cifar100':
                targets = fine_to_coarse(targets)
            inputs = inputs.to(device)
            # optimizer.zero_grad()
            if frame == 'fedAvg':
                outputs = test_net(inputs)
            elif frame == 'fedDis' or frame == 'fedAd1':
                outputs,_ = test_net(inputs)
            # labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
            # loss = criterion(outputs.cpu(), labels_)
            _, predicted = outputs.cpu().max(1)
            if conf:
                for i in range(len(predicted)):
                    confusion_matrix[targets[i]][predicted[i]] += 1
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
            functional.reset_net(test_net)
            if batch_idx % 100 == 0:
                acc = 100. * float(correct) / float(total)
                print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)
    test_net = test_net.cpu()
    torch.cuda.empty_cache()

    print('Iters:', epoch)
    acc = correct / total
    if acc > max_acc:
        max_acc = acc
        epoch_max_acc = epoch
    print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * acc))
    print('最高准确率：%.3f，所用轮数：%d' % (100 * max_acc, epoch_max_acc))
    if conf:
        return acc, max_acc, epoch_max_acc, confusion_matrix
    return acc, max_acc, epoch_max_acc


def send_list(sock: socket.socket, list_for_send: list, max_length = 100, buf_size: int = 1024):
    length = len(list_for_send)
    if max_length > 0:
        turns = length // max_length
        remain = length % max_length
    else:
        turns = 0
        remain = length
    sock.send(json.dumps(turns + 1 if remain != 0 else turns).encode())
    sock.recv(1)
    for i in range(turns):
        dump_and_send(sock, list_for_send[i * max_length: (i + 1) * max_length], buf_size)
    if remain != 0:
        dump_and_send(sock, list_for_send[turns * max_length:], buf_size)


def receive_list(sock: socket.socket, buf_size: int = 1024):
    turns = json.loads(sock.recv(1024))
    sock.send('\0'.encode())
    list_r = []
    for i in tqdm(range(turns)):
        l = receive_and_load(sock, buf_size)
        list_r.extend(l)
    return list_r


def dump_and_send(sock:socket.socket, any_for_send, buf_size:int=1024):
    global downstream_size
    global single_downstream_size
    while True:
        segment = json.dumps(any_for_send).encode()
        length = len(segment)
        downstream_size += length #下行通信量
        single_downstream_size += length
        sock.send(json.dumps(length).encode())
        sock.recv(1)
        sock.send(segment)
        back = sock.recv(1).decode()
        if back == '\0':
            break
        else:
            sock.send('\0'.encode())
            print('---重传---')

def receive_and_load(sock:socket.socket, buf_size:int=1024):
    global upstream_size
    global single_upstream_size

    retry = 0
    while True:
        length = json.loads(sock.recv(1024))
        turns = length // buf_size
        remain = length % buf_size

        sock.send('\0'.encode())
        segment_recv = b''
        segments = list([])
        for i in range(turns):
            rec = sock.recv(buf_size, socket.MSG_WAITALL) #(bufsize, flags)
            segment_recv += rec
            if len(rec) != buf_size:
                print('Error:', len(rec))
            if (i % 4096) == 0 and i != 0:
                segments.append(segment_recv)
                segment_recv = b''
        if remain > 0:
            segment_recv += sock.recv(remain, socket.MSG_WAITALL)
        segments.append(segment_recv)

        segment_recv = b''
        for seg in segments:
            segment_recv += seg
        if len(segment_recv) != length:
            print('--传输错误：缺少%d字节--' %(length - len(segment_recv)))
            return None
        single_upstream_size += len(segment_recv)
        upstream_size += len(segment_recv)  # 上行通信量

        obj = None
        try:
            obj = json.loads(segment_recv, strict=False)
        except json.decoder.JSONDecodeError:
            if retry >= 3:
                break
            sock.send('\9'.encode())

            back = sock.recv(1).decode()
            while back != '\8':
                back = sock.recv(1).decode()

            retry += 1
            print('---重传---')
        else:
            break

    sock.send('\0'.encode())
    return obj


def dump_and_send_pickle(sock:socket.socket, any_for_send, buf_size:int=1024):
    global downstream_size
    global single_downstream_size

    while True:
        segment = pickle.dumps(any_for_send)
        length = len(segment)
        downstream_size += length  # 下行通信量
        single_downstream_size += length
        single_downstream_size += length
        if length == 0:
            turns = 0
        else:
            turns = (length - 1)//buf_size + 1
        sock.send(pickle.dumps(length))
        sock.recv(1)
        sock.send(segment)
        back = sock.recv(1).decode()
        if back == '\0':
            break
        elif back == '\2':

            sock.send('\1'.encode())
            sock.recv(1)
            print('---重传---')


def receive_and_load_pickle(sock:socket.socket, buf_size:int=1024):
    global upstream_size
    global single_upstream_size
    retry = 0
    while True:
        length = pickle.loads(sock.recv(1024))
        turns = length // buf_size
        remain = length % buf_size

        sock.send('\0'.encode())
        segment_recv = b''
        segments = list([])
        for i in tqdm(range(int(turns))):
            segment_recv += sock.recv(buf_size, socket.MSG_WAITALL)
            if (i % 4096) == 0 and i != 0:
                segments.append(segment_recv)
                segment_recv = b''
        if remain > 0:
            segment_recv += sock.recv(remain, socket.MSG_WAITALL)
        segments.append(segment_recv)
        segment_recv = b''
        for seg in segments:
            segment_recv += seg
        upstream_size += len(segment_recv)  # 上行通信量
        single_upstream_size += len(segment_recv)

        obj = None
        try:
            obj = pickle.loads(segment_recv)
        except pickle.PickleError:
            if retry >= 3:
                break
            sock.send('\2'.encode())
            print('---加载不正确---')

            back = 'b'
            while back != '\1':
                back = sock.recv(1)
                try:
                    back = back.decode()
                    print('debug: back =',back)
                except UnicodeDecodeError:
                    back = 'b'

            sock.send('\0'.encode())
            retry += 1
        else:
            break

    sock.send('\0'.encode())
    return obj