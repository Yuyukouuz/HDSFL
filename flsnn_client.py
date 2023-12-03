import torch.nn

from tools import *

def client_start(data_size:int=n_clientDataset, buf_size:int=1024):
    sock = socket.socket()
    host = socket.gethostname()
    port = tcp_port
    sock.connect((host, port))
    global snn_net
    global buffer_size
    global datasets
    global spikes
    global datasets_size
    global train_sets
    global val_sets
    global acc_r

    datasets_size = data_size
    buffer_size = buf_size

    if use_confusion_matrix:
        start_time = time.strftime('%d-%H-%M-%S', time.localtime(time.time()))
        saver = ConfusionMatrixSave(f'client_{start_time}')

    init_state = snn_net.state_dict()
    while True:
        print('等待接受指令...')
        comment = sock.recv(1).decode()
        if comment == '\1':
            print('初始化协商...')
            #sock.send('\0'.encode())
            #information = trans_information()
            information = {'datasets_size':datasets_size, 'buf_size':buffer_size}
            #input()
            dump_and_send(sock=sock, any_for_send=information)
            #dataset_inds = receive_and_load(sock)
            dataset_inds = receive_list(sock)
            sock.send('\0'.encode())
            #share_inds = receive_and_load(sock)
            share_inds = receive_list(sock)
            sock.send('\0'.encode())
            datasets = get_train_set_byinds(dataset_inds)
            dataset_share = get_share_set_byinds(share_inds)
            #print('len=',len(datasets))

            if frame == 'fedDis' or frame == 'fedAd1':
                #train_sets = Subset(datasets, list(range(datasets_size))[2000:])
                train_sets = datasets
                val_sets = Subset(datasets, list(range(datasets_size))[:2000])
            elif frame == 'fedAvg':
                train_sets = datasets

            if frame == 'fedAvg':
            # 接收网络参数
                net_state = receive_and_load_pickle(sock)
                sock.send('\0'.encode())
                snn_net.load_state_dict(net_state)
            elif frame == 'fedDis' or frame == 'fedAd1':
                state_of_distillation = receive_and_load(sock)
                if state_of_distillation == 'wait':
                    sock.send('\0'.encode())

                    #spikes_compress_list = receive_and_load(sock)
                    spikes_compress_list = receive_list(sock)

                    if spikes_compress_list is not None:
                        spikes_compress = torch.tensor(spikes_compress_list)
                        spikes = spike_tensor_depress(spikes_compress)

                sock.send('\0'.encode())
            print('初始化完成')

        elif comment == '\2':
            print('本地训练...')
            sock.send('\0'.encode())
            parameters = receive_and_load(sock)

            if parameters is not None:
                epochs = parameters['epochs']
                epochs_distillation = parameters['epochs_distillation']
                load_parameters(parameters)

            if (frame == 'fedDis' or frame == 'fedAd1') and parameters['state_of_distillation'] == 'wait':
                #snn_net = SCNN() #随机重置网络参数
                snn_net.load_state_dict(init_state)
                #global optimizer
                #optimizer = torch.optim.Adam(snn_net.parameters(), lr=learning_rate)
                #distillation(epochs_distillation)
                if simulation:
                    print('--模拟模式：客户端蒸馏训练--')
                else:
                    train(snn_net, epochs=epochs_distillation, dataset=dataset_share, bs=batch_share,
                          spikes=spikes, lr=learning_rate_s) #蒸馏训练
                    test()

            #for i in range(epochs):
            #train(epochs)
            #print(batch_size)
            for i in range(epochs):
                if simulation:
                    print('--模拟模式：客户端训练第%d轮--' % (i,) )
                else:
                    train(snn_net, epochs=5, lossf=torch.nn.CrossEntropyLoss(),
                          dataset=train_sets, bs=batch_size,
                          onehot=False, num_classes=num_classes, lr=learning_rate) #本地训练
                    #if i % 5 == 4:
                    test()

            #print_net_state(snn_net.state_dict())

            if frame == 'fedAvg':
                print('训练完成，上传参数...')
                dump_and_send_pickle(sock, snn_net.state_dict())
                # sock.recv(1)
            elif frame == 'fedDis' or frame == 'fedAd1':
                spikes = compute_spikes(snn_net=snn_net, dataset_share=dataset_share, batch=batch_share, layer='all')
                ###
                spikes_hint = None
                if hint_layer:
                    spikes_hint = compute_spikes(snn_net=snn_net, dataset_share=dataset_share, batch=batch_share, layer='hint')
                ###
                if simulation:
                    print('--模拟模式：验证集测试并上传准确率--')
                    acc_r = 1
                else:
                    test(2000, sets=val_sets, record=True) #记录验证集准确率用于计算聚合权值
                print('蒸馏完成，压缩并上传脉冲张量...')

                spikes_compress = spike_tensor_compress(spikes)
                spikes_compress_list = spikes_compress.tolist()
                #dump_and_send(sock, spikes_compress_list)
                send_list(sock, spikes_compress_list)
                sock.recv(1)

                #spikes_compress = spike_tensor_compress(spikes_hint)
                #spikes_compress_list = spikes_compress.tolist()

                ###改成压缩后上传
                if hint_layer:
                    spikes_compress_hint = spike_tensor_compress(spikes_hint)
                    spikes_compress_list_hint = spikes_compress_hint.tolist()
                    #dump_and_send(sock, spikes_compress_list_hint)
                    send_list(sock, spikes_compress_list_hint)
                    sock.recv(1)
                ###

                dump_and_send(sock, acc_r)

            if use_confusion_matrix:
                _, confusion_matrix = test(s=-1, sets=dataset_test, record=False, conf=True, classes=10)
                saver.add_matrix(confusion_matrix)
                saver.save_matrix()

        elif comment == '\3':
            print('下载新的网络参数并解压缩...')
            sock.send('\0'.encode())
            # 接收网络参数
            spikes_compress_list = None
            if frame == 'fedDis' or frame == 'fedAd1':
                #spikes_compress_list = receive_and_load(sock)
                spikes_compress_list = receive_list(sock)

            elif frame == 'fedAvg':
                net_state = receive_and_load_pickle(sock)
                #net_state_list = receive_and_load(sock)
                #net_state = torch.tensor(net_state_list)

            sock.send('\0'.encode())

            #print_net_state(snn_net.state_dict(), net_state)

            if spikes_compress_list is not None:
                spikes_compress = torch.tensor(spikes_compress_list)
                spikes = spike_tensor_depress(spikes_compress)
            else:
                snn_net.load_state_dict(net_state)

        elif comment == '\4':
            sock.send('\0'.encode())
            sock.recv(1)
            break

'''class trans_information():
    def __init__(self):
        global datasets_size
        global buffer_size
        self.datasets_size = datasets_size
        self.buf_size = buffer_size'''


def load_parameters(parameters):
    global samples_size
    global num_classes
    global batch_size
    global learning_rate
    samples_size = parameters['samples_size']
    num_classes = parameters['num_classes']
    batch_size = parameters['batch_size']
    learning_rate = parameters['learning_rate']
    if frame == 'fedDis' or frame == 'fedAd1':
        global temperature
        temperature = parameters['temperature']
    #learning_rate = parameters.learning_rate


global datasets_size
snn_net = SCNN()
datasets = None
train_sets = None
val_sets = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
buffer_size = 1024
temperature = 100

samples_size = 80
if dataset_name == 'emnist':
    num_classes = 47
elif dataset_name == 'cifar100':
    num_classes = 20
else:
    num_classes = 10
learning_rate = 0.1
batch_size = 128 #也需要服务器提供，待补充
#optimizer = torch.optim.Adam(snn_net.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.95, weight_decay = 1e-4)

global spikes
acc_r = 0.0


'''def compute_spikes(batch):
    #计算脉冲张量
    global snn_net
    global spikes
    test_loader = DataLoader(dataset_share, batch_size=batch, num_workers=2)
    snn_net.to(device)
    functional.reset_net(snn_net)


    spike_tensor_list = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader)):
            inputs = inputs.to(device)
            _, outputs = snn_net(inputs, temperature)
            #spike_tensor[batch_idx] = outputs
            spike_tensor_list.append(outputs)

    spike_size = [len(spike_tensor_list)]
    spike_size.extend(spike_tensor_list[0].size())
    spike_tensor = torch.zeros(spike_size)

    for i in range(len(spike_tensor_list)):
        spike_tensor[i] = spike_tensor_list[i]

    spikes = spike_tensor'''


def test(s = 8000, sets = dataset_test, record = False, conf = False, classes=10):
    correct = 0
    total = 0
    global snn_net
    if s > 0:
        test_loader = DataLoader(sub(sets, s), batch_size=batch_size, shuffle=True, num_workers=2)
    else:
        test_loader = DataLoader(sets, batch_size=batch_size, shuffle=True, num_workers=2)

    if conf:
        confusion_matrix = torch.zeros([classes, classes], dtype=torch.int)

    snn_net.to(device)
    functional.reset_net(snn_net)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if dataset_name == 'cifar100':
                targets = fine_to_coarse(targets)
            inputs = inputs.to(device)
            # optimizer.zero_grad()
            if frame == 'fedAvg':
                outputs = snn_net(inputs)
            elif frame == 'fedDis' or frame == 'fedAd1':
                outputs,_ = snn_net(inputs)
            # labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
            # loss = criterion(outputs.cpu(), labels_)
            _, predicted = outputs.cpu().max(1)
            if conf:
                for i in range(len(predicted)):
                    confusion_matrix[targets[i]][predicted[i]] += 1
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
            functional.reset_net(snn_net)
            if batch_idx % 100 == 0:
                acc = 100. * float(correct) / float(total)
                print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)

    acc = correct / total
    print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * acc))
    if record:
        global acc_r
        acc_r = acc
    if conf:
        return acc, confusion_matrix
    return acc


'''def train(epochs=5):
    global samples_size
    global learning_rate
    global snn_net
    #optimizer = torch.optim.SGD(snn_net.parameters(), lr=learning_rate, momentum=0.95)
    optimizer = torch.optim.Adam(snn_net.parameters(), lr=learning_rate, )
    running_loss = 0
    #train_loader = getloader(samples_size)
    train_loader = DataLoader(train_sets, batch_size=batch_size, pin_memory=False)
    loss_function = torch.nn.MSELoss()

    snn_net.to(device)
    snn_net.train()
    for e in tqdm(range(epochs)):
        for i, (images, labels) in enumerate(train_loader):
            snn_net.zero_grad()
            optimizer.zero_grad()
            # print('获取训练数据')
            images = images.float().to(device)
            # print('前向传播')
            if frame == 'fedAvg':
                outputs = snn_net(images)
            elif frame == 'fedDis':
                outputs,_ =snn_net(images)
            # print('标签处理')
            labels_ = F.one_hot(labels, num_classes).float()
            # print('损失函数计算')
            loss = loss_function(outputs.cpu(), labels_)  # cpu? ###
            running_loss += loss.item()
            # print('反向传播')
            loss.backward()
            # print('参数优化')
            optimizer.step()
            # 使用spikingjelly必须加这句
            functional.reset_net(snn_net)

        #print('--设备已完成第%d轮训练，数据集大小：%d--' % (e + 1, len(train_sets)))

    del train_loader
    torch.cuda.empty_cache()
        # 1225


def distillation(epochs_distillation=1):
    global snn_net
    global spikes
    running_loss = 0
    distillation_loader = DataLoader(dataset_share, batch_size=batch_share, pin_memory=False)
    #optimizer_dis = torch.optim.SGD(snn_net.parameters(), lr=learning_rate_s, momentum=0.9)
    optimizer_dis = torch.optim.Adam(snn_net.parameters(), lr=learning_rate_s, )
    loss_function = my_loss_ann if net_name == 'ANN' else my_loss

    snn_net.to(device)
    snn_net.train()
    for e in tqdm(range(epochs_distillation)):
        for i, (images, labels) in enumerate(distillation_loader):
            snn_net.zero_grad()
            optimizer_dis.zero_grad()
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
            running_loss += loss.item()
            # print('反向传播')
            loss.backward()
            # print('参数优化')
            optimizer_dis.step()
            # 使用spikingjelly必须加这句
            functional.reset_net(snn_net)
            torch.cuda.empty_cache()
    del distillation_loader
    torch.cuda.empty_cache()
    # 1225'''

def send_list(sock:socket.socket, list_for_send:list, max_length = 100, buf_size:int=1024):
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
        dump_and_send(sock, list_for_send[i * max_length : (i+1) * max_length], buf_size)
    if remain != 0:
        dump_and_send(sock, list_for_send[turns * max_length : ], buf_size)


def receive_list(sock:socket.socket, buf_size:int=1024):
    turns = json.loads(sock.recv(1024))
    sock.send('\0'.encode())
    list_r = []
    for i in tqdm(range(turns)):
        l = receive_and_load(sock, buf_size)
        list_r.extend(l)
    return list_r


def dump_and_send(sock:socket.socket, any_for_send, buf_size:int=1024):
    while True:
        segment = json.dumps(any_for_send).encode()
        length = len(segment)
        if length == 0:
            turns = 0
        else:
            turns = (length-1)//buf_size + 1
        sock.send(json.dumps(length).encode())
        sock.recv(1)
        sock.send(segment)
        back = sock.recv(1).decode()
        if back == '\0':
            break
        else:
            sock.send('\0'.encode())
            print('---重传---')

'''def receive_and_load(sock:socket.socket, buf_size:int=1024):
    turns = pickle.loads(sock.recv(1024))
    sock.send('\0'.encode())
    segment_recv = b''
    for i in tqdm(range(int(turns))):
        segment_recv += sock.recv(buf_size)
    obj = pickle.loads(segment_recv)
    return obj'''

def receive_and_load(sock:socket.socket, buf_size:int=1024):
    retry = 0
    while True:
        length = json.loads(sock.recv(1024))
        turns = length // buf_size
        remain = length % buf_size

        sock.send('\0'.encode())
        segment_recv = b''
        segments = list([])
        for i in range(turns):
            rec = sock.recv(buf_size, socket.MSG_WAITALL)
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

        obj = None
        try:
            obj = json.loads(segment_recv, strict=False)
        except json.decoder.JSONDecodeError:
            if retry >= 3:
                break
            sock.send('\9'.encode())
            sock.recv(1)
            retry += 1
            print('---重传---')
        else:
            break

    sock.send('\0'.encode())
    return obj


def dump_and_send_pickle(sock:socket.socket, any_for_send, buf_size:int=1024):
    while True:
        segment = pickle.dumps(any_for_send)
        length = len(segment)
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

        obj = None
        try:
            obj = pickle.loads(segment_recv)
        except pickle.PickleError:
            if retry >= 3:
                break
            print('---加载不正确---')
            sock.send('\2'.encode())

            back = 'b'
            while back != '\1':
                back = sock.recv(1)
                try:
                    back = back.decode()
                    print('debug: back =',back)
                except UnicodeDecodeError:
                    back = 'b'
                    continue

                #back = sock.recv(1).decode()

            sock.send('\0'.encode())
            retry += 1

        else:
            break

    sock.send('\0'.encode())
    return obj

