
frame = 'fedDis' #fedDis or fedAd1 or fedAvg  所应用的方法
net_name = 'SNN' #所应用的snn网络模型，SNN，SVGG或newSVGG
hint_layer = True #是否应用提示层
sub_layer = hint_layer and True  #是否应用卷积子层
compression = True #是否应用脉冲张量压缩
dataset_name = 'fmnist'  #所应用的训练数据集
shareset_name = 'fmnist'  #公共数据集
min_device_number = 4  #最小设备数。只有当接入的设备大于等于这个数时才开始联邦学习
total_epochs = 50  #全局总轮数
n_clientDataset = 12000  #每个设备具有的私有数据集数量
share_clip = 12000 if frame == 'fedDis' else 0  #公共数据集数量
n_testDataset = 20000  #测试集数量。大于测试集总数时会应用全部
learning_rate_s = 1e-4 #蒸馏训练学习率。测试：0-cka,0.0015pass; 1-cka,0.00015,64,两次训练; 2-cka,0.00015,64; 3-cka，0.00015; 联邦平均：39.36
loss_f_distillation = 'mes'  #蒸馏训练损失函数，mse或cka
lr_train = 1e-4 #训练学习率。0.00015:cinic10
time_steps = 8  #snn时间步（时间窗口大小）
sub_layer_size = 32  #卷积子层数量

tcp_port = 8510  #网络通信tcp端口号
do_save = False if min_device_number == 1 else True  #是否保存训练结果
simulation = False  #模拟训练模式
use_confusion_matrix = (not simulation) and False #计算并保存混淆矩阵

def get_configs():
    configs = {'frame': frame,
               'net_name': net_name,
               'hint_layer': hint_layer,
               'sub_layer': sub_layer,
               'compression': compression,
               'dataset_name': dataset_name,
               'shareset_name': shareset_name,
               'min_device_number': 4,
               'total_epochs': total_epochs,
               'n_clientDataset': n_clientDataset,
               'share_clip': share_clip,
               'n_testDataset': n_testDataset,
               'learning_rate_s': learning_rate_s,
               'loss_f_distillation': loss_f_distillation,
               'lr_train': lr_train,
               'time_steps': time_steps,
               'sub_layer_size': sub_layer_size,
               }
    return configs