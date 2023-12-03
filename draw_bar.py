import matplotlib.pyplot as plt
import numpy as np

fig_dir = './result_save/mnist_targetacc' + '.svg'
data_dis_mnist = [1.734, 15.42]
data_avg_mnist = [133.6, 200.5]
data_dis_fmnist = [0.4353, 9.585]
data_avg_fmnist = [148.6, 297.1]
data_dis_cifar = [1.704, 3.393, 13.54]
data_avg_cifar = [384.8, 481.0, 577.2]
labels_mnist = ['0.55', '0.95', ]
labels_fmnist = ['0.78', '0.84', ]
labels_cifar = ['0.38', '0.53', '0.59']

plt.title('MNIST')
plt.ylabel("Upstream Communication [Unit: MB]")
plt.xlabel("Target Accuracy")

x = np.arange(2)
total_width, n = 0.8, 2
width = total_width / n
x = x - (total_width - width) / 2

plt.bar(x + 0.1, data_dis_mnist, width=0.2, )
plt.bar(x + width - 0.1, data_avg_mnist, width=0.2, )
plt.legend(labels=['FSD','FedAvg'],loc='best')

plt.xticks([0,1,], labels_mnist)

plt.savefig(fig_dir, format='svg')