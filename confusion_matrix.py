from tools import *

parser = argparse.ArgumentParser()
parser.add_argument('--d', type=str, default='use')
parser.add_argument('--open', type=str, default='confusion-matrix_server_13-11')
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--format', type=str, default='svg')
args = parser.parse_args()

dir = f'./{args.d}/'
file_open = args.open
state = torch.load(dir + file_open)
confusion_matrix = state['confusion_matrix_list'][args.epoch].numpy()
shape = confusion_matrix.shape
for i in range(shape[0]):
    for j in range(shape[1]):
        print(int(confusion_matrix[i][j]), '\t', end='')
    print()

data_path_cifar10 = '/data/lzt/datasets/cifar10'
tests_cifar10 = torchvision.datasets.CIFAR10(
        root=data_path_cifar10, train=False, download=False)

import seaborn as sn
import pandas as pd
df_cm = pd.DataFrame(confusion_matrix,
                     index = [i for i in list(tests_cifar10.classes)],
                     columns = [i for i in list(tests_cifar10.classes)])
plt.figure(figsize = (7,6))
sn.heatmap(df_cm, annot=True, cmap="Blues", fmt='d', cbar=False, annot_kws={'size':15})
plt.xticks(fontsize=14) #x轴刻度的字体大小（文本包含在pd_data中了）
plt.yticks(fontsize=14) #y轴刻度的字体大小（文本包含在pd_data中了）
plt.savefig(f'./{args.d}/{args.open}_{args.epoch}.{args.format}', format=args.format, bbox_inches='tight', pad_inches=0)