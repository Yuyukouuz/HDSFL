import os
from skimage import io
import torchvision.datasets.mnist as mnist
import matplotlib.pyplot as plt

from torchvision import transforms
from dataset_config import *

save_path = './sample/emnist/'

x_data = emnist_train

for i in range(30):
    a, l = x_data[i]
    print(a.type(), a.size())
    #a = a.transpose(1,2,0)


    toPIL = transforms.ToPILImage()
    img = a
    pic = toPIL(img)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    pic.save(save_path + str(l) + '.jpg')