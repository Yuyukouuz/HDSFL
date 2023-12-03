import random
from newVGG_layers import *

class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        #pool = APLayer(2)
        if dataset_name.find('mnist') > 0:
            num_channel = 1
        elif dataset_name.find('dvs') > 0:
            num_channel = 2
        else:
            num_channel = 3
        self.features = nn.Sequential(
            #Layer(3,64,3,1,1), #cifar10为3通道，mnist是1，dvs是2
            Layer(num_channel, 64, 3, 1, 1),
            Layer(64,128,3,1,1),
            pool,
            Layer(128,256,3,1,1),
            Layer(256,256,3,1,1),
            pool,
            Layer(256,512,3,1,1),
            Layer(512,512,3,1,1),
            pool,
            Layer(512,512,3,1,1),
            Layer(512,512,3,1,1),
            pool,
        )
        self.features_sub = nn.Sequential(
            Layer(512,256,3,1,1), #cifar10为3通道，mnist是1，dvs是2
            Layer(256,sub_layer_size,3,1,1),
        )
        W = int(32/2/2/2/2)  #cifar是32，mnist是28，dvs是48
        W_sub = int(32/2/2/2/2)
        self.T = time_steps
        self.classifier = SeqToANNContainer(nn.Linear(512*W*W,20 if dataset_name == 'cifar100' else 10))
        self.classifier_sub = SeqToANNContainer(nn.Linear(sub_layer_size*W_sub*W_sub,20 if dataset_name == 'cifar100' else 10))
        self.act = LIFSpike()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input, temp=1): #input维度:(batch size, 3, 32, 32)
        if dataset_name != 'cifar10_dvs':
            input = add_dimention(input, self.T) #input维度:(batch size, T, 3, 32, 32)
        x = self.features(input) #x维度：(b, T, 512, *, *)

        if sub_layer:
            x = self.features_sub(x)

        x = torch.flatten(x, 2) #x维度：(b, T, 512*)

        if sub_layer:
            x = self.classifier_sub(x)  # x维度：(b, T, classes)
        else:
            x = self.classifier(x) #x维度：(b, T, classes)

        x = self.act(x)

        if frame == 'fedDis':
            return x.mean(1), x
        else:
            return x.mean(1)

    def feature_forward(self, input, temp=1):
        if len(input.shape) == 4:
            input = add_dimention(input, self.T)
        x = self.features(input) #x维度：(b, T, 512, *, *)
        if sub_layer:
            x = self.features_sub(x) #x维度：(b, T, 32, *, *)
        x = torch.flatten(x, 2) #x维度：(b, T, 32*)
        return x.mean(1), x

    def forward_merge(self, input):
        if dataset_name != 'cifar10_dvs':
            input = add_dimention(input, self.T)  # input维度:(batch size, T, 3, 32, 32)
        x = self.features(input)  # x维度：(b, T, 512, *, *)

        if sub_layer:
            x = self.features_sub(x)

        x_all = torch.flatten(x, 2)  # x维度：(b, T, 512*)
        if sub_layer:
            x_all = self.classifier_sub(x_all)  # x维度：(b, T, classes)
        else:
            x_all = self.classifier(x_all)  # x维度：(b, T, classes)
        x_all = self.act(x_all)

        x_hint = torch.flatten(x, 2)  # x维度：(b, T, 32*)
        return x_all, x_hint

class VGGSNNwoAP(nn.Module):
    def __init__(self):
        super(VGGSNNwoAP, self).__init__()
        self.features = nn.Sequential(
            Layer(2,64,3,1,1),
            Layer(64,128,3,2,1),
            Layer(128,256,3,1,1),
            Layer(256,256,3,2,1),
            Layer(256,512,3,1,1),
            Layer(512,512,3,2,1),
            Layer(512,512,3,1,1),
            Layer(512,512,3,2,1),
        )
        W = int(48/2/2/2/2)
        # self.T = 4
        self.classifier = SeqToANNContainer(nn.Linear(512*W*W,10))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x



if __name__ == '__main__':
    model = VGGSNNwoAP()