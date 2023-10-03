from multiprocessing import freeze_support

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from PIL import Image
from PIL import ImageEnhance

import torch
import torch.nn as nn
from torch.nn import functional as F

# 构建神经网络
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(4 * 4 * 128, 5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        return x


# 训练函数
def trainfunc(dataloader, validloader, net, lossfunc, optimizer):
    # 分batch批次训练，最终全部数据都经过一次训练
    # 每一批训练后，对参数在原有基础上进行一次更新，对新误差进行计算
    for batch, (data, target) in enumerate(dataloader):
        # 现有参数下迭代一次网络
        output = net(data)
        # forward：将数据传入模型，前向传播求出预测的值
        # 求误差，优化，更新参数
        loss = 0  # 误差初始化为0
        optimizer.zero_grad()  # 梯度初始化为零，把loss关于weight的导数变成0
        loss = lossfunc(output, target)  # 用给定损失函数lossfunc求loss
        loss.backward()  # backward：反向传播求梯度
        optimizer.step()  # optimizer：更新所有参数
        # 网络中参数将损失函数和优化器连接

        # 全部数据训练完后输出
        if batch % 100 == 99:
            correct = 0
            print(batch, "测试")
            total = len(validset)
            for i, (images, lables) in enumerate(validloader):
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == lables).sum().item()
            print("accuracy:", correct / total)

def balance_data(dataset):
    return 1


if __name__ == '__main__':

    freeze_support()
    # 图像预处理
    # compose整合
    trans_form = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=2),  # 随机位置进行裁剪
        # 32*32（相当于没有随机）
        transforms.RandomHorizontalFlip(),  # 以0.5的概率水平翻转给定的PIL图像
        transforms.ToTensor(),  # 先由HWC转置为CHW格式；再转为float类型；最后，每个像素除以255
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 归一化
    ])

    print("preparing data...")
    # 导入下载
    # trainset0=torchvision.datasets.CIFAR10(root='D:/test_code/DDR/train/00',train=True,download=True,transform=transform)
    # ImageFolder将数据按文件夹名字分类贴上标签
    trainset = ImageFolder(root='D:/test_code/DDR/train', transform=trans_form)
    testset = ImageFolder(root='D:/test_code/DDR/test', transform=trans_form)
    validset = ImageFolder(root='D:/test_code/DDR/valid', transform=trans_form)


    print("set is ok")
    print(trainset.class_to_idx)
    print(testset.class_to_idx)
    print(validset.class_to_idx)

    print(len(trainset))
    print(len(testset))
    print(len(validset))

    # 加载
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True, num_workers=4)  # 打乱
    testloader = torch.utils.data.DataLoader(testset, batch_size=12, shuffle=True, num_workers=4)
    validloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=4)
    print("loader is ok")
    print(len(trainloader))


    net =Net()  # 构建网络

    print("net is ok")
    # 给定使用的损失函数和优化器
    # 交叉熵损失函数
    lossfunc = nn.CrossEntropyLoss()

    # optimizer=torch.optim.ASGD(net.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)

    # optim.
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 优化函数（随机梯度优化方法）
    # optimizer = optim.Adam(net.parameters(), lr=0.01)
    # optimizer=torch.optim.Adagrad(net.parameters(),lr=0.01, lr_decay=0, weight_decay=0)
    # optimizer=torch.optim.Adadelta(net.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
    optimizer = torch.optim.RMSprop(net.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0,
                                    centered=False)

    # 训练网络
    for epoch in range(20):
        print(epoch, "train")
        trainfunc(trainloader, validloader, net, lossfunc, optimizer)

    print('Finished Training')

    # 测试集
    total = len(testset)
    correct = 0
    for i, (images, lables) in enumerate(testloader):
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == lables).sum().item()
    print("accuracy:", correct / total)
