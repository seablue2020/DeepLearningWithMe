"""
文件名: Code/Chapter03/C11_DigitClassification/main.py
创建时间: 2023/1/17 21:37 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import matplotlib.pyplot as plt


def load_dataset():
    data = MNIST(root='~/Datasets/MNIST', train=True, download=True,
                 transform=transforms.ToTensor())
    # ToTenser()将原始图片由[0,255]的整数值转换为[0.0,1.0]的浮点数值
    # 目的是为了后续的归一化处理以及神经网络的训练
    # shape: [1,28,28]，1表示单通道灰度图
    return data


def visualization_loss(losses):
    plt.plot(range(len(losses)), losses)
    plt.xlabel('迭代次数', fontsize=15)
    plt.ylabel('损失值', fontsize=15)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.ylim(-.05, 0.5)
    plt.tight_layout()
    plt.show()


def train(data):
    epochs = 2  # 训练轮数
    lr = 0.03   # 学习率
    batch_size = 128  # 批次大小
    input_node = 28 * 28
    output_node = 10
    losses = []
    data_iter = DataLoader(data, batch_size=batch_size, shuffle=True)
    net = nn.Sequential(nn.Flatten(), nn.Linear(input_node, output_node))
    loss = nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # 定义优化器
    for epoch in range(epochs):
        for i, (x, y) in enumerate(data_iter):
            logits = net(x)
            l = loss(logits, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()  # 执行梯度下降
            acc = (logits.argmax(1) == y).float().mean().item()
            print(f"Epochs[{epoch + 1}/{epochs}]--batch[{i}/{len(data_iter)}]"
                  f"--Acc: {round(acc, 4)}--loss: {round(l.item(), 4)}")
            losses.append(l.item())
    return losses


if __name__ == '__main__':
    data = load_dataset()
    losses = train(data)
    visualization_loss(losses)
