"""
文件名: Code/Chapter03/C10_CrossEntropy/main.py
创建时间: 2023/2/2 20:15 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""
import torch.nn as nn
import torch
import numpy as np


def crossEntropy(y_true, logits):
    # y_true: one-hot编码 [m,n]
    # logits: 经过softmax后的概率分布 [m,n]
    # m是样本个数，n是类别个数
    loss = y_true * np.log(logits)  # [m,n]
    return -np.sum(loss) / len(y_true)


def softmax(x):
    s = np.exp(x)
    return s / np.sum(s, axis=1, keepdims=True)


if __name__ == '__main__':
    logits = torch.tensor([[0.5, 0.3, 0.6], [0.5, 0.4, 0.3]])
    y = torch.LongTensor([2, 0])
    loss = nn.CrossEntropyLoss(reduction='mean')  # 返回的均值是除以的每一批样本的个数（不一定是batch_size）
    # 如果reduction = 'sum'，则返回的值是所有样本的loss之和
    l = loss(logits, y)
    print(l)  # tensor(0.9874)

    logits = np.array([[0.5, 0.3, 0.6], [0.5, 0.4, 0.3]])
    y = np.array([2, 0])
    y_one_hot = np.eye(3)[y]
    print(crossEntropy(y_one_hot, softmax(logits)))  # 0.9874308806774512
