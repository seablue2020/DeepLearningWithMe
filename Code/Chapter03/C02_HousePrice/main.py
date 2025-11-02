"""
文件名: Code/Chapter03/C01_HousePrice/house_price.py
创建时间: 2023/1/7 2:48 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 配置 matplotlib 使用已安装的中文字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


def make_house_data():
    """
    构造数据集
    :return:  x:shape [100,1] y:shape [100,1]
    """
    np.random.seed(20)
    x = np.random.randn(100, 1) + 5  # 面积
    noise = np.random.randn(100, 1)
    y = x * 2.8 - noise  # 价格
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return x, y


def visualization(x, y, y_pred=None):
    plt.rcParams['ytick.direction'] = 'in'  # 刻度向内
    plt.rcParams['xtick.direction'] = 'in'  # 刻度向内
    plt.xlabel('面积', fontsize=15)
    plt.ylabel('房价', fontsize=15)
    plt.scatter(x, y, c='black')
    plt.plot(x, y_pred)
    plt.tight_layout()  # 调整子图间距
    
    # 在 dev container 等 headless 环境中保存图片而不是显示窗口
    import os
    output_path = 'house_price_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存到: {os.path.abspath(output_path)}")
    
    # 如果有 GUI 环境，也尝试显示（在 headless 环境中会被忽略）
    try:
        plt.show()
    except Exception:
        pass


def train(x, y):
    epochs = 40
    lr = 0.003
    input_node = x.shape[1]
    output_node = 1
    net = nn.Sequential(nn.Linear(input_node, output_node))
    loss = nn.MSELoss()  # 定义损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # 定义优化器
    for epoch in range(epochs):
        logits = net(x)
        l = loss(logits, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()  # 执行梯度下降
        print("Epoch: {}, loss: {}".format(epoch, l))
    logits = net(x)
    l = loss(logits, y)
    print("RMSE: {}".format(torch.sqrt(l / 2)))
    return logits.detach().numpy()


if __name__ == '__main__':
    x, y = make_house_data()
    y_pred = train(x, y)
    visualization(x, y, y_pred)
