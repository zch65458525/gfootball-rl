import torch
from torch import nn
from torch.nn import functional as F
#PPO网络结构
class PPO_net(nn.Module):
    def __init__(self):
        super(PPO_net, self).__init__()
        self.conv1 = nn.Conv2d(16, 32, 8, stride=4) #输入16通道，输出32通道，卷积核8*8,步长4
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2) #输入32通道，输出64通道，卷积核4*4,步长2
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1) #输入64通道，输出32通道，卷积核3*3,步长1
        self.fc1 = nn.Linear(32 * 5 * 8, 512)       #全连接层，把32*5*8的网络线性化为一个512维的向量
        #网络权值的初始化，正交初始化，解决梯度爆炸、梯度消失的问题，前面×的gain是relu函数给出的推荐值
        nn.init.orthogonal_(self.conv1.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv2.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv3.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.fc1.weight.data, gain=nn.init.calculate_gain('relu'))
        #偏置的初始化为0
        nn.init.constant_(self.conv1.bias.data, 0)
        nn.init.constant_(self.conv2.bias.data, 0)
        nn.init.constant_(self.conv3.bias.data, 0)
        nn.init.constant_(self.fc1.bias.data, 0)

    #前向传播，每个卷积层后用relu函数非线性化，全连接层之前规整
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * 5 * 8)
        x = F.relu(self.fc1(x))
        return x


class PPO_CNN(nn.Module):
    def __init__(self, num_actions):
        super(PPO_CNN, self).__init__()
        self.cnn_layer = PPO_net()
        self.critic = nn.Linear(512, 1)
        self.actor = nn.Linear(512, num_actions)
        #初始化
        nn.init.orthogonal_(self.critic.weight.data)
        nn.init.constant_(self.critic.bias.data, 0)
        nn.init.orthogonal_(self.actor.weight.data, gain=0.01)
        nn.init.constant_(self.actor.bias.data, 0)

    def forward(self, inputs):
        x = self.cnn_layer(inputs / 255.0)
        #返回critic值，来解决PG可能方差大，不收敛的问题
        value = self.critic(x)
        #返回动作分布
        pi = F.softmax(self.actor(x), dim=1)#归一化概率
        return value, pi
