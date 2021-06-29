import numpy as np
import torch
from torch.distributions.categorical import Categorical
import random
import logging

#选择动作
def select_actions(pi):
    actions = Categorical(pi).sample()
    #根据传入概率，选择动作，返回动作
    return actions.detach().cpu().numpy().squeeze()
    #阻断反向传播，传回CPU，让内存可以找到action向量，将tensor转为numpy

#对动作估值
def evaluate_actions(pi, actions):
    cate_dist = Categorical(pi)
    #计算在定义的正态分布中对应的概率的对数，以及熵损失函数
    log_prob = cate_dist.log_prob(actions).unsqueeze(-1)
    entropy = cate_dist.entropy().mean()
    return log_prob, entropy

#放日志
def config_logger(log_dir):
    logger = logging.getLogger()
    logger.setLevel('INFO')
    basic_format = '%(message)s'
    formatter = logging.Formatter(basic_format)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    #log存储路径
    fhlr = logging.FileHandler(log_dir)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger
