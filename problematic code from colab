!apt-get update
!apt-get install libsdl2-gfx-dev libsdl2-ttf-dev

# Make sure that the Branch in git clone and in wget call matches !!
!git clone -b v2.9 https://github.com/google-research/football.git
!mkdir -p football/third_party/gfootball_engine/lib

!wget https://storage.googleapis.com/gfootball/prebuilt_gameplayfootball_v2.8.so -O football/third_party/gfootball_engine/lib/prebuilt_gameplayfootball.so
!cd football && GFOOTBALL_USE_PREBUILT_SO=1 python3 -m pip install .




import numpy as np
import torch
from torch.distributions.categorical import Categorical
import random
import logging

# select actions
def select_actions(pi):
    actions = Categorical(pi).sample()
    # return actions
    return actions.detach().cpu().numpy().squeeze()

# evaluate actions
def evaluate_actions(pi, actions):
    cate_dist = Categorical(pi)
    log_prob = cate_dist.log_prob(actions).unsqueeze(-1)
    entropy = cate_dist.entropy().mean()
    return log_prob, entropy

# configure the logger
def config_logger(log_dir):
    logger = logging.getLogger()
    # we don't do the debug...
    logger.setLevel('INFO')
    basic_format = '%(message)s'
    formatter = logging.Formatter(basic_format)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    # set the log file handler
    fhlr = logging.FileHandler(log_dir)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger
    
    
    
    
    
import numpy as np
import torch
from torch import optim
from datetime import datetime
import os
import copy
import torch.nn as nn

class ddqn_agent:
    def __init__(self, envs, args, net, env_net=None):
        self.envs = envs
        self.args = args
        # define the newtork...
        self.net = net
        self.old_net = copy.deepcopy(self.net)
        # if use the cuda...
        if self.args.cuda:
            self.net.cuda()
            self.old_net.cuda()
        # define the optimizer...
        self.optimizer = optim.Adam(self.net.parameters(), self.args.lr, eps=self.args.eps)
        # check saving folder..
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # env folder..
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        # logger folder
        if not os.path.exists(self.args.log_dir):
            os.mkdir(self.args.log_dir)
        self.log_path = self.args.log_dir + self.args.env_name + '.log'
        # get the observation
        self.batch_ob_shape = (self.args.num_workers * self.args.nsteps, ) + self.envs.observation_space.shape
        self.obs = np.zeros((self.args.num_workers, ) + self.envs.observation_space.shape, dtype=self.envs.observation_space.dtype.name)
        self.obs[:] = self.envs.reset()
        self.dones = [False for _ in range(self.args.num_workers)]
        self.logger = config_logger(self.log_path)
        self.env_net = env_net
        self.policy_optimizer = optim.Adam(self.net.parameters(), lr=1e-5)
        self.env_optimizer = optim.Adam(self.env_net.parameters(), lr=1e-5)
        self.obs_shape = envs.observation_space.shape

    # start to train the network...
    def learn(self):
        num_updates = self.args.total_frames // (self.args.nsteps * self.args.num_workers)
        # get the reward to calculate other informations
        episode_rewards = torch.zeros([self.args.num_workers, 1])
        final_rewards = torch.zeros([self.args.num_workers, 1])
        reward_hist = []
        policy_loss_hist = []
        env_loss_hist = []
        for update in range(num_updates):
            mb_obs, mb_rewards, mb_actions, mb_nextobs, mb_dones, mb_values, Qvalues = [], [], [], [], [], [], []
            if self.args.lr_decay:
                self._adjust_learning_rate(update, num_updates)
            for step in range(self.args.nsteps):
                with torch.no_grad():
                    # get tensors
                    obs_tensor = self._get_tensors(self.obs)
                    _, q = self.net(obs_tensor)
                # select actions
                actions = np.zeros([8, ])
                # for i in range(8):
                    # if np.abs(np.random.randn(1)) < 0.3:
                    #     actions[i] = np.int(np.random.choice(np.arange(self.envs.action_space.n)))
                    # else:
                actions = select_actions(q)
                # get the input actions
                input_actions = actions
                # start to store information
                mb_obs.append(np.copy(self.obs))
                mb_actions.append(actions)
                mb_dones.append(self.dones)
                # start to excute the actions in the environment
                obs, rewards, dones, _ = self.envs.step(input_actions)
                mb_nextobs.append(obs)
                # update dones
                self.dones = dones
                mb_rewards.append(rewards)
                # clear the observation
                for n, done in enumerate(dones):
                    if done:
                        self.obs[n] = self.obs[n] * 0

                self.obs = obs
                # process the rewards part -- display the rewards on the screen
                rewards = torch.tensor(np.expand_dims(np.stack(rewards), 1), dtype=torch.float32)
                episode_rewards += rewards
                masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in dones], dtype=torch.float32)
                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards
                episode_rewards *= masks
            # process the rollouts
            mb_obs = np.asarray(mb_obs, dtype=np.float32)
            Qvalues = np.asarray(Qvalues, dtype=np.float32)
            mb_obs = mb_obs.swapaxes(0, 1).reshape(self.batch_ob_shape)
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
            mb_nextobs = np.asarray(mb_nextobs, dtype=np.float32)
            mb_nextobs = mb_nextobs.swapaxes(0, 1).reshape(self.batch_ob_shape)

            mb_actions = np.asarray(mb_actions, dtype=np.float32)
            # before update the network, the old network will try to load the weights
            self.old_net.load_state_dict(self.net.state_dict())
            # start to update the network
            policy_loss = self._update_network(mb_obs, mb_actions, mb_rewards, mb_nextobs)
            policy_loss_hist.append(policy_loss)
            # display the training information
            reward_hist.append(final_rewards.mean().detach().cpu().numpy())
            if update % self.args.display_interval == 0:
                self.logger.info('[{}] Update: {} / {}, Frames: {}, Rewards: {:.3f}, Min: {:.3f}, Max: {:.3f}'
                                 .format(datetime.now(), update, num_updates, (update + 1)*self.args.nsteps*self.args.num_workers, \
                                final_rewards.mean().item(), final_rewards.min().item(), final_rewards.max().item()))
                # save the model
                torch.save(self.net.state_dict(), self.model_path + '/model.pt')

        return reward_hist, env_loss_hist, policy_loss_hist



    # convert the numpy array to tensors
    def _get_tensors(self, obs):
        obs_tensor = torch.tensor(np.transpose(obs, (0, 3, 1, 2)), dtype=torch.float32)
        # decide if put the tensor on the GPU
        if self.args.cuda:
            obs_tensor = obs_tensor.cuda()
        return obs_tensor

    # adjust the learning rate
    def _adjust_learning_rate(self, update, num_updates):
        lr_frac = 1 - (update / num_updates)
        adjust_lr = self.args.lr * lr_frac
        for param_group in self.policy_optimizer.param_groups:
             param_group['lr'] = adjust_lr

        # update the network
    def _update_network(self, obs, action, rewards, next_obs):
        action = np.reshape(action, [-1, 1])
        rewards = np.reshape(rewards, [-1, 1])
        lens_data = np.minimum(len(obs), len(rewards))
        inds = np.arange(lens_data)
        nbatch_train = lens_data // self.args.batch_size
        criterion = nn.SmoothL1Loss()
        for _ in range(self.args.epoch):
            np.random.shuffle(inds)
            for start in range(0, lens_data, nbatch_train):
                # get the mini-batchs
                end = start + nbatch_train
                mbinds = inds[start:end]
                mb_obs = obs[mbinds]
                mb_actions = action[mbinds]
                mb_rewards = rewards[mbinds]
                mb_nextobs = next_obs[mbinds]
                # convert minibatches to tensor
                mb_obs = self._get_tensors(mb_obs)
                _, qpred = self.net(mb_obs)
                q_select = torch.gather(qpred, 1, torch.tensor(mb_actions).long())
                mb_nextobs_tensor = self._get_tensors(mb_nextobs)
                _, Q_sa_temp = self.net(mb_nextobs_tensor)
                action_next = select_actions(Q_sa_temp)
                Q_sa = torch.gather(self.old_net(mb_nextobs_tensor)[1], 1, torch.tensor(action_next.reshape(-1, 1), dtype=torch.float32).long())
                target = torch.tensor(mb_rewards, dtype=torch.float32) + 0.9 * Q_sa.view(-1, 1)
                loss = criterion(q_select.view(-1, 1), target)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.max_grad_norm)
                # update
                self.optimizer.step()
        return loss.detach().cpu().numpy()
        
        
        
        
        
import torch
from torch import nn
from torch.nn import functional as F

"""
this network is modified for the google football
"""

# the convolution layer of deepmind
class deepmind(nn.Module):
    def __init__(self):
        super(deepmind, self).__init__()
        self.conv1 = nn.Conv2d(16, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
        self.fc1 = nn.Linear(32 * 5 * 8, 512)        
        # start to do the init...
        nn.init.orthogonal_(self.conv1.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv2.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv3.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.fc1.weight.data, gain=nn.init.calculate_gain('relu'))
        # init the bias...
        nn.init.constant_(self.conv1.bias.data, 0)
        nn.init.constant_(self.conv2.bias.data, 0)
        nn.init.constant_(self.conv3.bias.data, 0)
        nn.init.constant_(self.fc1.bias.data, 0)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * 5 * 8)
        x = F.relu(self.fc1(x))

        return x

# in the initial, just the nature CNN
class cnn_net(nn.Module):
    def __init__(self, num_actions):
        super(cnn_net, self).__init__()
        self.cnn_layer = deepmind()
        self.critic = nn.Linear(512, 1)
        self.actor = nn.Linear(512, num_actions)

        # init the linear layer..
        nn.init.orthogonal_(self.critic.weight.data)
        nn.init.constant_(self.critic.bias.data, 0)
        # init the policy layer...
        nn.init.orthogonal_(self.actor.weight.data, gain=0.01)
        nn.init.constant_(self.actor.bias.data, 0)

    def forward(self, inputs):
        x = self.cnn_layer(inputs / 255.0)
        value = self.critic(x)
        pi = F.softmax(self.actor(x), dim=1)
        return value, pi
        
        
        
        
        
!pip install baselines





import torch.nn as nn
import torch

class env_net(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        self.obs_shape = obs_shape
        m, n, p = obs_shape
        self.action_shape = 1
        self.fco1 = nn.Linear(m * n * p, 64)
        self.fca1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, obs, action):
        out1 = torch.relu(self.fco1(obs))
        out2 = torch.relu(self.fca1(action))
        out = torch.cat([out1, out2], dim=1)
        out = torch.relu(self.fc2(out))
        out = torch.sigmoid(self.fc3(out))
        return out
        
        
        
        
        
        
        
        
        
        
import argparse

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--gamma', type=float, default=0.993, help='the discount factor of RL')
    parse.add_argument('--seed', type=int, default=123, help='the random seeds')
    parse.add_argument('--num-workers', type=int, default=8, help='the number of workers to collect samples')
    parse.add_argument('--env-name', type=str, default='academy_empty_goal_close', help='the environment name')
    parse.add_argument('--batch-size', type=int, default=8, help='the batch size of updating')
    parse.add_argument('--lr', type=float, default=0.00008, help='learning rate of the algorithm')
    parse.add_argument('--epoch', type=int, default=4, help='the epoch during training')
    parse.add_argument('--nsteps', type=int, default=128, help='the steps to collect samples')
    parse.add_argument('--vloss-coef', type=float, default=0.5, help='the coefficient of value loss')
    parse.add_argument('--ent-coef', type=float, default=0.01, help='the entropy loss coefficient')
    parse.add_argument('--tau', type=float, default=0.95, help='gae coefficient')
    parse.add_argument('--cuda', action='store_true', help='use cuda do the training')
    parse.add_argument('--total-frames', type=int, default=int(1e5), help='the total frames for training')
    parse.add_argument('--eps', type=float, default=1e-5, help='param for adam optimizer')
    parse.add_argument('--clip', type=float, default=0.27, help='the ratio clip param')
    parse.add_argument('--save-dir', type=str, default='saved_models/', help='the folder to save models')
    parse.add_argument('--lr-decay', action='store_true', help='if using the learning rate decay during decay')
    parse.add_argument('--max-grad-norm', type=float, default=0.5, help='grad norm')
    parse.add_argument('--display-interval', type=int, default=10, help='the interval that display log information')
    parse.add_argument('--log-dir', type=str, default='logs/')
    parse.add_argument('--policy_model_dir', type=str, default='policy_model')
    args = parse.parse_known_args()[0]

    return args
    
    
import argparse
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import gfootball.env as football_env
import matplotlib.pyplot as plt
import numpy as np
import os

# create the environment
def create_single_football_env(args):
    """Creates gfootball environment."""
    env = football_env.create_environment(\
            env_name=args.env_name, stacked=True#, with_checkpoints=False,
            )
    return env







if __name__ == '__main__':
   mode = 'DDQN'
   if mode == 'DDQN':
        # get the arguments
        args = get_args()
        # create environments
        envs = SubprocVecEnv([(lambda _i=i: create_single_football_env(args)) for i in range(args.num_workers)])
        # create networks
        network = cnn_net(envs.action_space.n)
        env_net = env_net(envs.observation_space.shape)
        # create the ppo agent
        ddqn_trainer = ddqn_agent(envs, args, network, env_net)
        reward_hist, _, policy_loss_hist = ddqn_trainer.learn()
        np.save('reward_ddqn.npy', reward_hist)
        np.save('policy_loss_ddqn.npy', policy_loss_hist)
