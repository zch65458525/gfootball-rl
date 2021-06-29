import numpy as np
import torch
from torch import optim
from utils import select_actions, evaluate_actions, config_logger
from datetime import datetime
import os
import copy

class PPO_agent:
    def __init__(self, envs, args, net):#初始化
        self.envs = envs 
        self.args = args
        self.net = net
        self.prev_net = copy.deepcopy(self.net) #原网络

        if self.args.cuda: #如果有GPU用GPU
            self.net.cuda()
            self.prev_net.cuda()
        #优化器初始化
        self.optimizer = optim.Adam(self.net.parameters(), self.args.lr, eps=self.args.eps)
        #确认保存的路径
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        if not os.path.exists(self.args.log_dir):
            os.mkdir(self.args.log_dir)
        self.log_path = self.args.log_dir + self.args.env_name + '.log'

        #PPO，设置batch的大小和obs空间的大小，一开始所有worker都未完成
        self.batch_ob_shape = (self.args.num_workers * self.args.nsteps, ) + self.envs.observation_space.shape
        self.obs = np.zeros((self.args.num_workers, ) + self.envs.observation_space.shape, dtype=self.envs.observation_space.dtype.name)
        self.obs[:] = self.envs.reset()
        self.dones = [False for _ in range(self.args.num_workers)]
        self.logger = config_logger(self.log_path)
        #self.net.load_state_dict(torch.load('/content/drive/MyDrive/Gfootball_Model_Env-master/PPO_Gfootball_test_and_model/saved_models/academy_3_vs_1_with_keeper/model1490.pt'))
        #如果colab上继续训需要

    def _update_network(self, obs, actions, returns, advantages):
        inds = np.arange(obs.shape[0])#shape[0]=numworkers*nsteps<=8*128=1024
        nbatch_train = obs.shape[0] // self.args.batch_size#一批8个，故minibatchsize=总步数//8<=128
        for _ in range(self.args.epoch):
            """
            generate random indeces to be minibatch sample index
            """
            np.random.shuffle(inds)
            for start in range(0, obs.shape[0], nbatch_train):

                #采样minibatch
                end = start + nbatch_train
                inds_vector = inds[start:end]
                obs_vec = obs[inds_vector]
                actions_vec = actions[inds_vector]
                returns_vec = returns[inds_vector]
                advs_vec = advantages[inds_vector]

                #转成tensor
                obs_vec = self._get_tensors(obs_vec)
                actions_vec = torch.tensor(actions_vec, dtype=torch.float32)
                returns_vec = torch.tensor(returns_vec, dtype=torch.float32).unsqueeze(1)
                advs_vec = torch.tensor(advs_vec, dtype=torch.float32).unsqueeze(1)

                #标准化
                advs_vec = (advs_vec - advs_vec.mean()) / (advs_vec.std() + 1e-8)
                if self.args.cuda:
                    actions_vec = actions_vec.cuda()
                    returns_vec = returns_vec.cuda()
                    advs_vec = advs_vec.cuda()
                values_vec, pis = self.net(obs_vec)
                
                #计算价值损失函数和策略损失函数
                value_loss = (returns_vec - values_vec).pow(2).mean()
                #根据旧网络选择动作
                with torch.no_grad():
                    _, prev_pis = self.prev_net(obs_vec)
                    prev_log_prob, _ = evaluate_actions(prev_pis, actions_vec)
                    prev_log_prob = prev_log_prob.detach()
                #评估现在的策略
                log_prob, ent_loss = evaluate_actions(pis, actions_vec)
                prob_ratio = torch.exp(log_prob - prev_log_prob)
                #tensor夹紧，稳定update
                surrogate1 = prob_ratio * advs_vec
                surrogate2 = torch.clamp(prob_ratio, 1 - self.args.clip, 1 + self.args.clip) * advs_vec
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                total_loss = policy_loss + self.args.vloss_coef * value_loss - ent_loss * self.args.ent_coef
                #梯度初始化为0
                self.optimizer.zero_grad()
                #反向转播求梯度
                total_loss.backward()
                #设置最大更新梯度，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.max_grad_norm)
                #更新参数
                self.optimizer.step()
        return policy_loss.item(), value_loss.item(), ent_loss.item()

    #把numpy转为tensor，同时边换
    def _get_tensors(self, obs):
        obs_tensor = torch.tensor(np.transpose(obs, (0, 3, 1, 2)), dtype=torch.float32)
        #如果需要，放到GPU上
        if self.args.cuda:
            obs_tensor = obs_tensor.cuda()
        return obs_tensor

    #学习率调整
    def _adjust_learning_rate(self, update, num_updates):
        lr_frac = 1 - (update / num_updates)
        adjust_lr = self.args.lr * lr_frac
        for param_group in self.optimizer.param_groups:
             param_group['lr'] = adjust_lr


    def train(self):
        num_updates = self.args.total_frames // (self.args.nsteps * self.args.num_workers)
        #一段时间update，统计回报
        episode_rewards = torch.zeros([self.args.num_workers, 1])
        final_rewards = torch.zeros([self.args.num_workers, 1])
        #中途重新开始训，要修改update
        for update in range(num_updates):
            #每次update，调整学习率
            if self.args.lr_decay:
                self._adjust_learning_rate(update, num_updates)
            #初始化训练数据的存放地
            obs_vector, rewards_vector, actions_vector, dones_vector, values_vector = [], [], [], [], []
            #开始训练
            for step in range(self.args.nsteps):
                with torch.no_grad():#得到的新tensorrequires_grad为False，不求导
                    obs_tensor = self._get_tensors(self.obs)#变换后的tensor
                    values, pis = self.net(obs_tensor) #调用forward得
                #选择动作
                actions = select_actions(pis)
                input_actions = actions 
                #存储训练数据
                obs_vector.append(np.copy(self.obs))
                actions_vector.append(actions)
                dones_vector.append(self.dones)
                values_vector.append(values.detach().cpu().numpy().squeeze())
                #得到交互结果
                obs, rewards, dones, _ = self.envs.step(input_actions)
                #比赛是否结束
                self.dones = dones
                rewards_vector.append(rewards)
                #若某个worker结束，观察清零
                for n, done in enumerate(dones):
                    if done:
                        self.obs[n] = self.obs[n] * 0
                self.obs = obs
                
                #处理reward，只有完成的结果才会被统计R
                rewards = torch.tensor(np.expand_dims(np.stack(rewards), 1), dtype=torch.float32)
                episode_rewards += rewards
                masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in dones], dtype=torch.float32)
                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards
                episode_rewards *= masks


            #更新网络
            #矩阵化
            obs_vector = np.asarray(obs_vector, dtype=np.float32)
            rewards_vector = np.asarray(rewards_vector, dtype=np.float32)
            actions_vector = np.asarray(actions_vector, dtype=np.float32)
            dones_vector = np.asarray(dones_vector, dtype=np.bool)
            values_vector = np.asarray(values_vector, dtype=np.float32)
            #计算最后阶段的值
            with torch.no_grad():
                obs_tensor = self._get_tensors(self.obs)
                last_values, _ = self.net(obs_tensor)
                last_values = last_values.detach().cpu().numpy().squeeze()
            #开始计算advantages， 在采样时，如果最后一步的状态不是即将终止的状态，由于lastgea = 0，因此获取不到下一步的 gae
            returns_vector = np.zeros_like(rewards_vector)
            advs_vector = np.zeros_like(rewards_vector)
            lastgaelam = 0

            #计算损失函数
            for t in reversed(range(self.args.nsteps)):
                if t == self.args.nsteps - 1:#步数到极限步数
                    next_non_terminal = 1.0 - self.dones
                    next_values = last_values
                else:#否则取向量里对应的数
                    next_non_terminal = 1.0 - dones_vector[t + 1]
                    next_values = values_vector[t + 1]
                delta = rewards_vector[t] + self.args.gamma * next_values * next_non_terminal - values_vector[t]
                advs_vector[t] = lastgaelam = delta + self.args.gamma * self.args.tau * next_non_terminal * lastgaelam
                #论文中公式
            returns_vector = advs_vector + values_vector
            #return = advantage+value

            #将数据reshape/展开，然后更新网络
            obs_vector = obs_vector.swapaxes(0, 1).reshape(self.batch_ob_shape)
            actions_vector = actions_vector.swapaxes(0, 1).flatten()
            returns_vector = returns_vector.swapaxes(0, 1).flatten()
            advs_vector = advs_vector.swapaxes(0, 1).flatten()
            self.prev_net.load_state_dict(self.net.state_dict())
            pl, vl, ent = self._update_network(obs_vector, actions_vector, returns_vector, advs_vector)

            #数据显示与模型保存
            if update % self.args.display_interval == 0:
                self.logger.info('[{}] Update: {} / {}, Frames: {}, Rewards: {:.3f}, Min: {:.3f}, Max: {:.3f}, PL: {:.3f},'\
                    'VL: {:.3f}, Ent: {:.3f}'.format(datetime.now(), update, num_updates, (update + 1)*self.args.nsteps*self.args.num_workers, \
                    final_rewards.mean().item(), final_rewards.min().item(), final_rewards.max().item(), pl, vl, ent))
                torch.save(self.net.state_dict(), self.model_path + '/model.pt')


