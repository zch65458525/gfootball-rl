from arguments import get_args
from PPO_agent import PPO_agent
from Networks_PPO import PPO_CNN
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import gfootball.env as football_env
import os

#创建足球环境
def create_single_football_env(args):
    env = football_env.create_environment(env_name=args.env_name, stacked=True)
    return env

if __name__ == '__main__':
    params = get_args()
    #对每个worker设立独立的环境，在其中执行各个函数
    envs = SubprocVecEnv([(lambda _i=i: create_single_football_env(params)) for i in range(params.num_workers)])
    network = PPO_CNN(envs.action_space.n)
    #初始化PPO
    PPO_trainer = PPO_agent(envs, params, network)
    PPO_trainer.train()
    #训练王关闭环境
    envs.close()
