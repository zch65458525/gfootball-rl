import argparse
#初始设置
def get_args():
    arg = argparse.ArgumentParser()
    #基本参数设置
    arg.add_argument('--env-name', type=str, default='academy_3_vs_1_with_keeper')#3vs1的环境
    arg.add_argument('--total-frames', type=int, default=int(2e6))#
    arg.add_argument('--nsteps', type=int, default=128)#到128步停止
    arg.add_argument('--num-workers', type=int, default=8)#同时处理8个batch
    arg.add_argument('--batch-size', type=int, default=8)#每一批八个
    arg.add_argument('--epoch', type=int, default=4)
    arg.add_argument('--cuda', action='store_true')
    #PPO超参数设置
    arg.add_argument('--gamma', type=float, default=0.995)
    #重设环境的种子
    arg.add_argument('--seed', type=int, default=1)
    #学习率
    arg.add_argument('--lr', type=float, default=0.00008)
    #论文中参数
    arg.add_argument('--vloss-coef', type=float, default=0.5)
    arg.add_argument('--ent-coef', type=float, default=0.01)
    arg.add_argument('--tau', type=float, default=0.95)
    arg.add_argument('--eps', type=float, default=1e-5)
    arg.add_argument('--clip', type=float, default=0.27)
    #是否使用学习率
    arg.add_argument('--lr-decay', action='store_true')
    #梯度更新上限
    arg.add_argument('--max-grad-norm', type=float, default=0.5)
    #存储路径，每10次update存一次模型，展示一次
    arg.add_argument('--save-dir', type=str, default='saved_models/')
    arg.add_argument('--display-interval', type=int, default=10)
    arg.add_argument('--log-dir', type=str, default='logs/')

    args = arg.parse_args()

    return args
