'''
QT-Opt: Q-value assisted CEM policy learning,
for reinforcement learning on robotics.

QT-Opt: https://arxiv.org/pdf/1806.10293.pdf
CEM: https://www.youtube.com/watch?v=tNAIHEse7Ms

Pytorch implementation
CEM for fitting the action directly, action is not directly dependent on state.
Actually CEM could used be fitting any part (the variable x or the variable y that parameterizes the variable x):
Q(s,a), a=w*s+b, CEM could fit 'Q' or 'a' or 'w', all possible and theoretically feasible. 
'''


import math
import random
import argparse

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical
from collections import namedtuple
import pickle
#from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
#from IPython.display import display
from reacher import Reacher
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()

GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch)) # stack for each element
        ''' q
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class ContinuousActionLinearPolicy(object):
    def __init__(self, theta, state_dim, action_dim):
        assert len(theta) == (state_dim + 1) * action_dim
        self.W = theta[0 : state_dim * action_dim].reshape(state_dim, action_dim)
        self.b = theta[state_dim * action_dim : None].reshape(1, action_dim)
    def act(self, state):
        # a = state.dot(self.W) + self.b
        a = np.dot(state, self.W) + self.b
        return a
    def update(self, theta):
        self.W = theta[0 : state_dim * action_dim].reshape(state_dim, action_dim)
        self.b = theta[state_dim * action_dim : None].reshape(1, action_dim)


class CEM():
    ''' 
    cross-entropy method, as optimization of the action policy 
    '''
    def __init__(self, theta_dim, ini_mean_scale=0.0, ini_std_scale=1.0):
        self.theta_dim = theta_dim
        self.initialize(ini_mean_scale=ini_mean_scale, ini_std_scale=ini_std_scale)

    def initialize(self, ini_mean_scale=0.0, ini_std_scale=1.0):
        self.mean = ini_mean_scale*np.ones(self.theta_dim)
        self.std = ini_std_scale*np.ones(self.theta_dim)
        
    def sample(self):
        # theta = self.mean + np.random.randn(self.theta_dim) * self.std
        theta = self.mean + np.random.normal(size=self.theta_dim) * self.std
        return theta

    def sample_multi(self, n):
        theta_list=[]
        for i in range(n):
            theta_list.append(self.sample())
        return np.array(theta_list)


    def update(self, selected_samples):
        self.mean = np.mean(selected_samples, axis = 0)
        # print('mean: ', self.mean)
        self.std = np.std(selected_samples, axis = 0)  # plus the entropy offset, or else easily get 0 std
        # print('std: ', self.std)

        return self.mean, self.std


class QNetwork(nn.Module):
    def __init__(self,  input_dim , init_w=3e-3):
        super(QNetwork, self).__init__()

        # self.linear3 = nn.Linear(hidden_dim, 1)
        #
        # self.linear3.weight.data.uniform_(-init_w, init_w)
        # self.linear3.bias.data.uniform_(-init_w, init_w)

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1)  # （46，46，16)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)  # （44，44，32)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1) # （42，42，32)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1) # （40，40，32)
        self.bn4 = nn.BatchNorm2d(32)

        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)

        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1)  # （38，38，32)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=1)  #（36，36，32)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.Conv2d(32, 32, kernel_size=3, stride=1)  #（16，16，32)
        self.bn7 = nn.BatchNorm2d(32)

        self.linear3 = nn.Linear(16*16*32, 64)
        self.linear4 = nn.Linear(64, 64)
        self.linear5 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, state, action):
        x_1 = F.relu(self.bn1(self.conv1(state)))
        x_1 = F.relu(self.bn2(self.conv2(x_1)))
        x_1 = F.relu(self.bn3(self.conv3(x_1)))
        x_1 = F.relu(self.bn4(self.conv4(x_1)))

        x_2 = F.relu(self.linear1(action))
        x_2 = F.relu(self.linear2(x_2))
        x_2 = x_2.view(-1, 32, 1, 1)
        #print(x_1.shape)
        #print(x_2.shape)
        x = x_1 + x_2

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.max_pool2d(x, 2)          #（18，18，32)
        x = F.relu(self.bn7(self.conv7(x)))

        x = x.view(-1, self.num_flat_features(x))  # view函数将张量x变形成一维的向量形式，总特征数并不改变，为接下来的全连接作准备。
        #print(x.shape)
        x = F.relu(self.linear3(x))  # 输入x经过全连接1，再经过ReLU激活函数，然后更新x
        x = F.relu(self.linear4(x))  # 输入x经过全连接2，再经过ReLU激活函数，然后更新x
        x = self.linear5(x)
        x = self.sigmoid(x)
        #print('x',x.shape)
        # x = torch.cat([state, action], 1) # the dim 0 is number of samples
        # x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))
        # x = self.linear3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 这里为什么要使用[1:],是因为pytorch只接受批输入，也就是说一次性输入好几张图片，那么输入数据张量的维度自然上升到了4维。【1:】让我们把注意力放在后3维上面
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class QT_Opt():
    def __init__(self, replay_buffer, action_dim, q_lr=3e-4, cem_update_itr=2, select_num=6, num_samples=64):
        self.num_samples = num_samples
        self.select_num = select_num
        self.cem_update_itr = cem_update_itr
        self.replay_buffer = replay_buffer
        self.qnet = QNetwork(action_dim).to(device) # gpu
        self.target_qnet1 = QNetwork(action_dim).to(device)
        self.target_qnet2 = QNetwork(action_dim).to(device)
        self.cem = CEM(theta_dim = action_dim)  # cross-entropy method for updating

        self.q_optimizer = optim.Adam(self.qnet.parameters(), lr=q_lr)
        self.step_cnt = 0

    def update(self, batch_size, gamma=0.9, soft_tau=1e-2, update_delay=100):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        self.step_cnt+=1

        
        state_      = torch.FloatTensor(state).to(device)
        next_state_ = torch.FloatTensor(next_state).to(device)
        action     = torch.FloatTensor(action).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        predict_q = self.qnet(state_, action) # predicted Q(s,a) value

        # get argmax_a' from the CEM for the target Q(s', a')
        new_next_action = []
        for i in range(batch_size):      # batch of states, use them one by one, to prevent the lack of memory
            new_next_action.append(self.cem_optimal_action(next_state[i]))
        new_next_action=torch.FloatTensor(new_next_action).to(device)

        target_q_min = torch.min(self.target_qnet1(next_state_, new_next_action), self.target_qnet2(next_state_, new_next_action))
        target_q = reward + (1-done)*gamma*target_q_min
        loss_function = nn.BCELoss()
        #print('pred',predict_q.shape)
        #print('target',target_q.detach().shape)
        loss = loss_function(predict_q.squeeze(), target_q.detach().squeeze())
        #q_loss = ((predict_q - target_q.detach())**2).mean()  # MSE loss, note that original paper uses cross-entropy loss
        #print(q_loss)
        print(loss)
        
        self.q_optimizer.zero_grad()
        #q_loss.backward()
        loss.backward()
        self.q_optimizer.step()

        # update the target nets, according to original paper:
        # one with Polyak averaging, another with lagged/delayed update
        self.target_qnet1=self.target_soft_update(self.qnet, self.target_qnet1, soft_tau)
        self.target_qnet2=self.target_delayed_update(self.qnet, self.target_qnet2, update_delay)
    


    def cem_optimal_action(self, state):
        ''' evaluate action wrt Q(s,a) to select the optimal using CEM '''
        cuda_states = torch.FloatTensor(np.vstack([state[np.newaxis,:,:,:]]*self.num_samples)).to(device)
        #print(state[np.newaxis,:,:,:].shape)
        #print(np.vstack([state[np.newaxis,:,:,:]]*self.num_samples).shape)
        #print(cuda_states.shape)
        self.cem.initialize() # every time use a new cem, cem is only for deriving the argmax_a'
        for itr in range(self.cem_update_itr):
            actions = self.cem.sample_multi(self.num_samples)
            #print(itr)
            #print('action',actions.shape)
            q_values = self.target_qnet1(cuda_states, torch.FloatTensor(actions).to(device)).detach().cpu().numpy().reshape(-1) # 2 dim to 1 dim
            #print('q_values',q_values.shape)
            max_idx=q_values.argsort()[-1]  # select one maximal q
            idx = q_values.argsort()[-int(self.select_num):]  # select top maximum q
            selected_actions = actions[idx]
            _,_=self.cem.update(selected_actions)
        optimal_action = actions[max_idx]
        return optimal_action
 

    def target_soft_update(self, net, target_net, soft_tau):
        ''' Soft update the target net '''
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        return target_net

    def target_delayed_update(self, net, target_net, update_delay):
        ''' delayed update the target net '''
        if self.step_cnt%update_delay == 0:
            for target_param, param in zip(target_net.parameters(), net.parameters()):
                target_param.data.copy_(  # copy data value into target parameters
                    param.data 
                )

        return target_net

    def save_model(self, path):
        torch.save(self.qnet.state_dict(), path)
        torch.save(self.target_qnet1.state_dict(), path)
        torch.save(self.target_qnet2.state_dict(), path)

    def load_model(self, path):
        self.qnet.load_state_dict(torch.load(path))
        self.target_qnet1.load_state_dict(torch.load(path))
        self.target_qnet2.load_state_dict(torch.load(path))
        self.qnet.eval()
        self.target_qnet1.eval()
        self.target_qnet2.eval()




def plot(rewards):
     # clear_output(True)
    plt.figure(figsize=(20,5))
    # plt.subplot(131)
    plt.plot(rewards)
    plt.savefig('qt_opt_kuka.png')
    # plt.show()

if __name__ == '__main__':

    # choose env
    ENV = ['Pendulum', 'Reacher', 'Kuka'][1]
    '''if ENV == 'Reacher':
        NUM_JOINTS=2
        LINK_LENGTH=[200, 140]
        INI_JOING_ANGLES=[0.1, 0.1]
        # NUM_JOINTS=4
        # LINK_LENGTH=[200, 140, 80, 50]
        # INI_JOING_ANGLES=[0.1, 0.1, 0.1, 0.1]
        SCREEN_SIZE=1000
        SPARSE_REWARD=False
        SCREEN_SHOT=False
        action_range = 10.0

        env=Reacher(screen_size=SCREEN_SIZE, num_joints=NUM_JOINTS, link_lengths = LINK_LENGTH, \
        ini_joint_angles=INI_JOING_ANGLES, target_pos = [369,430], render=True, change_goal=False)
        action_dim = env.num_actions
        state_dim  = env.num_observations
    elif ENV == 'Pendulum':
        env = gym.make("Pendulum-v0").unwrapped
        action_dim = env.action_space.shape[0]
        state_dim  = env.observation_space.shape[0]
        action_range=1.'''
    #elif ENV == 'Kuka':
    env = KukaDiverseObjectEnv(renders=False, isDiscrete=False, removeHeightHack=True, maxSteps=15)
    action_range = 1.
    action_dim = env.action_space.shape[0]    #
    #state_dim = env.observation_space.shape  # (48, 48, 3)



    hidden_dim = 512
    batch_size=100
    model_path = 'qt_opt_model'

    replay_buffer_size = 1e5
    replay_buffer = ReplayBuffer(replay_buffer_size)

    qt_opt = QT_Opt(replay_buffer, action_dim)
    
    if args.train:
        # hyper-parameters
        max_episodes  = 100000
        # max_steps   = 20 if ENV ==  'Reacher' else 150  # Pendulum needs 150 steps per episode to learn well, cannot handle 20
        max_steps = 15
        frame_idx   = 0
        episode_rewards = []

        for i_episode in range (max_episodes):
            '''            
            if ENV == 'Reacher':
                state = env.reset(SCREEN_SHOT)
            elif ENV == 'Pendulum':
                state =  env.reset()
            elif ENV == 'Kuka':'''
            state =  env.reset()

            episode_reward = 0

            for step in range(max_steps):
                # action = qt_opt.policy.act(state)  
                action = qt_opt.cem_optimal_action(state)
                '''if ENV ==  'Reacher':
                    next_state, reward, done, _ = env.step(action, SPARSE_REWARD, SCREEN_SHOT)
                elif ENV ==  'Pendulum':
                    next_state, reward, done, _ = env.step(action) 
                    env.render()
                elif ENV == 'Kuka':'''
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break

            if len(replay_buffer) > batch_size:
                qt_opt.update(batch_size)
                qt_opt.save_model(model_path)

            episode_rewards.append(episode_reward)
            
            if i_episode% 10==0:
                plot(episode_rewards)
                
            print('Episode: {}  | Reward:  {}'.format(i_episode, episode_reward),flush=True)
        output = open('data.pkl', 'wb')
        pickle.dump(replay_buffer, output)
        #print(replay_buffer.position)        

    if args.test:
        qt_opt.load_model(model_path)
        pkl_file = open('data.pkl', 'rb')
        replay_buffer = pickle.load(pkl_file)
        #print(replay_buffer.position)
        # hyper-parameters
        max_episodes  = 10
        #max_steps   = 20 if ENV ==  'Reacher' else 150  # Pendulum needs 150 steps per episode to learn well, cannot handle 20
        max_steps = 20
        frame_idx   = 0
        episode_rewards = []

        for i_episode in range (max_episodes):
            '''
            if ENV == 'Reacher':
                state = env.reset(SCREEN_SHOT)
            elif ENV == 'Pendulum':
                state =  env.reset()'''
            state = env.reset()
            episode_reward = 0

            for step in range(max_steps):
                # action = qt_opt.policy.act(state)  
                action = qt_opt.cem_optimal_action(state)
                '''if ENV ==  'Reacher':
                    # next_state, reward, done, _ = env.step(action, SPARSE_REWARD, SCREEN_SHOT)
                elif ENV ==  'Pendulum':
                    next_state, reward, done, _ = env.step(action)  
                    env.render()'''
                next_state, reward, done, _ = env.step(action)               
                episode_reward += reward
                state = next_state

            episode_rewards.append(episode_reward)
            # plot(episode_rewards)
            print('Episode: {}  | Reward:  {}'.format(i_episode, episode_reward))
    
        
