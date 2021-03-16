import gym
import math
import random
import matplotlib
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple, deque
from itertools import count
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DQN:
    def __init__(self,env):
        self.env = env
        self.memory = deque(maxlen=5000)
        self.gamma = 0.85
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.create_model()
        self.target_model =  self.create_model()
        self.target_model.load_state_dict(self.model.state_dict())
    def create_model(self):
        model = nn.Sequential(
            nn.Linear(self.env.observation_space.shape[0],8),
            nn.ReLU(),
            nn.Linear(8,4),
            nn.ReLU(),
            nn.Linear(4, self.env.action_space.n),
        )
        return model.double()
    def action(self,state):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if np.random.random() < self.epsilon:
            return torch.tensor([random.randrange(2)])
        act = torch.argmax(self.model(torch.tensor(state)))
        return act
    def remember(self,state,action,reward,new_state,done):
        self.memory.append([state,action,reward,new_state,done])
    def replay(self, epoch):
        batch_size = 512
        if len(self.memory) < batch_size:
            return
        #sample batch size
        samples =  random.sample(self.memory,batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            #target values
            target = self.target_model(torch.tensor(state.reshape(1,4)))
            if done== True:
                #if state is done no future
                target[0][action] = reward
            else:
                #q = max of target_model (next state)  * discount + reward
                q_future = max(self.target_model(torch.tensor(new_state.reshape(1,4)))).detach()
                target[0][action] = reward + q_future[0]*self.gamma
            #compute loss (predicted values, target)
            predicted_vals = self.model(torch.tensor(state.reshape(1,4)))
            l = nn.MSELoss(reduction =  'mean')
            loss = l(predicted_vals,target)
            loss = torch.clamp(loss,-1,1)
            if (epoch % 5 == 0):
                writer.add_scalar('Loss/Epoch', loss, epoch)
            optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
            optimizer.zero_grad()
            loss.backward()
            for param in self.model.parameters():
                param.grad.data.clamp_(-1,1)
            optimizer.step()
    def train_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

writer = SummaryWriter("runs/5_mse_batch_512")
env = gym.make("CartPole-v0")
agent = DQN(env)
for e in range(10000):
    cur_state =  env.reset()
    r = 0
    for i in range(500):
        action = agent.action(cur_state.reshape(1,4))
        next_state, reward, done, _ = env.step(action.item())
        agent.remember(cur_state,action, reward, next_state, done)
        agent.replay(e)
        cur_state = next_state
        r += reward
        if done:
            break
        agent.train_target_model()
    print(f"Reward:{r} Episode: {e}", end = '\r') 
    writer.add_scalar('Reward/Episode', r, e)
writer.flush()
writer.close()


