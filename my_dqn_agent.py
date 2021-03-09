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

writer = SummaryWriter()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DQN:
    def __init__(self,env):
        self.env = env
        self.memory = deque(maxlen=2000)
        self.gamma = 0.90
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.model = self.create_model()
        self.target_model =  self.create_model()
        self.target_model.load_state_dict(self.model.state_dict())
    def create_model(self):
        model = nn.Sequential(
            nn.Linear(self.env.observation_space.shape[0],64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.env.action_space.n),
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
        batch_size = 64
        if len(self.memory) < batch_size:
            return
        #sample batch size
        samples =  random.sample(self.memory,batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            #target values
            target = self.target_model(torch.tensor(state))
            if done== True:
                #if state is done no future
                target[action] = reward
            else:
                #q = max of target_model (next state)  * discount + reward
                q_future = max(self.target_model(torch.tensor(new_state))).detach()
                target[action] = reward + q_future*self.gamma
            #compute loss (predicted values, target)
            #print(torch.tensor(state.reshape(1,4)).size(), target.unsqueeze(1).size())
            predicted_vals = self.model(torch.tensor(state.reshape(1,4)))
            loss = F.smooth_l1_loss(predicted_vals,target.unsqueeze(0))
            writer.add_scalar('Loss/Epoch', loss, epoch)
            optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
            optimizer.zero_grad()
            loss.backward()
            for param in self.model.parameters():
                param.grad.data.clamp_(-1,1)
            optimizer.step()

    def train_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

env = gym.make("CartPole-v0")
update_target = 10
agent = DQN(env)
avg = 0
e = 0
for i in range(1000):
    current_state =  (env.reset())
    r = 0
    z = 0
    while z < 200: 
        e+=1
        action = agent.action(current_state)
        next_state,reward,done,_ = env.step(action.item())
        r+= reward
        if done:
            agent.remember(current_state, action, reward, None,done)
            break
        agent.remember(current_state, action, reward, next_state,done)
        current_state = next_state
        agent.replay(e)
        z+=1
        agent.train_target_model()
    writer.flush()
    print(i,r)
