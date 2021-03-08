import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DQN:
    def __init__(self,env):
        self.env = env
        self.memory = deque(maxlen=2000)
        self.gamma = 0.85
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.tau = 0.05
        self.model = self.create_model()
        self.target_model =  self.create_model()
        self.target_model.load_state_dict(self.model.state_dict())
    def create_model(self):
        model = nn.Sequential(
            nn.Linear(self.env.observation_space.shape[0],64),
            nn.ReLU(),
            nn.Linear(64, self.env.action_space.n),
        )
        return model.to(device)
    def action(self,state):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        act = torch.argmax(self.model(state)[0])
        return act
    def remember(self,state):
        self.memory.append([state,action,reward,new_state,done])
    def replay(self, transitions):
        batch_size = 32
        if len(self.memory) < batch_size:
            return
        
        

    def train_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    def train_model(self, conc_states, targets):
        y_preds = self.model(conc_states)
        loss = nn.MSEloss()
        self.optimizer = optim.RMSprop(self.model.parameters())

        optimizer.zero_grad()
        loss.backward()

    
