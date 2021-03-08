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
    def create_model(self):
        model = nn.Sequential(
            nn.Linear(self.env.observation_space.shape[0],64),
            nn.ReLU(),
            nn.Linear(64, self.env.action_space.n),
        )