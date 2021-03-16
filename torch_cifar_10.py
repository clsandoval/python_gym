import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import sampler
import numpy as np
#Cifar 10 exercise


#Load datasets
N,C,W,H =  32, 3, 32, 32
transform = T.Compose([
    T.ToTensor(),
])
cifar_10 = dset.CIFAR10(
    '/datasets',
    train = True,
    download = True,
    transform = transform
)
cifar_10_test = dset.CIFAR10(
    '/datasets',
    train = False,
    download = True,
    transform = transform
)
loader_train = DataLoader(
    cifar_10,
    batch_size = 32,
    sampler=sampler.SubsetRandomSampler(range(49000)),
    drop_last =True
)
loader_val = DataLoader(
    cifar_10,
    batch_size = 32,
    sampler=sampler.SubsetRandomSampler(range(49000,50000)),
    drop_last =True
)
loader_test = DataLoader(
    cifar_10_test,
    batch_size = 32,
    drop_last =True
)


class my_model(nn.Module):
    def __init__(self,w,h,outputs):
        super(my_model,self).__init__()
    #3 Convolutional Layers, BatchNorm, RelU activation
    #TO DO: Downsample with 1x1 Conv layer for ResNet architecture 
        self.conv1 = nn.Conv2d(3,16,kernel_size = 5, stride = 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,32,kernel_size = 5, stride = 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,32,kernel_size = 5, stride = 2)
        self.bn3 = nn.BatchNorm2d(32)
        #out_size given by (size-k-2)/ + 1
        def conv2_size(size, kernel_size = 5, stride  = 2):
            return ( size - (kernel_size - 1)-1)//stride + 1
        conv_w = conv2_size(conv2_size(conv2_size(w)))
        conv_h = conv2_size(conv2_size(conv2_size(h)))
        #out_size is # of features
        self.last = nn.Linear(conv_h * conv_w * 32, 10)
    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.last(x.view(x.size()[0],-1))

def check(agent, loader):
#runs model in eval mode to predict values and check with validation set
    agent.eval()
    num_cor = 0
    num_samples = 0
    with torch.no_grad():
        samplex = []
        for step, (x,y) in enumerate(loader):
            scores = agent(x)
            _,preds = scores.max(1)
            num_cor += (preds == y).sum()
            num_samples += preds.size(0)
            samplex = x
        acc = float(num_cor)/float(num_samples)
        print(f"accuracy: {acc}")
    agent.train()
    return

agent = my_model(32,32,10)
#main training loop, e is epochs
for e in range(6):
    for step, (x,y) in enumerate(loader_train):
        y_preds =  agent(x)
        loss = F.cross_entropy(y_preds,y)
        optimizer = optim.Adam(agent.parameters(), lr = .0002)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    check(agent, loader_val)
check(agent,loader_test)

    

    
    
