import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import sampler

N,C,W,H =  32, 3, 32, 32
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5), (0.2))
])
mnist_data = dset.CIFAR10(
    '/datasets',
    train = True,
    download = True,
    transform = transform
)
mnist_data_test = dset.CIFAR10(
    '/datasets',
    train = False,
    download = True,
    transform = transform
)
loader_train = DataLoader(
    mnist_data,
    batch_size = 32,
    sampler=sampler.SubsetRandomSampler(range(19000)),
    drop_last =True
)
loader_test = DataLoader(
    mnist_data_test,
    batch_size = 32,
    drop_last =True
)
model = nn.Sequential(
    nn.Linear(C*W*H,256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Linear(256,128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Linear(128,10)
)
for e in range(50):
    for step, (x,y) in enumerate(loader_train):
        y_preds =  model(x.view(N,-1))
        loss = F.cross_entropy(y_preds,y)
        optimizer = optim.Adam(model.parameters(), lr = .0002)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    num_cor = 0
    num_samples = 0
    with torch.no_grad():
        samplex = []
        for step, (x,y) in enumerate(loader_test):
            scores = model(x.view(N,-1))
            _,preds = scores.max(1)
            num_cor += (preds == y).sum()
            num_samples += preds.size(0)
            samplex = x
        acc = float(num_cor)/float(num_samples)
        print(f"accuracy: {acc}")
        print(y)

    
    
