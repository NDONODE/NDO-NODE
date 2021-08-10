import torch 
import torch.optim as optim
import torch.nn as nn
import numpy as np
import sympy as sp
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import random
import os
import math
import argparse
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, hidden_dim,layers_num, embedding_dim = 1,out_dim = 1):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers_num = layers_num
        self.dp = nn.Dropout(config.dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=layers_num,
                            batch_first=True,dropout=config.dropout,bidirectional=True)
        self.out =  nn.Sequential(
            nn.Linear(self.hidden_dim*2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Dropout(config.dropout),
            nn.ReLU(),
            nn.Linear(32, out_dim)
            )
    def forward(self, x, hidden=None):
        if hidden is None:
            batch_size, seq_len = x.size()[0:2]
            h_0 = x.data.new(self.layers_num*2, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = x.data.new(self.layers_num*2, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        output, hidden = self.lstm(x, (h_0, c_0)) 
        output = self.out(output)
        return output, hidden

import copy
global minloss
minloss = 1e9
global best_model

def train(model, device, loader, optimizer, epoch):
    model.train()
    criterion = nn.MSELoss()
    enum = math.ceil(len(loader.sampler)/config.batch_size)
    enum = enum*epoch
    for i, (inputs,labels) in enumerate(loader):
        inputs, labels = inputs.to(device,non_blocking=True), labels.to(device,non_blocking=True)
        optimizer.zero_grad()
        output,hidden = model(inputs)
        output = output[:,:,0]
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        global minloss
        global best_model
        if ((enum+i) % 1000) == 0:
            print('[%d, %5d] loss: %.3f' %
                   (epoch + 1, i + enum, loss.item() ))
            if (minloss >  loss.item()):
                print("New best")
                best_model = copy.deepcopy(model)
                minloss = loss.item()

# pre-train 2nd order NDO
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
datatmp = np.load('N10k-2nd-100-Osin100xd100-no1x-irr-10pn.npz')
Nf = datatmp['Nf']
Nddf = Nf[:,3,:]
Nf = Nf[:,:3,:]
Nf = np.concatenate((Nf,Nf[:,[0],:]),axis=1)

for i in range(len(Nf)):
    L = len(Nf[i][0])
    for j in range(L-1,0,-1):
        Nf[i][0][j] -= Nf[i][0][j-1]
    Nf[i][0][0] = 0

Nf = np.transpose(Nf,(0,2,1))
traindata = TensorDataset(torch.Tensor(Nf),torch.Tensor(Nddf)) 

parser = argparse.ArgumentParser('Pretrain NDO')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--maxiter', type=int, default=100000)
parser.add_argument('--epochs', type=int, default=641)
parser.add_argument('--embedding_dim', type=int, default=3)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--layers_num', type=int, default=2)
parser.add_argument('--dropout', type=int, default=0)
parser.add_argument('--T_max', type=int, default=641)
parser.add_argument('--lr', type=float, default=0.003)
config = parser.parse_args()

config.epochs =  math.ceil(config.maxiter /  math.ceil(len(traindata)/config.batch_size))
config.T_max = config.epochs
 
trainloader = DataLoader(traindata, batch_size=config.batch_size,
                shuffle=True,drop_last=True,pin_memory=True, num_workers=2)

net = Model(embedding_dim=config.embedding_dim,
              hidden_dim=config.hidden_dim,
              layers_num=config.layers_num)

model = net.to(device)

optimizer = optim.Adam(model.parameters(), lr=config.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.T_max)

for epoch in range(config.epochs):
    train(model,device,trainloader,optimizer, epoch)
    scheduler.step()
    
torch.save(best_model.state_dict(), "N10k-2nd-100-Osin100xd100-no1x-irr-10pn-LSTM-dt-f-t.pth")
