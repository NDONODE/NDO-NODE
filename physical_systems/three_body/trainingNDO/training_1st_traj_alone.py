import torch 
#import torchvision
import torch.optim as optim
import torch.nn as nn
import numpy as np
import sympy as sp
#import torchvision.transforms as transforms
import torch.nn.functional as F
#import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import random
import wandb
import os
import math
os.environ["WANDB_API_KEY"] = "c46e7ccd6731858fa3ca9629d9c1863f212af2c3"
# 如果能用GPU则选用GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class Model(nn.Module):
    def __init__(self, hidden_dim,layers_num, embedding_dim = 1, out_dim = 1):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers_num = layers_num
        #self.embeddings = nn.Embedding.from_pretrained(torch.from_numpy(embed_weight),freeze=False) #加载预训练word2vec
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
        #x = x.reshape(-1,256,2)
        #embeds = self.embeddings(input)  # [batch, seq_len] => [batch, seq_len, embed_dim]
        if hidden is None:
            batch_size, seq_len = x.size()[0:2]
            h_0 = x.data.new(self.layers_num*2, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = x.data.new(self.layers_num*2, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        #x = self.dp(x)
        #print(x)
        output, hidden = self.lstm(x, (h_0, c_0)) 
        output = self.out(output)
        #output = output.reshape(batch_size * seq_len, -1)
        return output, hidden

import copy
global minloss
minloss = 1e9
global best_model

def train(model, device, loader, optimizer, epoch):
    model.train()
    criterion = nn.MSELoss(reduction = 'sum')
    enum = math.ceil(len(loader.sampler)/config.batch_size)
    enum = enum*epoch
    for i, (inputs,labels) in enumerate(loader):
        inputs, labels = inputs.to(device,non_blocking=True), labels.to(device,non_blocking=True)
        optimizer.zero_grad()
        output,hidden = model(inputs)
        #print(output.shape,labels.shape)
        #output = output[:,:,0]   # 只取最后输出作为判断
        loss = criterion(output, labels)  / loader.batch_size
        loss.backward()
        optimizer.step()
        global minloss
        global best_model
        if ((enum+i) % 1000) == 0:
            wandb.log({"Train Loss": loss.item()}  )
            print('[%d, %5d] loss: %.3f' %
                   (epoch + 1, i + enum, loss.item() ))
            #test(model,device,testloader,epoch)
            if (minloss >  loss.item()):
                print("New best")
                best_model = copy.deepcopy(model)
                minloss = loss.item()
            #test(model,device,testloader,epoch)

def test(model, device, loader, epoch):
    model.eval()
    criterion = nn.MSELoss(reduction='sum')
    avg_loss = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device,non_blocking=True), labels.to(device,non_blocking=True)
            output,hidden = model(inputs)
            #output = output[:,:,0]
            loss = criterion(output, labels)
            #print(output,labels)
            avg_loss += loss.item()
    total = len(loader.dataset)
    avg_loss /= total
    global minloss
    global best_model
    if (minloss >   avg_loss):
        print("New best")
        best_model = copy.deepcopy(model)
        minloss = avg_loss
    print('Avg Loss : %.3f \n' % (avg_loss) )
    wandb.log({"Test Loss": avg_loss})
    model.train()

# three body traj
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

t = np.linspace(0,1,1000)
sol = np.load('sol.npy')
Tnum = len(sol)
M_LEN = 100
solm = np.zeros([Tnum, M_LEN,18])
tm = np.zeros([Tnum,M_LEN])
dtm = np.zeros([Tnum,M_LEN])
d2t = np.zeros([Tnum,M_LEN,9])
d2t0 = np.load('d2t.npy')
dff = np.zeros([len(t),9])
for i in range(len(sol)):
    #print(i)
    mask =  random.sample(list(range(0,1000-1)),M_LEN)
    mask.sort()
    tm[i] = t[mask]
    for j in range(1, M_LEN):
        dtm[i][j] = tm[i][j] - tm[i][j-1]
    
#     for j in range(1,len(t)-1):
#         dff[j] = (sol[i,j+1,9:]-sol[i,j-1,9:])/(t[j+1]-t[j-1])
#     dff[0] = (sol[i,1,9:]-sol[i,0,9:])/(t[1]-t[0])
#     dff[len(t)-1] = (sol[i,len(t)-1,9:]-sol[i,len(t)-2,9:])/(t[len(t)-1]-t[len(t)-2]) 
    d2t[i] = copy.deepcopy(d2t0[i][mask])#[mask])
    solm[i] = sol[i,mask,:]

tm = np.array(tm).reshape(-1,M_LEN,1)
dtm = np.array(dtm).reshape(-1,M_LEN,1)
solm = np.array(solm)#.reshape(-1,M_LEN,1)

# traj 1st alone

Nf = solm[:,:,:9]
Ndf = solm[:,:,9:]
Nddf = torch.Tensor(d2t)
r1 = torch.Tensor(Nf[...,:3])
r2 = torch.Tensor(Nf[...,3:6])
r3 = torch.Tensor(Nf[...,6:9])

innd1 = torch.zeros(Tnum,M_LEN,9)
innd1[...,:9] = torch.cat((r1,r2,r3),dim = -1)

innd2 = torch.zeros(Tnum,M_LEN,9)
innd2[...,:9] = torch.cat((r1,r2,r3),dim = -1)

Nf = innd1


Scale_factor = 20

tm=np.tile(tm,(3,1,1))
dtm=np.tile(dtm,(3,1,1))
Nf = np.concatenate((Nf[:,:,:3],Nf[:,:,3:6],Nf[:,:,6:9]),axis=0)
Ndf = np.concatenate((Ndf[:,:,:3],Ndf[:,:,3:6],Ndf[:,:,6:9]),axis=0)
Nddf = np.concatenate((Nddf[:,:,:3],Nddf[:,:,3:6],Nddf[:,:,6:9]),axis=0)



Nf = np.concatenate((dtm,Nf,tm),axis = 2)
Nf = torch.Tensor(Nf)
Ndf = torch.Tensor(Ndf)
mask_nan = (torch.sum(torch.isnan(Nf),axis=[1,2])==0)
Nf = Nf[mask_nan]
Ndf = Ndf[mask_nan]
Nddf = Nddf[mask_nan] / Scale_factor
print(Nf.shape)

traindata = TensorDataset(torch.Tensor(Nf),torch.Tensor(Ndf))
# x_train, x_test, y_train, y_test = train_test_split(Nf, Ndf, test_size = 0.2)#,shuffle=True)
# traindata = TensorDataset(torch.Tensor(x_train),torch.Tensor(y_train))
# testdata = TensorDataset(torch.Tensor(x_test),torch.Tensor(y_test))



bs = 64
run = wandb.init(name = f"LSTM-1nd-dt-f-t-5-irr-ThreeBody-traj-alone-{bs}",project="diff",reinit=True)#,resume = true)

config = wandb.config          # Initialize config
config.batch_size = bs           # input batch size for training (default: 64)
config.test_batch_size = 1000    # input batch size for testing (default: 1000)
#config.maxiter = 12000
config.maxiter = 100000
config.epochs =  math.ceil(config.maxiter /  math.ceil(len(traindata)/config.batch_size))

config.embedding_dim = 3 + 2
config.out_dim = 3
config.hidden_dim = 64
config.layers_num = 2
config.dropout = 0
config.T_max = config.epochs
config.lr = 0.003 # learning rate
#config.momentum = 0.9
#config.weight_decay = 0.0001
#config.log_interval = 10 
 
trainloader = DataLoader(traindata, batch_size=config.batch_size,
                shuffle=True,drop_last=True,pin_memory=True, num_workers=2)
# testloader = DataLoader(testdata, batch_size=config.test_batch_size,
#                 shuffle=False, drop_last=True,pin_memory=True, num_workers=2)

 
net = Model(embedding_dim=config.embedding_dim,
              hidden_dim=config.hidden_dim,
              layers_num=config.layers_num,
              out_dim=config.out_dim )
model = net.to(device)
wandb.watch(model, log="all")

optimizer = optim.Adam(model.parameters(), lr=config.lr)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.T_max)
#torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,5,10,20], gamma=0.5)
    
for epoch in range(config.epochs):
    train(model,device,trainloader,optimizer, epoch)
    scheduler.step()

torch.save(best_model.state_dict(), "LSTM-1st-dt-f-t-5-irr-traj-alone.pth")

run.join()
