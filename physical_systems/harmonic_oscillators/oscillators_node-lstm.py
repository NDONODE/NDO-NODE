import time
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn  import functional as F 

parser = argparse.ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lrdecay', type=float, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--experiment_no', type=int, default=0)
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--noise', type=float, default=0.0)
parser.add_argument('--lba', type=float, default=0.0)
args = parser.parse_args()
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


    


class ODEfunc(nn.Module):

    def __init__(self, dim, nhidden):
        super(ODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=False)
        self.fc1 = nn.Linear(2*dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, 2*dim)
        self.nfe = 0

    def forward(self, t, z):
        self.nfe += 1
        out = self.fc1(z)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out
    
    
class ODEBlock(nn.Module):

    def __init__(self, odefunc, integration_times):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_times = integration_times

    def forward(self, x):
        out = odeint(self.odefunc, x, self.integration_times, rtol=args.tol, atol=args.tol)
        return out

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
       
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
class LstmModel(nn.Module):
    def __init__(self, hidden_dim,layers_num, embedding_dim = 1):
        super(LstmModel, self).__init__()
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
            nn.Linear(32, 1)
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


if __name__ == '__main__':
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    filename = ''


    import argparse
    config = argparse.ArgumentParser()
    config.embedding_dim = 3
    config.hidden_dim = 64
    config.layers_num = 2
    config.dropout = 0
    # 1st derivative
    D1Net = LstmModel(embedding_dim=config.embedding_dim,
                hidden_dim=config.hidden_dim,
                layers_num=config.layers_num)
    D1Net.load_state_dict(torch.load('N10k-1st-100-Osin50xd50-no1x-irr-10pn-LSTM-t-f-dt-0006.pth',map_location=torch.device(device)))
    D1Net = D1Net.to(device)

    # 2nd derivative
    config.embedding_dim = 4
    config.hidden_dim = 64
    config.layers_num = 2
    config.dropout = 0
    D2Net = LstmModel(embedding_dim=config.embedding_dim,
                hidden_dim=config.hidden_dim,
                layers_num=config.layers_num)
    D2Net.load_state_dict(torch.load('N10k-2th-100-Osin50xd50-no1x-irr-10pn-LSTM-t-f-df-dt-0813.pth',map_location=torch.device(device)))
    D2Net = D2Net.to(device)

    filename = f'{args.lr}_{args.lrdecay}/lba_{args.lba:g}/'+'node-lstm/'+str(args.noise)+'/'+str(args.experiment_no)+'/'

    try:
        os.makedirs('./'+filename)
    except FileExistsError:
        pass


    data_dim = 1
    dim = data_dim
    #dim does not equal data_dim for ANODEs where they are augmented with extra zeros
 
    #download data
    z0 = torch.tensor(np.load('data/z0.npy')).float().to(device)
    z_in = torch.tensor(np.load('data/z.npy')).float().to(device)
    samp_ts = torch.tensor(np.load('data/samp_ts.npy')).float().to(device)

    cutoff = 30
    x0 = z0[:cutoff]
    v0 = z0[cutoff:]
    z0 = torch.cat((x0, v0), dim=1).to(device)
    
    z = torch.empty((int(len(samp_ts)), cutoff, 2)).to(device)
    
    for i in range(int(len(samp_ts))):
        xi = z_in[i][:cutoff]
        vi = z_in[i][cutoff:]
        z[i] = torch.cat((xi, vi), dim=1)
    # noise
    z = z + (torch.rand_like(z)-0.5)*2*args.noise

    tz0 = torch.tensor(np.load('data/tz0.npy')).float().to(device)
    tz_in = torch.tensor(np.load('data/tz.npy')).float().to(device)
    tsamp_ts = torch.tensor(np.load('data/tsamp_ts.npy')).float().to(device)
    thalf = tsamp_ts.shape[0] // 2
    cutoff = 30
    tx0 = tz0[:cutoff]
    tv0 = tz0[cutoff:]
    tz0 = torch.cat((tx0, tv0), dim=1).to(device)
    
    tz = torch.empty((int(len(tsamp_ts)), cutoff, 2)).to(device)
    for i in range(int(len(tsamp_ts))):
        txi = tz_in[i][:cutoff]
        tvi = tz_in[i][cutoff:]
        tz[i] = torch.cat((txi, tvi), dim=1)

    # model
    nhidden = 20
    func = ODEfunc(dim, nhidden).to(device)


    aL = len(samp_ts)
    dT = torch.tensor([0] * aL).float().to(device)
    for i in range(1,aL):
        dT[i] = samp_ts[i]-samp_ts[i-1]

    # D1 + D2
    tN = samp_ts.max() # normalize t to [0,1]
    tM = 1             # normalize f(t)
    dT_n = dT / tN
    ts_n = samp_ts / tN
    
    #1st
    input1 = torch.cat((dT_n.reshape(-1,1,1).repeat(1,cutoff,1),tM*z[:,:,:1],ts_n.reshape(-1,1,1).repeat(1,cutoff,1)),dim = -1).transpose(1,0)
    with torch.no_grad():
        output1 = D1Net(input1)[0] / tN / tM
    df = output1.transpose(1,0)
    
    #2nd
    tSF = 20
    input2 = torch.cat((dT_n.reshape(-1,1,1).repeat(1,cutoff,1),tM*z[:,:,:1]/tSF,tM*df,ts_n.reshape(-1,1,1).repeat(1,cutoff,1)),dim = -1).transpose(1,0)
    with torch.no_grad():
        output2 = D2Net(input2)[0] / tN / tM
    ddf = output2.transpose(1,0)


    df = torch.cat((df,ddf),dim=-1)

    feature_layers = [ODEBlock(func, samp_ts)]
    model = nn.Sequential(*feature_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.MSELoss()
    scheduler =  torch.optim.lr_scheduler.ExponentialLR(optimizer,args.lrdecay)

    itr_arr = np.empty(args.niters)
    loss_arr = np.empty(args.niters)
    t_in_loss_arr = np.empty(args.niters)
    t_ex_loss_arr = np.empty(args.niters)
    nfe_arr = np.empty(args.niters)
    time_arr = np.empty(args.niters)
    min_loss = 9999999
    best_num = 0
    # training
    start_time = time.time() 
    for itr in range(1, args.niters+1):
        feature_layers[0].nfe = 0
        iter_start_time = time.time()
        optimizer.zero_grad()
        #forward in time and solve ode
        pred_z = model(z0).to(device)
        # compute loss
        a = loss_func(pred_z[:,:,0], z[:,:,0])
        b =  F.mse_loss(df, func(samp_ts,z))
        loss = a + args.lba * b
        loss.backward()
        optimizer.step()
        scheduler.step()
        iter_end_time = time.time()

        with torch.no_grad():
            tfeature_layers = [ODEBlock(func, tsamp_ts)]
            tmodel = nn.Sequential(*tfeature_layers).to(device)
            tpred_z = tmodel(tz0).to(device)
            t_in_loss = F.mse_loss(tpred_z[:thalf], tz[:thalf])
            t_ex_loss = F.mse_loss(tpred_z[thalf:], tz[thalf:])

        if (loss < min_loss):
            min_loss = loss
            best_num = itr
        t_in_loss_arr[itr-1] = t_in_loss
        t_ex_loss_arr[itr-1] = t_ex_loss
        itr_arr[itr-1] = itr
        loss_arr[itr-1] = a
        nfe_arr[itr-1] = feature_layers[0].nfe
        time_arr[itr-1] = iter_end_time-iter_start_time
        
        print('Iter: {}, running MSE: {:.4f}, In Test MSE: {:.4f} Ex Test MSE: {:.4f}'.format(itr, loss, t_in_loss, t_ex_loss))
            

    end_time = time.time()
    print('\n')
    print('Training complete after {} iterations.'.format(itr))
    loss = loss.detach().cpu().numpy()
    print('Train MSE = ' +str(loss))
    print('Best Test MSE = ' +str(min_loss) + str(best_num))
    print('NFE = ' +str(feature_layers[0].nfe))
    print('Total time = '+str(end_time-start_time))
    print('No. parameters = '+str(count_parameters(model)))
    
    np.save(filename+'t_in_loss_arr.npy', t_in_loss_arr)
    np.save(filename+'t_ex_loss_arr.npy', t_ex_loss_arr)
    np.save(filename+'best_iter.npy', best_num)
    np.save(filename+'itr_arr.npy', itr_arr)
    np.save(filename+'nfe_arr.npy', nfe_arr)
    np.save(filename+'loss_arr.npy', loss_arr)
    np.save(filename+'time_arr.npy', time_arr)
    torch.save(model, filename+'model.pth')

    plt.clf()
    plt.plot(samp_ts.cpu().numpy(),z[:,0,0].cpu().numpy(),'r-.')
    plt.plot(samp_ts.cpu().numpy(),pred_z[:,0,0].detach().cpu().numpy(),'b:',label = 'Predtion') 
    plt.plot(tsamp_ts.cpu().numpy(),tz[:,0,0].cpu().numpy(), label = 'Ground Truth')
    plt.plot(tsamp_ts.cpu().numpy(),tpred_z[:,0,0].cpu().numpy(),'b:') 
    plt.legend()
    plt.savefig(filename + 'best.png')