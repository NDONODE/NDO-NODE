import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn  import functional as F 
import matplotlib.pyplot as plt
import copy
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=100)
parser.add_argument('--test_size', type=int, default=500)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--lstm', action='store_true')
parser.add_argument('--noise', type=float, default=0.0)
parser.add_argument('--lba', type=float, default=0.08)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lrdecay', type=float, default=0.995)
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--experiment_no', type=int, default=0)
args = parser.parse_args()
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint
if args.lstm:
    filename = f'{args.lr}_{args.lrdecay}/lba_{args.lba:.g}/'+'node-lstm/'+str(args.noise)+'/'+str(args.experiment_no)+'/'
else:
    filename = f'{args.lr}_{args.lrdecay}/'+'node-o/'+str(args.noise)+'/'+str(args.experiment_no)+'/'
try:
    os.makedirs('./'+filename)
except FileExistsError:
    pass
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
device = 'cpu'
true_y0 = torch.tensor([[2., 0.]]).to(device)
t = torch.linspace(0., 5., args.data_size).to(device)
test_t = torch.linspace(0., 10., args.test_size*2).to(device)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)


class Lambda(nn.Module):

    def forward(self, t, y):
        return torch.mm(y, true_A)


with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')
    t_true_y = odeint(Lambda(), true_y0, test_t, method='dopri5')
    
# noise
true_y = true_y + (torch.rand_like(true_y)-0.5)*2*args.noise 

def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return s, batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs(filename+'png')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, n_true_y, pred_y, t, n_t, odefunc, itr):

    if args.viz:
        la = len(t) // 2
        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--')
        # ax_traj.vlines(5,-2,2,color='c', linestyle=':',lw =1)
        # ax_traj.plot(n_t.cpu().numpy(), n_true_y.cpu().numpy()[:, 0, 0], n_t.cpu().numpy(), n_true_y.cpu().numpy()[:, 0, 1], 'g.')
        #ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--')

        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.set_ylim(-2, 2)
        # ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0][la:], true_y.cpu().numpy()[:, 0, 1][la:],  '-', color='orange',alpha = 0.5)
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0][:la], true_y.cpu().numpy()[:, 0, 1][:la], 'g-',alpha = 0.5)
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
        # ax_phase.plot(n_true_y.cpu().numpy()[:, 0, 0], n_true_y.cpu().numpy()[:, 0, 1], 'g.')

        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)

        ax_vecfield.cla()
        # ax_vecfield.plot(t.cpu().numpy(),true_y[:,0].cpu().numpy())
        # print(t.shape,true_y.shape,odefunc(0,true_y).shape,odefunc(0,true_y)[:,0].shape)
        ax_vecfield.plot(t.cpu().numpy(),odefunc(0,true_y)[:,0].cpu().numpy(),'b:')
        if (args.lstm == True):
            ax_vecfield.plot(torch.linspace(0., 5., args.data_size).cpu().numpy(),df[:,0].cpu().numpy(),label = "lstm Grad")
        ax_vecfield.legend()
        ax_vecfield.set_ylim(-5, 5)
        # ax_vecfield.set_title('Learned Vector Field')
        # ax_vecfield.set_xlabel('x')
        # ax_vecfield.set_ylabel('y')

        # y, x = np.mgrid[-2:2:21j, -2:2:21j]
        # dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
        # mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        # dydt = (dydt / mag)
        # dydt = dydt.reshape(21, 21, 2)

        # ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        # ax_vecfield.set_xlim(-2, 2)
        # ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig(filename+'png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 20),
            nn.ELU(),
            nn.Linear(20, 20),
            nn.ELU(),
            nn.Linear(20, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)

class LstmModel(nn.Module):
    def __init__(self, hidden_dim,layers_num, embedding_dim = 1):
        super(LstmModel, self).__init__()
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
            nn.Linear(32, 1)
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

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':

    func = ODEFunc().to(device)
    
    if (args.lstm == True):
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
        D1Net.load_state_dict(torch.load('N10k-1st-100-Osin50xd50-no1x-irr-10pn-LSTM-t-f-dt-0006.pth',map_location=device))

        D1Net = D1Net.to(device)

        aL = len(t)
        dT = torch.tensor([0] * aL).float().to(device)
        for i in range(1,aL):
            dT[i] = t[i]-t[i-1]
        tN = t.max()
        tM = 0.5 / true_y.max(0)[0].transpose(1,0).unsqueeze(1).repeat(1,aL,1)
        dT_n = dT / tN
        ts_n = t / tN

        dT_n = dT_n.view(-1,aL,1).repeat(2,1,1)
        ts_n = ts_n.view(-1,aL,1).repeat(2,1,1)
        fss = true_y.transpose(1,0).transpose(0,2)
        inputss = torch.cat((dT_n,fss * tM ,ts_n),dim = -1)
        with torch.no_grad():
            output1 = D1Net(inputss)[0] / tN / tM
        df = output1.transpose(1,0).transpose(2,1)

    optimizer = optim.Adam(func.parameters(), lr=args.lr)
    scheduler =  torch.optim.lr_scheduler.ExponentialLR(optimizer,args.lrdecay)
    end = time.time()

    itr_arr = np.empty(args.niters)
    loss_arr = np.empty(args.niters)
    t_in_loss_arr = np.empty(args.niters)
    t_ex_loss_arr = np.empty(args.niters)
    min_loss = 9999999
    best_num = 0
    ii = 0
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        mask, batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0, batch_t).to(device)
        if (args.lstm == True):
            a = F.mse_loss(pred_y , batch_y) 
            b = F.mse_loss(df, func(t,true_y))
            loss = a + args.lba * b 
        else:
            loss = F.mse_loss(pred_y , batch_y) 

        loss.backward()
        optimizer.step()
        scheduler.step()

        if itr % args.test_freq == 0:
            with torch.no_grad():
                if (loss < min_loss):
                    best_model = copy.deepcopy(func)
                    min_loss = loss
                    best_num = itr
                
                itr_arr[ii] = itr
                loss_arr[ii] = loss
                t_pred_y = odeint(func, true_y0, test_t)
                visualize(t_true_y, true_y, t_pred_y, test_t, t, func, ii)
                t_in_loss = F.mse_loss(t_pred_y[:args.test_size], t_true_y[:args.test_size])
                t_ex_loss = F.mse_loss(t_pred_y[args.test_size:], t_true_y[args.test_size:])
                print('Iter {:04d} | In Loss : {:.6f} Ex Loss {:.6f} Total Loss {:.6f}'.format(itr, t_in_loss.item(), t_ex_loss.item(), (t_in_loss + t_ex_loss)/2))
                t_in_loss_arr[ii] = t_in_loss
                t_ex_loss_arr[ii] = t_ex_loss

                ii += 1

        end = time.time()

    np.save(filename+'t_in_loss_arr.npy', t_in_loss_arr)
    np.save(filename+'t_ex_loss_arr.npy', t_ex_loss_arr)
    np.save(filename+'best_iter.npy', best_num)
    np.save(filename+'itr_arr.npy', itr_arr)
    np.save(filename+'loss_arr.npy', loss_arr)
    torch.save(best_model, filename+'model.pth')