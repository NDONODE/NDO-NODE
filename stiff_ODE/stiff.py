import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn  import functional as F 
import matplotlib.pyplot as plt
import copy
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=120) #default=1000)
parser.add_argument('--batch_time', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--ntest', type=int, default=10)
parser.add_argument('--n_units', type=int, default=500)
parser.add_argument('--min_length', type=float, default=0.001)
parser.add_argument('--normal_std', type=float, default=0.01)
parser.add_argument('--stiffness_ratio', type=float, default=1000.0)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--version', type=str, choices=['standard','steer','normal'], default='steer')
parser.add_argument('--lstm', action='store_true')
parser.add_argument('--L2', action='store_true')
parser.add_argument('--lba', type=float, default=0.00)
parser.add_argument('--experiment_no', type=int, default=0)
args = parser.parse_args()
torch.manual_seed(6)
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
    from torchdiffeq import odeint_adjoint_stochastic_end_v3 as odeint_stochastic_end_v3
    from torchdiffeq import odeint_adjoint_stochastic_end_normal as odeint_stochastic_end_normal
else:
    from torchdiffeq import odeint_stochastic_end_v3
    from torchdiffeq import odeint

if args.version == 'steer':
    if args.lstm:
        filename = f'res/lba_{args.lba:g}/'+'node-steer-lstm/'+str(args.experiment_no)+'/'
    else:
        filename = f'res/'+'node-steer/'+str(args.experiment_no)+'/'
elif args.version == 'standard':
    if args.L2:
        filename = f'res/lba_{args.lba:g}/'+'node-o-L2/'+str(args.experiment_no)+'/'
    elif args.lstm:
        filename = f'res/lba_{args.lba:g}/'+'node-o-lstm/'+str(args.experiment_no)+'/'
    else:
        filename = f'res/'+'node-o/'+str(args.experiment_no)+'/'
try:
    os.makedirs('./'+filename)
except FileExistsError:
    pass
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
device = 'cpu'
true_y0 = torch.tensor([0.])
t = torch.linspace(0., 15., args.data_size)
test_t = torch.linspace(0., 25., args.data_size)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]])


class Lambda(nn.Module):

    def forward(self, t, y):
        t = t.unsqueeze(0)
        # equation = -1*y*args.stiffness_ratio + 3*args.stiffness_ratio - 2*args.stiffness_ratio * torch.exp(-1*t)
        equation = -1*y*args.stiffness_ratio + 3*args.stiffness_ratio - 2*args.stiffness_ratio * torch.exp(-1*t)# - 2*args.stiffness_ratio * torch.exp(-10000*t)
        # equation = -1000*y + 3000 - 2000 * torch.exp(-t) + 1000 * torch.sin(t)
        return equation


with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')
    true_y_test = odeint(Lambda(), true_y0, test_t, method='dopri5')


def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs(filename+'png')
    import matplotlib.pyplot as plt


def visualize(true_y, pred_y, odefunc, test_t, itr):
    if args.viz:

        plt.clf()
        plt.xlabel('t')
        plt.ylabel('y')
        plt.plot(test_t.numpy(), true_y.numpy()[:, 0], 'g-', label='True')
        plt.plot(test_t.numpy(), pred_y.numpy()[:, 0], 'b--' , label='Predicted' )
        plt.ylim((-1, 25))
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(filename+'png/{:04d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, args.n_units),
            nn.Tanh(),
            nn.Linear(args.n_units, args.n_units),
            nn.Tanh(),
            nn.Linear(args.n_units, 1),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        t=t.unsqueeze(0)
        t = t.view(-1,1)
        y = y.view(y.size(0),1)
        t = t.expand_as(y)
        equation = torch.cat([t,y],1)
        result = self.net(equation)

        if y.size(0)==1:
            result = result.squeeze()
        return result

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
        tM = 0.5 / true_y.max(0)[0].unsqueeze(1).repeat(1,aL,1)
        dT_n = dT / tN
        ts_n = t / tN
        dT_n = dT_n.view(-1,aL,1)
        ts_n = ts_n.view(-1,aL,1)
        fss = true_y.view(1,aL,1)
        

        inputss = torch.cat((dT_n,fss * tM ,ts_n),dim = -1)
        with torch.no_grad():
            output1 = D1Net(inputss)[0] / tN / tM
        df = output1.transpose(1,0).transpose(2,1)


    ii = 0

    func = ODEFunc()
    optimizer = optim.RMSprop(func.parameters(), lr=1e-4)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    itr_arr = np.empty(args.niters)
    loss_arr = np.empty(args.niters)
    t_in_loss_arr = np.empty(args.niters)
    t_ex_loss_arr = np.empty(args.niters)
    min_loss = 9999999
    best_num = 0

    if  (args.L2):
        df = torch.zeros_like(func(t,true_y).squeeze())

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        if args.version=='standard':
            pred_y = odeint(func, batch_y0, batch_t)
        elif args.version=='steer':
            pred_y = odeint_stochastic_end_v3(func, batch_y0, batch_t,min_length=args.min_length,mode='train')
        elif args.version=='normal':
            pred_y = odeint_stochastic_end_normal(func, batch_y0, batch_t,std=args.normal_std,mode='train')
        a = torch.mean(torch.abs(pred_y - batch_y))
        if (args.lstm or args.L2):
            b = F.mse_loss(df.squeeze(), func(t,true_y).squeeze())
            loss = a + args.lba * b
        else:
            loss = a
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                torch.save(func, filename+f'model{ii}.pth')
                itr_arr[ii] = itr
                t_in_loss_arr[ii] = loss

                pred_y = odeint(func, true_y0, test_t)
                loss = torch.mean(torch.abs(pred_y - true_y_test))
                t_ex_loss_arr[ii] = loss
                print('Iter {:04d} | In Loss : {:.6f} Ex Loss {:.6f}'.format(itr, t_in_loss_arr[ii].item(), t_ex_loss_arr[ii].item()))
                visualize(true_y_test, pred_y, func, test_t,  ii )
                if (loss < min_loss):
                    best_model = copy.deepcopy(func)
                    min_loss = loss
                    best_num = itr
                    print(min_loss)
                ii += 1

        end = time.time()

    np.save(filename+'t_in_loss_arr.npy', t_in_loss_arr)
    np.save(filename+'t_ex_loss_arr.npy', t_ex_loss_arr)
    np.save(filename+'best_iter.npy', best_num)
    np.save(filename+'itr_arr.npy', itr_arr)
    np.save(filename+'loss_arr.npy', loss_arr)
    torch.save(best_model, filename+'best_model.pth')