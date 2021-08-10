import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc as rc
import matplotlib.axes
import seaborn as sns
import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--npoints', type=int, default=100)
parser.add_argument('--noise', type=float, default=0.0)
parser.add_argument('--scale', type=float, default=1)
parser.add_argument('--experiment_no', type=int, default=3)
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--prefix', type = str, default='')
parser.add_argument('--lba', type=float, default=0.0)
args = parser.parse_args()


def draw(method, noise, scale, label, color, prefix = ''):

    if (prefix == ''):
        prefix = args.prefix 
    filename = prefix + f'/{noise:.3f}/'
    global iters
    try:
        iters = np.arange(len(pd.read_csv(filename + str(1)+ '/three_body.csv').iloc[:,1]))
    except:
        return
    pretrain = False
    time_pre = np.empty(3)
    best_iter = np.empty(3)
    best_tloss = np.empty(3)
    times = np.empty((3,len(iters)))
    testloss = np.empty((3,len(iters)))
    t_testloss = np.empty((3,len(iters)))
    for i in range(3):
        try :
            time_pre[i] = np.load(filename + str(i+1) +'/time_pre.npy')
            pretrain = True
        except:
            pass
        training = pd.read_csv(filename + str(i+1)+ '/three_body.csv')
        testloss[i] = training.iloc[:,1]
        best_iter = np.argmin(testloss[i][:args.niters])

        if (plot_type =='train'):
            df = pd.read_csv(filename + str(i+1)+ '/three_body.csv')
            testloss[i] = df.iloc[:,1]
        elif (plot_type == 'in'):
            df = pd.read_csv(filename + str(i+1)+ '/t_loss_intrp_history.csv')
            testloss[i] = df.iloc[:,1]
        elif (plot_type == 'ex'):
            df = pd.read_csv(filename + str(i+1)+ '/t_loss_intrp_history.csv')
            df2 = pd.read_csv(filename + str(i+1)+ '/t_three_body.csv')
            testloss[i] = (df2.iloc[:,1]*2 - df.iloc[:,1])

        best_tloss[i] = testloss[i][best_iter]
        

    iters = df.iloc[:,0]
    lstmnode_1 = testloss[0]
    lstmnode_2 = testloss[1]
    lstmnode_3 = testloss[2]

    lstmnode_loss = np.empty((len(lstmnode_1), 3))
    for i in range(len(lstmnode_1)):
        lstmnode_loss[i][0] = lstmnode_1[i]
        lstmnode_loss[i][1] = lstmnode_2[i]
        lstmnode_loss[i][2] = lstmnode_3[i]
        
    lstmnode_mean = np.empty(len(lstmnode_1))
    for i in range(len(lstmnode_1)):
        lstmnode_mean[i] = np.mean(lstmnode_loss[i])

    lstmnode_std = np.empty(len(lstmnode_1))
    for i in range(len(lstmnode_1)):
        lstmnode_std[i] = np.std(lstmnode_loss[i])

    lstmnode_p = lstmnode_mean+lstmnode_std
    lstmnode_m = lstmnode_mean-lstmnode_std
    testlossmean = np.mean(best_tloss) * scale
    testlossstd = np.std(best_tloss)* scale

    print(plot_type, method, noise, args.lba, testlossmean, testlossstd)

    plt.plot(iters[:args.niters], lstmnode_mean[:args.niters], color=color, label=f'{label:20}, Best :{testlossmean:.3f}$\pm${testlossstd:.3f}')
    #ax.fill_between(x=iters[:args.niters], y1=lstmnode_p[:args.niters], y2=lstmnode_m[:args.niters], alpha=0.2, color=color)

plot_type = 'train' 
sns.set_style('darkgrid')
rc('font', family='serif')
fig, ax = plt.subplots(dpi=600)
draw(method = 'NDO-NODE',
     noise = args.noise,
     scale = args.scale,
     label = 'NDO-NODE',
     color = '#AA8844') 

draw(method = 'vanilla NODE',
     noise = args.noise,
     scale = args.scale,
     label = 'NODE',
     color = '#DDAA33',
     prefix = './100/vanilla/') 

rc('font', family='serif')
plt.yscale('log')
plt.legend(fontsize=12)
plt.title('Three Body Training MSE', fontsize=20)
plt.xlabel('Iterations', fontsize=16)
plt.ylabel('MSE', fontsize=16)
plt.savefig(args.prefix + f'training_loss-{str(args.noise)}.png', bbox_inches='tight')

plot_type = 'in' 
plt.clf()
sns.set_style('darkgrid')
rc('font', family='serif')
fig, ax = plt.subplots(dpi=600)
draw(method = 'NDO-NODE',
     noise = args.noise,
     scale = args.scale,
     label = 'NDO-NODE',
     color = '#AA8844') 

draw(method = 'vanilla NODE',
     noise = args.noise,
     scale = args.scale,
     label = 'NODE',
     color = '#DDAA33',
     prefix = './100/vanilla/') 

rc('font', family='serif')
plt.yscale('log')
plt.legend(fontsize=12)
plt.title('Three Body In. MSE', fontsize=20)
plt.xlabel('Iterations', fontsize=16)
plt.ylabel('MSE', fontsize=16)
plt.savefig(args.prefix + f'test_in_loss-{str(args.noise)}.png', bbox_inches='tight')


plot_type = 'ex' 
plt.clf()
sns.set_style('darkgrid')
rc('font', family='serif')
fig, ax = plt.subplots(dpi=600)
draw(method = 'NDO-NODE',
     noise = args.noise,
     scale = args.scale,
     label = 'NDO-NODE',
     color = '#AA8844') 

draw(method = 'vanilla NODE',
     noise = args.noise,
     scale = args.scale,
     label = 'NODE',
     color = '#DDAA33',
     prefix = './100/vanilla/') 

rc('font', family='serif')
plt.yscale('log')
plt.legend(fontsize=12)
plt.title('Three Body Ex. MSE', fontsize=20)
plt.xlabel('Iterations', fontsize=16)
plt.ylabel('MSE', fontsize=16)
plt.savefig(args.prefix + f'test_ex_loss-{str(args.noise)}.png', bbox_inches='tight')
