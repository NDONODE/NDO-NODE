import torch 
import torch.optim as optim
import torch.nn as nn
import numpy as np
import sympy as sp
import torch.autograd as autograd
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import random
import os
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


x = sp.Symbol('x')
Sym_basis = []

Sym_basis.append(sp.Float(1))
Sym_basis.append(x)
Sym_basis.append(x**2)
Sym_basis.append(x**3)
Sym_basis.append(sp.sin(x)/1)
Sym_basis.append(sp.cos(x)/1)
Sym_basis.append(sp.cos(5*x)/5)
Sym_basis.append(sp.sin(5*x)/5)
Sym_basis.append(sp.cos(10*x)/10)
Sym_basis.append(sp.sin(10*x)/10)
Sym_basis.append(sp.cos(20*x)/20)
Sym_basis.append(sp.sin(20*x)/20)
Sym_basis.append(sp.cos(50*x)/50)
Sym_basis.append(sp.sin(50*x)/50)
Sym_basis.append(sp.cos(75*x)/75)
Sym_basis.append(sp.sin(75*x)/75)
Sym_basis.append(sp.cos(100*x)/100)
Sym_basis.append(sp.sin(100*x)/100)
Num_basis = len(Sym_basis)


Func = []
Num_func = 10000
for i in range(Num_func):
    tn = random.randint(1,Num_basis)
    terms = random.sample(Sym_basis,tn)
    f = 0
    for bx in terms:
        f += random.uniform(-10,10)*bx
    Func.append(f)

DFunc = [f.diff(x) for f in Func]
DDFunc = [f.diff(x) for f in DFunc]

Nf = []
Ndf = []

LEN = 100 
def Nfun(k):
    Range = [random.uniform(0,1) for i in range(LEN)]
    Range.sort()
    return (Range,[float(Func[k].subs(x,r).n()) for r in Range],[float(DFunc[k].subs(x,r).n()) for r in Range],[float(DDFunc[k].subs(x,r).n()) for r in Range])
import multiprocessing as mp
pool = mp.Pool()
Nf = pool.map(Nfun, list(range(len(Func))))
Nf = np.array(Nf)

np.savez('N10k-2nd-100-Osin100xd100-no1x-irr-10pn',Nf=Nf)