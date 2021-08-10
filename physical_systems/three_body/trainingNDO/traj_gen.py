#!/usr/bin/env python
# coding: utf-8

# In[27]:
#pydevd.settrace(suspend=False, trace_only_current_thread=True)

# Import scipy
import scipy as sci
# Import matplotlib and associated modules for 3D and animations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_no', type=int, default=1)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--traj_num', type=int, default=10000)

args = parser.parse_args()


seed = args.seed
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed) 
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


save_fig_path = './figures_epoch_three_body_100_%d' % (args.experiment_no)

if not os.path.exists(save_fig_path):
    os.makedirs(save_fig_path)



# mask = 500 iter = 10000 batch = 50 lr = 0.005
time_span = np.linspace(0, 1.0, 1000)  # 20 orbital periods and 500 points
# import random
# mask =  random.sample(list(range(0,1000-1)),100)
# mask.sort()
# time_span = time_span[mask]
#print(time_span)

# Define universal gravitation constant
G = 6.67408e-11  # N-m2/kg2
# Reference quantities
m_nd = 1.989e+30  # kg #mass of the sun
r_nd = 5.326e+12  # m #distance between stars in Alpha Centauri
v_nd = 30000  # m/s #relative velocity of earth around the sun
t_nd = 79.91 * 365 * 24 * 3600 * 0.51  # s #orbital period of Alpha Centauri
# Net constants
K1 = G * t_nd * m_nd / (r_nd ** 2 * v_nd)
K2 = v_nd * t_nd / r_nd

# Define masses
m1 = 1.1  # Alpha Centauri A
m2 = 1.1  # Alpha Centauri B
m3 = 0.90  # Third Star

m1 = np.random.uniform(0.8,1.2)
m2 = np.random.uniform(0.8,1.2)
m3 = np.random.uniform(0.8,1.2)

r1 = [0.0, 0.2, 0]  # m
r2 = [0.5, 0, 0.5]  # m
r3 = [0, 0., 0.5]  # m

r1 = np.random.uniform(-0.5,0.5,[3])
r2 = np.random.uniform(-0.5,0.5,[3])
r3 = np.random.uniform(-0.5,0.5,[3])
# # Convert pos vectors to arrays
# r1 = np.array(r1, dtype="float64")
# r2 = np.array(r2, dtype="float64")
# r3 = np.array(r3, dtype="float64")

# Define initial velocities
v1 = [0.0, 0.1, 0.0]  # m/s
v2 = [-0.0, 0, 0.1]  # m/s
v3 = [ 0.1, 0, 0]

v1 = np.random.uniform(-0.1,0.1,[3])
v2 = np.random.uniform(-0.1,0.1,[3])
v3 = np.random.uniform(-0.1,0.1,[3])

# # Convert velocity vectors to arrays
# v1 = np.array(v1, dtype="float64")
# v2 = np.array(v2, dtype="float64")
# v3 = np.array(v3, dtype="float64")




def ThreeBodyEquations(w, t, G, m1, m2, m3):
    r1 = w[:3]
    r2 = w[3:6]
    r3 = w[6:9]
    v1 = w[9:12]
    v2 = w[12:15]
    v3 = w[15:18]
    r12 = sci.linalg.norm(r2 - r1)
    r13 = sci.linalg.norm(r3 - r1)
    r23 = sci.linalg.norm(r3 - r2)
    dv1bydt = K1 * m2 * (r2 - r1) / r12 ** 3 + K1 * m3 * (r3 - r1) / r13 ** 3
    dv2bydt = K1 * m1 * (r1 - r2) / r12 ** 3 + K1 * m3 * (r3 - r2) / r23 ** 3
    dv3bydt = K1 * m1 * (r1 - r3) / r13 ** 3 + K1 * m2 * (r2 - r3) / r23 ** 3
    dr1bydt = K2 * v1
    dr2bydt = K2 * v2
    dr3bydt = K2 * v3
    r12_derivs = np.concatenate((dr1bydt, dr2bydt))
    r_derivs = np.concatenate((r12_derivs, dr3bydt))
    v12_derivs = np.concatenate((dv1bydt, dv2bydt))
    v_derivs = np.concatenate((v12_derivs, dv3bydt))
    derivs = np.concatenate((r_derivs, v_derivs))
    return derivs


# Package initial parameters
init_params = np.array([r1, r2, r3, v1, v2, v3])  # Initial parameters
init_params = init_params.flatten()  # Flatten to make 1D array

# Run the ODE solver
import scipy.integrate

three_body_sol = sci.integrate.odeint(ThreeBodyEquations, init_params, time_span, args=(G, m1, m2, m3), rtol=1e-7, atol=1e-7)

r1_sol = three_body_sol[:, :3]
r2_sol = three_body_sol[:, 3:6]
r3_sol = three_body_sol[:, 6:9]

sol = []
for i in range(args.traj_num):
    m1 = np.random.uniform(0.8,1.2)
    m2 = np.random.uniform(0.8,1.2)
    m3 = np.random.uniform(0.8,1.2)
    r1 = np.random.uniform(-0.5,0.5,[3])
    r2 = np.random.uniform(-0.5,0.5,[3])
    r3 = np.random.uniform(-0.5,0.5,[3])
    v1 = np.random.uniform(-0.1,0.1,[3])
    v2 = np.random.uniform(-0.1,0.1,[3])
    v3 = np.random.uniform(-0.1,0.1,[3])
    init_params = np.array([r1, r2, r3, v1, v2, v3])  # Initial parameters
    init_params = init_params.flatten()  # Flatten to make 1D array
    three_body_sol = sci.integrate.odeint(ThreeBodyEquations, init_params, time_span, args=(G, m1, m2, m3), rtol=1e-7, atol=1e-7)
    sol.append(three_body_sol)

sol = np.array(sol)
np.save('sol',sol)

print(sol.shape)
exit(0)

# plot
# Create figure
fig = plt.figure(figsize=(15, 15))
# Create 3D axes
ax = fig.add_subplot(111, projection="3d")
# Plot the orbits
ax.plot(r1_sol[:, 0], r1_sol[:, 1], r1_sol[:, 2])
ax.plot(r2_sol[:, 0], r2_sol[:, 1], r2_sol[:, 2])
ax.plot(r3_sol[:, 0], r3_sol[:, 1], r3_sol[:, 2])

# Plot the final positions of the stars
ax.scatter(r1_sol[-1, 0], r1_sol[-1, 1], r1_sol[-1, 2], marker="o", s=100, label="Alpha Centauri A")
ax.scatter(r2_sol[-1, 0], r2_sol[-1, 1], r2_sol[-1, 2], marker="o", s=100, label="Alpha Centauri B")
ax.scatter(r3_sol[-1, 0], r3_sol[-1, 1], r3_sol[-1, 2], marker="o", s=100, label="Alpha Centauri C")

# Add a few more bells and whistles
ax.set_xlabel("x-coordinate", fontsize=14)
ax.set_ylabel("y-coordinate", fontsize=14)
ax.set_zlabel("z-coordinate", fontsize=14)
ax.set_title("Visualization of orbits of stars in a two-body system\n", fontsize=14)
ax.legend(loc="upper left", fontsize=14)

plt.draw()
plt.pause(0.001)
plt.savefig('%s/truth.png' % save_fig_path)