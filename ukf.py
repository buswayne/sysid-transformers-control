# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 09:34:36 2015

@author: rlabbe
"""

import numpy as np
import matplotlib.pyplot as plt

from numpy import array, asarray
from numpy.random import randn
import math
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints


def fx(x, dt):
     # state transition function - predict next state based
     # on constant velocity model x = vt + x_0
     F = np.array([[1, dt, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, dt],
                   [0, 0, 0, 1]], dtype=float)
     return np.dot(F, x)

def hx(x):
    # measurement function - convert state into a measurement
    # where measurements are [x_pos, y_pos]
    return np.array([x[0], x[2]])

dt = 0.1
# create sigma points to use in the filter. This is standard for Gaussian processes
points = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=-1)

kf = UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=dt, fx=fx, hx=hx, points=points)
kf.x = np.array([-1., 1., -1., 1]) # initial state
kf.P *= 0.2 # initial uncertainty
z_std = 0.1
kf.R = np.diag([z_std**2, z_std**2]) # 1 standard
kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.01**2, block_size=2)

zs = np.array([[i+randn()*z_std, i+randn()*z_std] for i in range(50)]) # measurements

x_hat = np.zeros((len(zs),2))

x_0 = np.array([-1., 1., -1., 1])

for i, z in enumerate(zs):
    x_real = fx(x_0, dt)
    kf.predict()
    x_hat[i,0] = kf.x[0]
    x_hat[i,1] = kf.x[2]
    kf.update(z)
    #print(kf.x, 'log-likelihood', kf.log_likelihood)

print(kf)

plt.subplot(121)
plt.plot(x_hat[:,0])
plt.plot(zs[:,0])
plt.subplot(122)
plt.plot(x_hat[:,1])
plt.plot(zs[:,1])
plt.show()