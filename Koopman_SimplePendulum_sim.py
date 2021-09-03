
### Derivative-Based Koopman Operators ###

import matplotlib.pyplot as plt 
import numpy.random as npr
import math
import os
import autograd.numpy as np
from autograd import jacobian
import scipy.linalg
from scipy.linalg import solve_discrete_are
from scipy.linalg import logm
from scipy.integrate import solve_ivp

Start_time = float(0)          # sec
End_time = float(8)            # sec
Sampling_Frequency = float(50) # hz 
N = End_time*Sampling_Frequency
st = 1/Sampling_Frequency      # Sampling time
Sampling_time = [0, st]
l = np.arange(0, int((End_time-Start_time)/st), 1)

g = 9.81  # Gravity(m/s^2)
L = 1     # Length of pendulum(m)
b = 0.0   # Damping factor(kg/s) 
m = 0.5   # Mass(kg)

def Single_pendulum(t,s):
    theta, dtheta = s
    f = np.array([dtheta, (-b/m)*dtheta + (-g/L)*np.sin(theta) + (1/(m*L*L))*u])
    return f

NKoopman = 4   # Number of basis functions (keep inside [2,4])
Nstates = 2    # Number of system states
Ncontrol = 1   # Number of system inputs

Nstates_obs = 4                    # Number of system state terms in basis functions
Ncontrol_obs = 1                   # Number of system input terms in basis functions
N_obs = Nstates_obs + Ncontrol_obs # Number of total terms in basis functions

## Set basis functions -> Psi
def Psi_k(s,u): 
    theta, dtheta = s
    psi = zeros([NKoopman,1])
    psi[0:2, 0] = s
    if NKoopman == 3:
        psi[2, 0] = g/l*sin(theta) + u
    if NKoopman == 4:
        psi[3, 0] = g/l*cos(theta)*dtheta
    return psi

## Set list of term in a basis functions -> z
def z(s,u):
    theta, dtheta = s
    if NKoopman == 3:
        z = np.array([theta, dtheta, math.sin(theta), u])
    if NKoopman == 4:
        z = np.array([theta, dtheta, math.sin(theta), math.cos(theta)*dtheta, u])
    return z

N_sample_s = 50             # Number of sample state
N_sample_u = 100            # Number of input per state
M = N_sample_s * N_sample_u # M sized data

_A = np.zeros((N_obs, N_obs))
_G = np.zeros((N_obs, N_obs))
count = 0

## Set bounds of state and input
s_bounds = np.array([2*math.pi, 3.0])
u_bounds = np.array([2.])

## Loop to collect data
for k in range(N_sample_s):
    s0 = npr.uniform(low=-s_bounds, high=s_bounds) # Set initial random state

    for t in range(N_sample_u):
        u0 = npr.normal(-0.*u_bounds, u_bounds)    # Set initial random input
        u = u0[0]
        s_list = solve_ivp(Single_pendulum, Sampling_time, s0)
        u0 = u
        theta = s_list.y[0,:]
        theta = theta[-1]
        dtheta = s_list.y[1,:]
        dtheta = dtheta[-1]
        sn = [theta, dtheta]
        un = u0

        count = count + 1.0

        _A = _A + np.outer(z(sn, un), z(s0, u0))/count
        _G = _G + np.outer(z(s0, u0), z(s0, u0))/count

        # reset for next loop
        s0 = sn
        u0 = un
        
## Discrete Koopman Operator
K_d = np.dot(_A, np.linalg.pinv(_G))
print('Discrete Koopman Operator:',K_d)

## Continuous Koopman Operator
K_c = logm(K_d)/st
print('Continuous Koopman Operator:',K_c)

A = K_c[:Nstates_obs,:Nstates_obs]
B = K_c[:Nstates_obs, Nstates_obs:]
Q = np.diag([1.0, 1.0, 0., 0.])
R = np.diag([1.0]) * 1e-2
P = scipy.linalg.solve_continuous_are(A, B, Q, R)
K_lqr = np.linalg.inv(R).dot(B.T.dot(P)) # LQR gain

s_0 = np.array([0., 0.]) # Initial state
u_0 = 0                  # Initial control input

target_s = np.array([math.pi, 0.])           # Target state
target_z = z(target_s, u_0)[:Nstates_obs]    # Transform the target state into the lifted functions

theta_data = []
dtheta_data = []

for i in l:
    z_error = z(s_0, u_0)[:Nstates_obs] - target_z
    u = -np.dot(K_lqr, z_error)[0]
    s_list = solve_ivp(Single_pendulum, Sampling_time, s_0)
    theta = s_list.y[0,:]
    theta = theta[-1]
    dtheta = s_list.y[1,:]
    dtheta = dtheta[-1]
    theta_data.append(theta)
    dtheta_data.append(dtheta)
    s_0 = [theta, dtheta]

print('error(rad):',target_s[0]-theta_data[-1])

## Simulation
x =  L*np.sin(theta_data)
y = -L*np.cos(theta_data)

fig = plt.figure()
for point in l:
    plt.plot(x[point], y[point], 'bo')
    plt.plot([0,x[point]],[0,y[point]])
    plt.xlim(-L-0.5,L+0.5)
    plt.ylim(-L-0.5,L+0.5)
    plt.xlabel('x-direction')
    plt.ylabel('y-direction')
    plt.pause(0.01) 
    fig.clear()
plt.draw()

## Plot
plt.plot(l*st, theta_data, label = 'theta')
plt.plot(l*st, [target_s[0]]*len(l*st), label = 'target vlaue')
plt.plot(l*st, np.array([target_s[0]]*len(l*st)) - np.array(theta_data), label = 'error')
plt.xlabel('time')
plt.ylabel('rad')
plt.legend()
plt.show()
