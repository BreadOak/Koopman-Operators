
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

g  = float(-9.81) # Gravity(m/s^2)
L  = float(2)     # Length of pendulum(m)
d  = float(0)     # 
m  = float(1)     # Mass of pendulum(kg)
M  = float(10)    # Mass of cart(kg)
y  = float(0.5)   # Hight of cart's center(m)

def InvertedPendulum(t,s):
	x, th, dx, dth = s
	D = m*L**2*(M + m*(1-np.cos(th)**2))
	dx_dt1 = dx
	dx_dt2 = dth
	dx_dt3 = (1/D)*(-m**2*L**2*g*np.cos(th)*np.sin(th) + m*L**2*(m*L*dth**2*np.sin(th) - d*dx)) + m*L**2*(1/D)*u
	dx_dt4 = (1/D)*((M + m)*m*g*L*np.sin(th) - m*L*np.cos(th)*(m*L*dth**2*np.sin(th) - d*dx)) - m*L*np.cos(th)*(1/D)*u
	return np.array([dx_dt1, dx_dt2, dx_dt3, dx_dt4]) 

def wrap2Pi(theta):
    th = np.fmod(theta + np.pi, 2.0 * np.pi)
    if th < 0:
        th = th + 2.0 * np.pi
    return th - np.pi

Nstates_obs = 8                    # Number of system state terms in basis functions
Ncontrol_obs = 2                   # Number of system input terms in basis functions
N_obs = Nstates_obs + Ncontrol_obs # Number of total terms in basis functions

## Set list of state terms in a basis functions -> z
def z(s):
    x, th, dx, dth = s
    # th = wrap2Pi(th)
    z = np.array([x, th, dx, dth, np.cos(th)*np.sin(th)/(M + m*np.sin(th)**2), np.sin(th)*dth/(M + m*np.sin(th)**2), np.cos(th)*np.sin(th)*dth/(M + m*np.sin(th)**2), np.sin(th)/(M + m*np.sin(th)**2)])
    return z

## Set list of input terms in a basis functions -> v
def v(s,u):
	x, th, dx, dth = s
	# th = wrap2Pi(th)
	v = np.array([u/(M + m*np.sin(th)**2), u*np.cos(th)/(M + m*np.sin(th)**2)])
	return v

def dvdu(s,u0):
	x, th, dx, dth = s
	# th = wrap2Pi(th)
	u = u0[0]
	return np.array([[1/(M + m*np.sin(th)**2)], [np.cos(th)/(M + m*np.sin(th)**2)]])

def dvdz(s,u0):
	x, th, dx, dth = s
	# th = wrap2Pi(th)
	u = u0[0]
	out = np.zeros((Ncontrol_obs, Nstates_obs))
	out[:,0] = np.array([0., 0.]).T # dv/d(x)
	out[:,1] = np.array([-2*m*np.sin(th)*np.cos(th)*u/(M + m*np.sin(th)**2)**2, \
	-(2*m*np.sin(th)*np.cos(th)**2 + np.sin(th))*u/(M + m*np.sin(th)**2)**2]).T # dv/d(th)
	out[:,2] = np.array([0., 0.]).T # dv/d(dx)
	out[:,3] = np.array([0., 0.]).T # dv/d(dth)
	out[:,4] = np.array([0., 0.]).T # dv/d(np.cos(th)*np.sin(th)/(M + m*np.sin(th)**2))
	out[:,5] = np.array([0., 0.]).T # dv/d(np.sin(th)*dth/(M + m*np.sin(th)**2))
	out[:,6] = np.array([0., 0.]).T # dv/d(np.cos(th)*np.sin(th)*dth/(M + m*np.sin(th)**2))
	out[:,7] = np.array([0., 0.]).T # dv/d(np.sin(th)/(M + m*np.sin(th)**2))
	return out

N_sample_s = 100            # Number of sample state
N_sample_u = 100            # Number of input per state
M = N_sample_s * N_sample_u # M sized data

_A = np.zeros((N_obs, N_obs))
_G = np.zeros((N_obs, N_obs))
count = 0

## Set bounds of state and input
s_bounds = np.array([5.0, math.pi, 3.0, 3.0])
u_bounds = np.array([5.])

## Loop to collect data
for k in range(N_sample_s):
    #s0 = npr.uniform(low=-s_bounds, high=s_bounds) 
    s0 = npr.normal(s_bounds*0., s_bounds)      # Set initial random state
    for t in range(N_sample_u):
        u0 = npr.normal(-u_bounds, u_bounds)    # Set initial random input
        u = u0[0]
        s_list = solve_ivp(InvertedPendulum, Sampling_time, s0)
        u0 = u
        x   = s_list.y[0,:]
        th  = s_list.y[1,:]
        dx  = s_list.y[2,:]
        dth = s_list.y[3,:]
        x   = x[-1]
        th  = th[-1]
        dx  = dx[-1]
        dth = dth[-1]

        sn = [x, th, dx, dth]
        un = u0

        z1 = np.concatenate([z(s0), v(s0, u0)])
        z2 = np.concatenate([z(sn), v(sn, un)])

        count = count + 1.0
        _A = _A + np.outer(z2, z1)/count
        _G = _G + np.outer(z1, z1)/count

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
Q = np.diag([0.1, 10, 0.1, 10, 0., 0., 0., 0.])
R = np.diag([0.00001])

s_0 = np.array([0., math.pi-math.pi/18, 0., 0.]) # Initial state
u_0 = np.array([0.])                            # Initial control input

target_s = np.array([2., math.pi, 0., 0.])   # Target state
target_z = z(target_s)                       # Transform the target state into the lifted functions

x_data = []
th_data = []
dx_data = []
dth_data = []
cnt = 0

for i in l:
    z_error = z(s_0) - target_z
    A_ = A + np.dot(B,dvdz(s_0,u_0))
    B_ = np.dot(B,dvdu(s_0,u_0))

    P = scipy.linalg.solve_continuous_are(A_, B_, Q, R)
    K_lqr = np.linalg.inv(R).dot(B_.T.dot(P)) # LQR gain
    u = -np.dot(K_lqr, z_error)[0]
    print('u',u)
    s_list = solve_ivp(InvertedPendulum, Sampling_time, s_0)
    x   = s_list.y[0,:]
    th  = s_list.y[1,:]
    dx  = s_list.y[2,:]
    dth = s_list.y[3,:]
    x   = x[-1]
    th  = th[-1]
    dx  = dx[-1]
    dth = dth[-1]

    # th = wrap2Pi(th)

    x_data.append(x)
    th_data.append(th)
    dx_data.append(dx)
    dth_data.append(dth)

    s_0 = [s_0[0], s_0[1], s_0[2], s_0[3]]
    print(s_0)
    u_0 = u_0[0]
    s_n = [x, th, dx, dth]
    u_n = u

    z1 = np.concatenate([z(s_0), v(s_0, u_0)])
    z2 = np.concatenate([z(s_n), v(s_n, u_n)])

    count = count + 1.0
    _A = _A + np.outer(z2, z1)/count
    _G = _G + np.outer(z1, z1)/count
    K_d = np.dot(_A, np.linalg.pinv(_G))
    K_c = logm(K_d)/st
    A = K_c[:Nstates_obs,:Nstates_obs]
    B = K_c[:Nstates_obs, Nstates_obs:]

    u_n = np.array([u])
    s_0 = s_n
    u_0 = u_n

    print('count',i)
    print('error_x',z_error[0])
    print('error_th',z_error[1])

## Simulation 
x  = x_data
th = th_data
fig = plt.figure()
for point in l:
	plt.plot( L*math.sin(th[point]) + x[point], -L*math.cos(th[point]) + y, 'bo' )
	plt.plot( [x[point], L*math.sin(th[point]) + x[point]], [y, -L*math.cos(th[point]) + y] )
	plt.plot( [x[point]-1,x[point]+1], [y+0.5, y+0.5],'r')
	plt.plot( [x[point]-1,x[point]+1], [y-0.5, y-0.5],'r')
	plt.plot( [x[point]-1,x[point]-1], [y-0.5, y+0.5],'r')
	plt.plot( [x[point]+1,x[point]+1], [y-0.5, y+0.5],'r')
	plt.xlim(-20,20)
	plt.ylim(-1,20)
	plt.xlabel('x-direction')
	plt.ylabel('y-direction')
	plt.pause(0.01)	 
	fig.clear()
plt.draw()