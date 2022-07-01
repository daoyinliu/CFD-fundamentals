"""
One-dimensional Advection Equation solved by 
(1) Forward in Time and Central Difference in Scape
(2) Lax-Friedrichs Scheme
(3) Upwind Scheme
(4) Cubic Semi-Lagrange
(5) CIP (Constrained Interpolation Profile) Method
Ref: 计算流体力学基础与应用, 东南大学出版社, 2021.
"""

import sys, os
sys.path.append("..")
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ------------------------------ global variables------------------------------
subFolder = "results_adv1d/"

scheme = int(input("Choose a Scheme for convection term:\n \
1-Forward in Time and Central Difference in Scape \n \
2-Lax Scheme \n \
3-1st Order Upwind \n \
4-Cubic Polynominal \n \
5-CIP \n ======>"))

### physical and geometry condition
uc = 1.; xLen = 1.; ni = 101; nci = ni - 1; dx = xLen/nci
cfl = 0.2; dt = cfl*dx/uc
nstep = 200
time = 0.

### field variables
x = np.zeros((ni,1))
f = np.zeros((ni,1))
fn = np.zeros((ni,1))
fx = np.zeros((ni,1))
fxn = np.zeros((ni,1))

# ---------------------------------- functions---------------------------------
def initializeFlowField():
    for i in range(0,ni):
        x[i] = i*dx
        f[i] = 0.
    for i in range(int(ni/5),int(ni/5*2)):
        f[i] = 1.

def each_timeloop():
    if scheme == 1: 
        for i in range(1,nci):
            fn[i] = f[i] - 0.5*cfl*(f[i+1]-f[i-1])
        f[1:-1] = fn[1:-1]
        f[0] = fn[-2]
        f[-1] = fn[1]
    elif scheme == 2:
        for i in range(1,nci):
            fn[i] = 0.5*(f[i+1]+f[i-1]) - 0.5*cfl*(f[i+1]-f[i-1])
        f[1:-1] = fn[1:-1]
        f[0] = fn[-2]
        f[-1] = fn[1]
    elif scheme == 3: 
        for i in range(1,nci):
            fn[i] = f[i] - cfl*(f[i]-f[i-1])
        f[1:-1] = fn[1:-1]
        f[0] = fn[-2]
        f[-1] = fn[1]
    elif scheme == 4:
        for i in range(2,ni-2):
            a = (f[i+1] - 3.0*f[i] + 3.0*f[i-1] - f[i-2]) / (6.0*dx*dx*dx)
            b = (f[i+1] - 2.0*f[i] + f[i-1]) / (2.0*dx*dx)
            c = (2.0*f[i+1] + 3.0*f[i] - 6.0*f[i-1] + f[i-2]) / (6.0*dx)
            z = - uc*dt
            fn[i] = a*z*z*z + b*z*z + c*z + f[i]
        f[2:-2] = fn[2:-2]  
        f[0] = fn[-2]
        f[1] = fn[-3]
        f[-1] = fn[1]
        f[-2] = fn[2]
    elif scheme == 5:
        for i in range(1,ni-1):
            a = (fx[i] + fx[i-1])/(dx*dx) - 2.0*(f[i] - f[i-1])/(dx*dx*dx)
            b = (2.0*fx[i] + fx[i-1])/dx - 3.0*(f[i] - f[i-1])/(dx*dx)
            z = - uc*dt
            fn[i] = a*z*z*z + b*z*z + fx[i]*z + f[i]
            fxn[i] = 3.0*a*z*z + 2.0*b*z + fx[i]
        fn[0] = fn[-2]
        fn[-1] = fn[1]
        fxn[0] = fxn[-2]
        fxn[-1] = fxn[1]
        f[:] = fn[:]
        fx[:] = fxn[:]
    else:
        print ("input error, exit.")
        sys.exit(1)

#--------------------------------main program----------------------------------
if __name__ == "__main__":
    ### initial setting
    initializeFlowField()
    if not os.path.exists(subFolder):
        os.makedirs(subFolder)      

    ### begin of time loop
    for istep in range(1,nstep+1):
        time += dt
        each_timeloop()
        if istep % 50 == 0 or istep == nstep:
            print('istep=%4d'%(istep))
            
    ### after time loop
# plot results
    fig = plt.figure(figsize = (6, 4.5))
    plt.plot(x, f, 'b', linewidth=2.0)
    figname = subFolder + 'fi_vs_x.jpg'
    plt.savefig(figname, format='jpg', dpi=600, bbox_inches='tight')    
    plt.show()
# save results
    resultsFileName = "adv1d_scheme%04d.csv" %(scheme)
    resultsFileName = subFolder + resultsFileName
    np.savetxt(fname = resultsFileName, X=np.hstack((x,f)), encoding='utf-8')
    print('Program Complete')
