"""
one-dimensional advection-diffusion problem
Finite Difference using uniform Cartesian grid, Implicit
Ref: 计算流体力学基础与应用, 东南大学出版社, 2021.
"""

import sys, os
sys.path.append("..")
#import matplotlib
#matplotlib.use('Agg')
from cfdbooktools import plot1dCompareFi
import numpy as np
import math

# ------------------------------ global variables------------------------------
subFolder = "results_FD1D/"

ngrid = int(input("Please enter number of grid (ngrid=50): "))
dt = float(input("Please enter time step (dt=0.01): "))

### physical and geometry condition
xLen = 1.; uc = 0.5; D = 0.01; tend = 1.
Pe = uc*xLen/D
ni = ngrid + 1
h = xLen/ngrid
nstep = int(tend/dt)
nsave = int(0.2*tend/dt)
time = 0.

### field variables
x = np.zeros((ni,1))
fi = np.zeros((ni,1))
fi0 = np.zeros((ni,1))
fi_exact = np.zeros((ni,1))

### coefficient arrays
Aw = np.zeros((ni,1))
Ap = np.zeros((ni,1))
Ae = np.zeros((ni,1))
Qp = np.zeros((ni,1))
Ap_t = np.zeros((ni,1))
Qp_t = np.zeros((ni,1))

# ---------------------------------- functions---------------------------------
def initializeFlowField():
    for i in range(ni):
        x[i] = i*h
        fi[i] = 0.

def getExact(pe, x, L, y0, y1):
    if abs(pe)>1e-10:
        return y0 + (math.exp(x/L*pe)-1.)/(math.exp(pe)-1.)*(y1-y0)
    else:
        return y0 + (x-0.)/(1-0.)*(y1-y0)    

def TDMAsolve():
    Ap_t[1] = Ap[1]
    Qp_t[1] = Qp[1]
    # forward
    for i in range(2,ni-1):
        t = - Aw[i] / Ap_t[i-1]
        Ap_t[i] = Ap[i] + t*Ae[i-1]
        if (abs(Ap_t[i])<1e-30): 
            print ("TDMA coef failed.")
            exit
        Qp_t[i] = Qp[i] + t*Qp_t[i-1]
    # backward
    fi[ni-2] = Qp_t[ni-2] / Ap_t[ni-2]
    for i in range(ni-3,0,-1):	
        fi[i] = (Qp_t[i] - Ae[i]*fi[i+1]) / Ap_t[i]
            
#--------------------------------main program----------------------------------
if __name__ == "__main__":
    ### initial setting
    initializeFlowField()
    if not os.path.exists(subFolder):
        os.makedirs(subFolder)      

    ### begin of time loop
    for istep in range(1,nstep+1):
        time += dt

        fi[-1] = 1.
        fi[0] = 0.
        fi0[:] = fi[:]
        
        # construct algebraic equations
        for i in range(1,ni-1):
            Aw[i] = -0.5*uc*dt/h - D*dt/h/h
            Ae[i] =  0.5*uc*dt/h - D*dt/h/h
            Ap[i] = 1 + 2*D*dt/h/h
            Qp[i] = fi0[i]      
        Qp[1] -= Aw[1]*fi0[0]
        Aw[1] = 0.     
        Qp[-2] -= Ae[-2]*fi0[-1]
        Ae[-2] = 0.

        TDMAsolve()
        
        if istep % 20 == 0 or istep == nstep:
            if istep == nstep:
                for i in range(ni):
                    fi_exact[i] = getExact(Pe, x[i], xLen, fi[0], fi[-1])
            elif istep % 20 == 0:
                fi_exact[:] = 0. 
            figtitle = "time = %.2f" %(time)
            figname = "FD1D_implicit_dt%04dms_time%04dms" %(dt*1000,int(time*1000))
            figname = subFolder + figname            
            plot1dCompareFi(x,fi,fi_exact,[0,1,-1,1],figname,figtitle)
            
    ### after time loop
    resultsFileName = "FD1D_implicit_grid%04d_dt%04dms.csv" %(ngrid,dt*1000)
    resultsFileName = subFolder + resultsFileName
    np.savetxt(fname = resultsFileName, X=np.hstack((x,fi,fi_exact)), encoding='utf-8')
    print('Program Complete')
    
