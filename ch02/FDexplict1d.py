"""
one-dimensional advection-diffusion problem
Finite Difference using uniform Cartesian grid, Explicit
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

#--------------------------------main program----------------------------------
if __name__ == "__main__":
    ### initial setting
    initializeFlowField()
    if not os.path.exists(subFolder):
        os.makedirs(subFolder)      

    ### begin of time loop
    for istep in range(1,nstep+1):
        time += dt
        fi0[:] = fi[:]
        
        for i in range(1,ni-1):
            aw = 0.5*dt/h*uc + dt/h/h*D
            ap = 1. - 2*dt/h/h*D
            ae = -0.5*dt/h*uc + dt/h/h*D
            fi[i] = aw*fi0[i-1] + ap*fi0[i] + ae*fi0[i+1]
        fi[0] = 0.
        fi[-1] = 1.
        
        if istep % 20 == 0 or istep == nstep:
            if istep == nstep:
                for i in range(ni):
                    fi_exact[i] = getExact(Pe, x[i], xLen, fi[0], fi[-1])
            elif istep % 20 == 0:
                fi_exact[:] = 0. 
            figTitle = "time = %.2f" %(time)
            figName = "FD1D_explicit_dt%04dms_time%04dms" %(dt*1000,int(time*1000))
            figName = subFolder + figName
            plot1dCompareFi(x,fi,fi_exact,[0,1,-1,1],figName,figTitle)

    ### after time loop
    resultsFileName = "FD1D_explicit_grid%04d_dt%04dms.csv" %(ngrid,dt*1000)
    resultsFileName = subFolder + resultsFileName
    np.savetxt(fname = resultsFileName, X=np.hstack((x,fi,fi_exact)), encoding='utf-8')
    print('Program Complete')
    
