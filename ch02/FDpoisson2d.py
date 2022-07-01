"""
two-dimensional poisson equation, Finite Difference scheme
Ref: 计算流体力学基础与应用, 东南大学出版社, 2021.
"""

import sys, os
sys.path.append("..")
import matplotlib
#matplotlib.use('Agg')
from cfdbooktools import userSurface3D
import numpy as np
import scipy.io as scio

# ------------------------------ global variables------------------------------
subFolder = "results_poisson2d/"

### physical and geometry condition
xmin = 0.; xmax = 1.; ymin = 0.; ymax = 1.
fi_top = 100.; fi_bottom = 0.; fi_left = 0.; fi_right = 0
ni = 51; nj = 51
dx = (xmax-xmin) / (ni-1)
dy = (ymax-ymin) / (nj-1)
fi_guess = 30.
omega = 1.2
errnorm_target = 1e-4
maxstep = 1000

### field variables
x = np.linspace(xmin, xmax, ni)
y = np.linspace(ymin, ymax, nj)
fi  = np.zeros((nj, ni))
fi0 = np.zeros((nj, ni))
b = np.zeros((nj, ni))

# ---------------------------------- functions---------------------------------

#--------------------------------main program----------------------------------
if __name__ == "__main__":
    ### initial setting
    if not os.path.exists(subFolder):
        os.makedirs(subFolder)

    fi.fill(fi_guess)
    
    ### begin of iteration
    iterHistCount = []
    iterHistErr = []
    istep = 0
    errnorm = 100.
    while (istep<maxstep):
        istep += 1
        
        fi[:,-1] = fi_top
        fi[:,0] = fi_bottom
        fi[-1,:] = fi_right
        fi[0,:] = fi_left
        fi0[:,:] = fi[:,:]
        
        for i in range(1,ni-1):
            for j in range(1,nj-1):
                fi[i,j] = (((fi[i+1,j]+fi[i-1,j])*dy**2 + (fi[i,j+1]+fi[i,j-1])*dx**2 \
                         - b[i,j] * dx**2 * dy**2) / (2*(dx**2+dy**2)))*omega \
                         + (1-omega)*fi[i,j]
        
        errnorm = np.max(np.abs(fi-fi0))/(np.max(np.abs(fi0))+1e-30)
        print ("iter = %4d, errnorm=%8.4e" %(istep,errnorm))
        iterHistCount.append(istep)
        iterHistErr.append(errnorm)

        if (istep == 1 or istep%200 == 0):
            figname = "laplace2d_1stbc_%04d.jpg" %(istep)
            figname = subFolder + figname
            userSurface3D(x, y, fi, figname, ' ')

        if (errnorm < errnorm_target):
            figname = "laplace2d_1stbc_%04d.jpg" %(istep)
            figname = subFolder + figname
            userSurface3D(x, y, fi, figname, ' ')
            print ("solution converged")
            break

    ### after converged
    errorHisFileName = "case1_error_history.csv"
    errorHisFileName = subFolder + errorHisFileName
    errMat = np.vstack((np.array(iterHistCount),np.array(iterHistErr)))
    np.savetxt(fname = errorHisFileName, X = errMat.transpose(), encoding='utf-8')
    
    resultsFileName = "case1_steady_solution.mat"
    resultsFileName = subFolder + resultsFileName
    scio.savemat(resultsFileName, dict([('x', x), ('y', y), ('fi', fi)]))

    figname = "converged_laplace2d_1stbc_%04d.jpg" %(istep)
    figname = subFolder + figname
    userSurface3D(x, y, fi, figname, ' ')

    print('Program Complete')
    
