"""
Cavity problem, MAC FD(FV) on a uniform staggered grid
Ref: 计算流体力学基础与应用, 东南大学出版社, 2021.
"""

import sys, os, platform
sys.path.append("..")
if (platform.system()=='Linux'):
    import matplotlib
    matplotlib.use('Agg')
from cfdbooktools import \
     meshing1dUniform, calcGeometry, exportToTecplot, userContour2D
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------ global variables------------------------------
### problme state
subFolder = "results_cavity/"
problemTitle = "cavity"
numUDS = 5
titleUDS = ("stream","vortex","Yi","k","epsi")
global tecFileID

nxcv = int(input("Please enter number of CV x direction (nxcv=20): "))
nycv = int(input("Please enter number of CV y direction (nycv=20): "))

### geometry setting
xwest = 0.; xeast = 1.; ysouth = 0.; ynorth = 1.

### physical setting
RE = 100.; rho = 1.; uLID = 1.
vis = rho*uLID*(xeast-xwest)/RE
nu = vis/rho

### numerical setting
maxInnerIter = 1000; errnorm_target = 1e-4; omega = 1.7; tau = 0.25
t_end = 2.; dt = 0.02; dt_save = 0.05
ntsave = int(dt_save/dt)
curTime = 0.

### CREATE mesh array
xh, xFace = meshing1dUniform(xwest,xeast,nxcv)
yh, yFace = meshing1dUniform(ysouth,ynorth,nycv)
xNode,yNode,xFraSe,yFraSn,volCell,Rf,Rp = \
    calcGeometry(nxcv,xFace,nycv,yFace,0)
dt = 1/(1/xh**2+1/yh**2)*RE/2 * tau # recommended time step

### CREATE filed variable array
U = np.zeros((nxcv+2,nycv+2)) # fictitious points: U[:,0], U[:,nycv+1]
V = np.zeros((nxcv+2,nycv+2)) # fictitious points: V[0,:], V[nxcv+1,:]
P = np.zeros((nxcv+2,nycv+2))
P0 = np.zeros((nxcv+2,nycv+2))
Ustar = np.zeros((nxcv+2,nycv+2))
Vstar = np.zeros((nxcv+2,nycv+2))
RHSP = np.zeros((nxcv+2,nycv+2))
Uc = np.zeros((nxcv+2,nycv+2)) 
Vc = np.zeros((nxcv+2,nycv+2))
UDS = np.zeros((nxcv+2,nycv+2,numUDS)) # store UDS variables

# -------------------------------- functions-----------------------------------
def initializeFlowField():
    U[:,:] = 0.
    V[:,:] = 0.

def setBC_UV():
    # bottom, wall
    U[:,0] = -U[:,1]
    V[:,0] = 0.
    # top, moving wall
    U[:,nycv+1] = 2*uLID - U[:,nycv]
    V[:,nycv] = 0.
    # left, wall
    U[0,:] = 0.
    V[0,:] = -V[1,:]
    # right, wall
    U[nxcv,:] = 0.
    V[nxcv+1,:] = -V[nxcv,:] 

def setBC_P():
    # dp/dn=0
    P[:,0] = P[:,1]
    P[:,nycv+1] = P[:,nycv] 
    P[0,:] = P[1,:]
    P[nxcv+1,:] = P[nxcv,:]
    # correct to 0 at right-top corner
    pre_ref = P[-1,-1]
    P[:,:] = P[:,:] - pre_ref

def solveUVstar():
    for i in range(1,nxcv):
        for j in range(1,nycv+1):
            duudx = ((U[i,j]+U[i+1,j])**2 - (U[i-1,j]+U[i,j])**2)/(4.0*xh)
            duvdy = ((V[i,j]+V[i+1,j])*(U[i,j]+U[i,j+1]) \
                   - (V[i,j-1]+V[i+1,j-1])*(U[i,j-1]+U[i,j]))/(4.0*yh)
            laplu = (U[i+1,j]-2.0*U[i,j]+U[i-1,j])*(nu/xh/xh) \
                  + (U[i,j+1]-2.0*U[i,j]+U[i,j-1])*(nu/yh/yh)
            Ustar[i,j] = U[i,j] + dt*(laplu-duudx-duvdy)
                        
    for i in range(1,nxcv+1):
        for j in range(1,nycv):
            duvdx = ((U[i,j]+U[i,j+1])*(V[i,j]+V[i+1,j]) \
                   - (U[i-1,j]+U[i-1,j+1])*(V[i-1,j]+V[i,j]))/(4.0*xh)
            dvvdy = ((V[i,j]+V[i,j+1])**2 - (V[i,j-1]+V[i,j])**2)/(4.0*yh)
            laplv = (V[i+1,j]-2.0*V[i,j]+V[i-1,j])*(nu/xh/xh) \
                  + (V[i,j+1]-2.0*V[i,j]+V[i,j-1])*(nu/yh/yh)
            Vstar[i,j] = V[i,j] + dt*(laplv-duvdx-dvvdy)

# if xh==yh, the calculation can be reduced as the following:
#    h = xh
#    for i in range(1,nxcv):
#        for j in range(1,nycv+1):
#            rhs1 = (-(0.25/h)*( \
#              -(U[i+1,j]+U[i,j])**2+(U[i,j]+U[i-1,j])**2 \
#              -(U[i,j+1]+U[i,j])*(V[i+1,j]+V[i,j]) \
#              +(U[i,j]+U[i,j-1])*(V[i+1,j-1]+V[i,j-1])) \
#              +(nu/h/h)*(U[i+1,j]+U[i-1,j]+U[i,j+1]+U[i,j-1]-4*U[i,j]))
#            Ustar[i,j] = U[i,j] + dt*rhs1
#                        
#    for i in range(1,nxcv+1):
#        for j in range(1,nycv):           
#            rhs2 = (-(0.25/h)*( \
#              (V[i,j+1]+V[i,j])**2-(V[i,j]+V[i,j-1])**2 \
#              +(U[i,j+1]+U[i,j])*(V[i+1,j]+V[i,j]) \
#              -(U[i-1,j+1]+U[i-1,j])*(V[i,j]+V[i-1,j])) \
#              +(nu/h/h)*(V[i+1,j]+V[i-1,j]+V[i,j+1]+V[i,j-1]-4*V[i,j]))
#            Vstar[i,j] = V[i,j] + dt*rhs2
            
    # at external boundary
    Ustar[0,:] = U[0,:]
    Ustar[nxcv,:] = U[nxcv,:]
    Vstar[:,0] = V[:,0]
    Vstar[:,nycv] = V[:,nycv]

def calcRHSofP():
    dtxh = 1.0/(dt*xh)
    dtyh = 1.0/(dt*yh)
    for i in range(1,nxcv+1):
        for j in range(1,nycv+1):
            RHSP[i][j] = (Ustar[i][j]-Ustar[i-1][j])*dtxh \
                      + (Vstar[i][j]-Vstar[i][j-1])*dtyh

def solvePoissonEqu():
    it = 0
    errnorm = 100.
    while (it<maxInnerIter):
        it += 1
        setBC_P()
        P0[:,:] = P[:,:]
        for i in range(1,nxcv+1):
            for j in range(1,nycv+1):
                P[i,j] = (((P[i+1,j]+P[i-1,j])*yh**2 + (P[i,j+1]+P[i,j-1])*xh**2 \
                         - RHSP[i,j] * xh**2 * yh**2) / (2*(xh**2+yh**2)))*omega \
                         + (1-omega)*P[i,j]        

        if (it>2):
            errnorm = np.max(np.abs(P-P0))/np.max(np.abs(P0))
            #print ("iter = %4d, errnorm=%8.4e" %(it,errnorm))
        if (errnorm < errnorm_target):
            print ("poisson solver: converged at iter = %4d, errnorm=%8.4e" %(it,errnorm))
            break
    return errnorm
    
def correctUV():
    U[1:nxcv,1:nycv+1] = Ustar[1:nxcv,1:nycv+1] \
                       - (dt/xh)*(P[2:nxcv+1,1:nycv+1]-P[1:nxcv,1:nycv+1])
    V[1:nxcv+1,1:nycv] = Vstar[1:nxcv+1,1:nycv] \
                       - (dt/yh)*(P[1:nxcv+1,2:nycv+1]-P[1:nxcv+1,1:nycv])
    # extrapolate values at (virtual) Edges
    U[nxcv+1,:] = U[nxcv,:]
    V[:,nycv+1] = V[:,nycv]

def openAndSaveInitialResults():
    global tecFileID    
    filename = "RE%04d_x%04dy%04d_tec.dat" %(int(RE), nxcv, nycv)
    filename = subFolder + problemTitle + "_tran_" + filename
    tecFileID = open(filename, mode='w')
    exportToTecplot(tecFileID,curTime,xNode,yNode,Uc,Vc,P, \
                    UDS,titleUDS,numUDS,nxcv,nycv)

def calcStreamFunctionVortex():
    for j in range(1,nycv+1):
        for i in range(1,nxcv+1):
            Uc[i][j] = 0.5 * (U[i][j] + U[i-1][j])
            Vc[i][j] = 0.5 * (V[i][j] + V[i][j-1])
    # extrapolation values at (virtual) Edges
    Uc[0,:] = U[0,:]
    Uc[-1,:] = U[-1,:]
    Uc[:,0] = U[:,0]
    Uc[:,-1] = U[:,-1]
    Vc[0,:] = V[0,:]
    Vc[-1,:] = V[-1,:]
    Vc[:,0] = V[:,0]
    Vc[:,-1] = V[:,-1]
    # stream function, using the V velocity
    # U = d(rho*psi)/dy, V = -d(rho*psi)/dx
    UDS[0,0,0] = 0.
    for j in range(1,nycv+2):
        UDS[0,j,0] = UDS[0,j-1,0] + rho*U[0][j]*yh
    for i in range(1,nxcv+2):
        UDS[i,0:nycv+2,0] = UDS[i-1,0:nycv+2,0] - rho*V[i,0:nycv+2]*xh
    # vortex, w = dv/dx - du/dy
    for i in range(1,nxcv+1):
      for j in range(1,nycv+1):
        UDS[i,j,1] = (V[i+1,j]-V[i,j])/xh - (U[i,j+1]-U[i,j])/yh

#--------------------------------main program----------------------------------
if __name__ == "__main__":
    global tecFileID
    initializeFlowField()
    setBC_UV()
    if not os.path.exists(subFolder):
        os.makedirs(subFolder)
    openAndSaveInitialResults()
    
    ### begin of time loop
    iterCount = []
    iterErr = []
    istep = 0
    while curTime<t_end:
        istep += 1
        curTime += dt
        solveUVstar()
        calcRHSofP()
        continuityErr = solvePoissonEqu()
        print ("======timeid=%05d, time =%6.2e, continutyErr=%6.2e======" \
               %(istep,curTime,continuityErr))
        correctUV()
        setBC_UV()
        iterCount.append(curTime)
        iterErr.append(continuityErr)
        if istep%ntsave == 0:
            calcStreamFunctionVortex()
            exportToTecplot(tecFileID,curTime,xNode,yNode,Uc,Vc,P, \
                            UDS,titleUDS,numUDS,nxcv,nycv)
    
    ### after time loop
    tecFileID.close()
    figtitle = 'stream function'
    figname = subFolder + 'converged_stream.jpg'
    userContour2D(xNode, yNode, UDS[:,:,0], figname, figtitle, [8,6.5])

    fig = plt.figure(figsize = (6, 4))
    plt.semilogy(iterCount, iterErr)
    plt.xlabel('time (s)', fontsize=15)
    plt.ylabel('residual error', fontsize=15)
    figname = subFolder + 'residual_error.jpg'
    plt.savefig(figname, format='jpg', dpi=600, bbox_inches='tight')
    plt.show()

    uxfilename = "Re%04d_N%04d.csv" %(int(RE), nycv)
    fullname = subFolder + problemTitle + "_ux_center_x_" + uxfilename
    df = pd.DataFrame({"a":yNode, "b":U[int(nxcv/2)]})   
    df.to_csv(fullname,index=False,header=None)

    uyfilename = "Re%04d_N%04d.csv" %(int(RE), nycv)
    fullname = subFolder + problemTitle + "_uy_center_y_" + uyfilename 
    df = pd.DataFrame({"a":xNode, "b":V[:,int(nycv/2)]})
    df.to_csv(fullname,index=False,header=None)

    print('Program Complete')
