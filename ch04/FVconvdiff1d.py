"""
1d steady convective-diffusion problem
Finite Volumn Method using uniform Cartesian grid
Ref: 计算流体力学基础与应用, 东南大学出版社, 2021.
"""

import sys, os
sys.path.append("..")
#import matplotlib
#matplotlib.use('Agg')
from cfdbooktools import meshing1dUniform, calcGeometry1d, TDMA1d
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------ global variables------------------------------
subFolder = "results_convdiff1d/"
problemTitle = "convdiff1d"

nxcv = int(input("Please enter number of CV x direction (nxcv=50): "))
adveScheme = int(input("Choose a Scheme for convection term\n \
1-CDS, 2-UDS, 3-CDS/UDS Hybird, 4-QUICK, 5-TVD(QUICK), 6-Power-Law: "))

### physical and geometry condition
gam = 0.01; uc = 0.5; fi0 = 0.; fi1 = 1.; xwest = 0.; xeast = 1.
isUnsteady = 0
if adveScheme == 4 or adveScheme == 5: isUnsteady=1
if (isUnsteady == 1):
    timeStep = 0.05; ntmax = 50; ntsave = 50
else:
    timeStep = 1e30; ntmax = 1; ntsave = 1
epsi = 1e-3; curTime = 0.

### create mesh
xh, xFace = meshing1dUniform(xwest,xeast,nxcv)
xNode, xFraSe, DeltaV = calcGeometry1d(nxcv,xFace)

### define field variables
RHO = np.zeros((nxcv+2))
GAM = np.zeros((nxcv+2))
UX = np.zeros((nxcv+2))
FI = np.zeros((nxcv+2))
FI0 = np.zeros((nxcv+2))

### define coefficients of linear algebraic eqs
AW = np.zeros((nxcv+2))
AP = np.zeros((nxcv+2))
AE = np.zeros((nxcv+2))
SV  = np.zeros((nxcv+2))
DcondX = np.zeros((nxcv+1))
mdotX = np.zeros((nxcv+1))

# ---------------------------------- functions---------------------------------
def initializeFlowField():
    UX[:] = uc
    FI[:] = 0.

def setupMaterialPropertiesAndSource():
    RHO[:] = 1.
    GAM[:] = gam
    SV[:] = 0.

def calcFaceFlux():
    for i in range(0,nxcv+1):
        DcondX[i] = GAM[i]/(xNode[i+1]-xNode[i])
        if i == 0: ux = UX[i]
        elif i == nxcv: ux = UX[i+1]
        else: ux = 0.5*(UX[i]+UX[i+1])
        mdotX[i] = RHO[i]*ux

def assemblyCoeffsInteriorNodes():
    for i in range(1,nxcv+1):
        if adveScheme == 1:
            AE[i] = DcondX[i] - mdotX[i]/2
            AW[i] = DcondX[i-1] + mdotX[i-1]/2
            AP[i] = - (AW[i] + AE[i])
            SV[i] = 0.
        elif adveScheme == 2:
            AE[i] = DcondX[i] + max(-mdotX[i], 0.)
            AW[i] = DcondX[i-1] + max(mdotX[i-1], 0.)
            AP[i] = - (AW[i] + AE[i])
            SV[i] = 0.
        elif adveScheme == 3:
            AE[i] = max(-mdotX[i], DcondX[i]-mdotX[i]/2, 0.)
            AW[i] = max(mdotX[i-1], DcondX[i-1]+mdotX[i-1]/2, 0.)
            AP[i] = - (AW[i] + AE[i])
            SV[i] = 0.
        elif adveScheme == 4 and i>1: # only works for me>0 and mw>0
            AE[i] = DcondX[i] - 3./8*mdotX[i]
            AW[i] = DcondX[i-1] + 6./8*mdotX[i-1] + 1./8*mdotX[i]
            aww = -1./8*mdotX[i-1]
            AP[i] = - (AW[i] + AE[i] + aww)
            SV[i] =- aww*FI0[i-2]
        elif adveScheme == 5: # only works for me>0 and mw>0
            AE[i] = DcondX[i]
            AW[i] = DcondX[i-1] + mdotX[i-1]
            AP[i] = - (AW[i] + AE[i])
            r = (FI0[i]-FI0[i-1])/((FI0[i+1]-FI0[i])+1e-30)
            t1 = 0.5 * (3+r)/4 * (FI0[i+1]-FI0[i])
            t2 = 0.5 * (3+r)/4 * (FI0[i]-FI0[i-1])
            SV[i] = - mdotX[i]*t1 + mdotX[i-1]*t2
        else:
            s = pow(1-0.1*abs(mdotX[i])/DcondX[i], 5.0)
            AE[i] = DcondX[i]*max(s,0) + max(-mdotX[i],0.)
            s = pow(1-0.1*abs(mdotX[i-1])/DcondX[i-1], 5.)
            AW[i] = DcondX[i-1]*max(s,0) + max(mdotX[i-1],0.)
            AP[i] = - (AW[i] + AE[i])
            SV[i] = 0.

def set1stBC():
    FI[0] = fi0
    FI[-1] = fi1 

def modifyCoeffsBCNeighborNodes():
    SV[1] -= AW[1]*FI[0]
    AW[1] = 0.
    SV[-2] -= AE[-2]*FI[-1]
    AE[-2] = 0.

def updateBC():
    return

def saveFi(xNode, FI, figfullpath):
    np.savetxt(fname = figfullpath, X=np.c_[xNode, FI], encoding='utf-8')

#--------------------------------main program----------------------------------
if __name__ == "__main__":
    ### initial setting
    initializeFlowField()
    if not os.path.exists(subFolder):
        os.makedirs(subFolder)
    figname = "_scheme%1d_CVx%04d_%04d.csv" %(adveScheme, nxcv, 0)
    figfullpath = subFolder + problemTitle + figname
    saveFi(xNode, FI, figfullpath)
    
    ### begin of time loop
    for it in range(1,ntmax+1):
        curTime += timeStep
        print ("outiter = %4d, time=%8.5f" %(it,curTime))
        FI0[:] = FI[:]
        setupMaterialPropertiesAndSource()
        set1stBC()
        calcFaceFlux()
        assemblyCoeffsInteriorNodes()
        modifyCoeffsBCNeighborNodes()
        TDMA1d(AE, AP, AW, SV, FI, nxcv)
        updateBC()
        if it%ntsave == 0:
            figname = "_scheme%1d_CVx%04d_%04d.csv" %(adveScheme, nxcv, it)
            figfullpath = subFolder + problemTitle + figname
            saveFi(xNode, FI, figfullpath)

    ### after time loop
    fig = plt.figure(figsize = (6, 4.5))
    plt.plot(xNode, FI, 'b', linewidth=2.0)    
    figname = subFolder + 'fi_vs_x.jpg'
    plt.savefig(figname, format='jpg', dpi=600, bbox_inches='tight')    
    plt.show()
    print('Program Complete')
