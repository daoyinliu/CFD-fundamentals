"""
1d steady heat conduction problem
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
subFolder = "results_cond1d/"
problemTitle = "cond1d"

nxcv = int(input("Please enter number of CV x direction (nxcv=50): "))

### physical and geometry condition
gam = 1.; fi0 = 0.; fi1 = 1.; xwest = 0.; xeast = 1.
isUnsteady = 0
if (isUnsteady == 1):
    timeStep = 0.05; ntmax = 20; ntsave = 5
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

# ---------------------------------- functions---------------------------------
def initializeFlowField():
    FI[:] = 0.

def setupMaterialPropertiesAndSource():
    RHO[:] = 1.
    GAM[:] = gam
    SV[:] = 0.
    
def calcFaceFlux():
    for i in range(0,nxcv+1):
        DcondX[i] = GAM[i]/(xNode[i+1]-xNode[i])

def assemblyCoeffsInteriorNodes():
    for i in range(1,nxcv+1):
        AE[i] = DcondX[i]
        AW[i] = DcondX[i-1]
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
    figname = "_CVx%04d_%04d.csv" %(nxcv, 0)
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
            figname = "_CVx%04d_%04d.csv" %(nxcv, it)
            figfullpath = subFolder + problemTitle + figname
            saveFi(xNode, FI, figfullpath)

    ### after time loop
    fig = plt.figure(figsize = (6, 4.5))
    plt.plot(xNode, FI, 'b', linewidth=2.0)
    figname = subFolder + 'fi_vs_x.jpg'
    plt.savefig(figname, format='jpg', dpi=600, bbox_inches='tight')    
    plt.show()
    print('Program Complete')
