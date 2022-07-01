"""
1d general problem: unsteady convective-diffusion with source
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
subFolder = "results_general1d/"
problemTitle = "general1d"

nxcv = int(input("Please enter number of CV x direction (nxcv=200): "))
#adveScheme = int(input("Choose a Scheme for convection term\n \
#1-CDS, 2-UDS, 3-CDS/UDS Hybird, 4-Power-Law: "))
adveScheme = 1
#timeScheme = int(input("Choose a Scheme for time term\n \
#1-Euler Implicit, 2-Crank-Nicolson: "))
timeScheme = 2

### physical and geometry condition
rho = 1.; gam = 0.03; uc = 2.; fi0 = 0.; xwest = 0.; xeast = 1.5
isUnsteady = 1
if (isUnsteady == 1):
    timeStep = 0.01; ntmax = 150; ntsave = 10
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
SC  = np.zeros((nxcv+2))
SP  = np.zeros((nxcv+2))
DcondX = np.zeros((nxcv+1))
mdotX = np.zeros((nxcv+1))

# ---------------------------------- functions---------------------------------
def initializeFlowField():
    UX[:] = uc
    FI[:] = 0.

def setupMaterialPropertiesAndSource():
    RHO[:] = rho
    GAM[:] = gam
    for i in range(1,nxcv+1):
        SP[i] = 0.
        if (xNode[i] <= 0.6): SC[i] = -200 * xNode[i] + 100
        elif (xNode[i] <= 0.8): SC[i] = 100 * xNode[i] - 80
        else: SC[i] = 0.

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
        elif adveScheme == 2:
            AE[i] = DcondX[i] + max(-mdotX[i], 0.)
            AW[i] = DcondX[i-1] + max(mdotX[i-1], 0.)
        elif adveScheme == 3:
            AE[i] = max(-mdotX[i], DcondX[i]-mdotX[i]/2, 0.)
            AW[i] = max(mdotX[i-1], DcondX[i-1]+mdotX[i-1]/2, 0.)
        else:
            s = pow(1-0.1*abs(mdotX[i])/DcondX[i], 5.0)
            AE[i] = DcondX[i]*max(s,0) + max(-mdotX[i],0.)
            s = pow(1-0.1*abs(mdotX[i-1])/DcondX[i-1], 5.)
            AW[i] = DcondX[i-1]*max(s,0) + max(mdotX[i-1],0.)
        
        # Ap and b due to unsteady and source
        dVrdt = DeltaV[i] / timeStep * RHO[i]
        if timeScheme == 1:
            AP[i] = -(AW[i] + AE[i]) - dVrdt + SP[i] * DeltaV[i]
            SV[i] = -SC[i] * DeltaV[i] - dVrdt * FI0[i]
        elif timeScheme == 2 and it > 1:
            AW[i] *= 0.5
            AE[i] *= 0.5
            AP[i] = -(AW[i] + AE[i]) - dVrdt + SP[i] * DeltaV[i]
            SV[i] = -SC[i] * DeltaV[i] \
                    +(AW[i] + AE[i] - dVrdt) * FI0[i] \
                    -(AW[i] * FI0[i-1] + AE[i] * FI0[i+1])
        else:
            AP[i] = -(AW[i] + AE[i]) - dVrdt + SP[i] * DeltaV[i]
            SV[i] = -SC[i] * DeltaV[i] - dVrdt * FI0[i]

def set1stBC():
    FI[0] = fi0

def modifyCoeffsBCNeighborNodes():
    SV[1] -= AW[1]*FI[0]
    AW[1] = 0.
    AP[-2] += AE[-2]
    AE[-2] = 0.
    
def updateBC():
    FI[-1] = FI[-2]
    return

def saveFi(it):
    if (isUnsteady == 1): 
        fname = "_CVx%04d_%04d.csv" %(nxcv, it)
    else:
        fname = "_CVx%04d_%04d.csv" %(nxcv, 0)
    resfullpath = subFolder + problemTitle + fname
    np.savetxt(fname = resfullpath, X=np.c_[xNode, FI], encoding='utf-8')
            
#--------------------------------main program----------------------------------
if __name__ == "__main__":
    ### initial setting
    initializeFlowField()
    if not os.path.exists(subFolder):
        os.makedirs(subFolder)
    saveFi(0)
    
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
            saveFi(it)

    ### after time loop
    fig = plt.figure(figsize = (6, 4.5))
    plt.plot(xNode, FI, 'b', linewidth=2.0)
    figname = subFolder + 'fi_vs_x.jpg'
    plt.savefig(figname, format='jpg', dpi=600, bbox_inches='tight')
    plt.show()
    print('Program Complete')
