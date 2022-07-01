"""
stagnation problem, 2d (steady/unsteady) convection-diffusion
Finite Volumn Method using uniform Cartesian grid
Ref: 计算流体力学基础与应用, 东南大学出版社, 2021.
"""

import sys, os
sys.path.append("..")
#import matplotlib
#matplotlib.use('Agg')
from cfdbooktools import \
    meshing1dUniform, calcGeometry, SIPSOL2D, userContour2D
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

# ------------------------------ global variables------------------------------
subFolder = "results_stag2d/"
problemTitle = "stag2d"

nxcv = int(input("Please enter number of CV x direction (nxcv=40): "))
nycv = int(input("Please enter number of CV y direction (nycv=40): "))
adveScheme = int(input("Choose a Scheme for convection term (3)\n \
1-CDS, 2-UDS, 3-CDS/UDS Hybird: "))
timeScheme = int(input("Choose a Scheme for time term (1)\n \
1-Euler Implicit, 2-Crank-Nicolson: "))

### physical and geometry condition
rho = 1.; gam = 0.05
xwest = 0.; xeast = 1.; ysouth = 0.; ynorth = 1.

### numerical controls
isUnsteady = 1
if (isUnsteady == 1):
    timeStep = 0.25; ntmax = 20; ntsave = 1
else:
    timeStep = 1e30; ntmax = 1; ntsave = 1
epsi = 1e-3
curTime = 0.

### create mesh
xh, xFace = meshing1dUniform(xwest,xeast,nxcv)
yh, yFace = meshing1dUniform(ysouth,ynorth,nycv)
xNode,yNode,xFraSe,yFraSn,DeltaV,Rf,Rp = \
    calcGeometry(nxcv,xFace,nycv,yFace,0)
    
### define field variables
RHO = np.zeros((nxcv+2,nycv+2))
GAM = np.zeros((nxcv+2,nycv+2))
UX = np.zeros((nxcv+2,nycv+2))
UY = np.zeros((nxcv+2,nycv+2))
FI = np.zeros((nxcv+2,nycv+2))
FI0 = np.zeros((nxcv+2,nycv+2))

### define coefficients of linear algebraic eqs
AW = np.zeros((nxcv+2,nycv+2))
AP = np.zeros((nxcv+2,nycv+2))
AE = np.zeros((nxcv+2,nycv+2))
AS = np.zeros((nxcv+2,nycv+2))
AN = np.zeros((nxcv+2,nycv+2))
SV  = np.zeros((nxcv+2,nycv+2))
SP = np.zeros((nxcv+2,nycv+2))
SC = np.zeros((nxcv+2,nycv+2))
DcondX = np.zeros((nxcv+1,nycv+1))
DcondY = np.zeros((nxcv+1,nycv+1))
mdotX = np.zeros((nxcv+1,nycv+1))
mdotY = np.zeros((nxcv+1,nycv+1))

# ---------------------------------- functions---------------------------------
def initializeFlowField():
    FI[:,:] = 0.  
    for i in range(0,nxcv+2):
        for j in range(0,nycv+2):
            UX[i,j] = xNode[i]
            UY[i,j] = -yNode[j]

def setupMaterialPropertiesAndSource():
    RHO[:,:] = rho
    GAM[:,:] = gam
    SP[:,:] = 0.
    SC[:,:] = 0.  

def calcFaceFlux():
    # x direction, faces [0, nxcv], [1,nycv]
    for j in range(1,nycv+1):
        for i in range(0,nxcv+1):
            if i == 0: ux = UX[i,j]
            elif i == nxcv: ux = UX[i+1,j]
            else: ux = 0.5*(UX[i,j]+UX[i+1,j])
            mdotX[i,j] = RHO[i,j]*ux*(yFace[j]-yFace[j-1])*Rp[j] 
            DcondX[i,j] = GAM[i,j]*(yFace[j]-yFace[j-1])/(xNode[i+1]-xNode[i])*Rp[j] 
    # flux y direction, faces [1,nxcv], [0, nycv] 
    for i in range(1,nxcv+1):
        for j in range(0,nycv+1):
            if j == 0: uy = UY[i,j]
            elif j == nycv: uy = UY[i,j+1]
            else: uy = 0.5*(UY[i,j]+UY[i,j+1])
            mdotY[i,j] = RHO[i,j]*uy*(xFace[i]-xFace[i-1])*Rf[j]
            DcondY[i,j] = GAM[i,j]*(xFace[i]-xFace[i-1])/(yNode[j+1]-yNode[j])*Rf[j]

def assemblyCoeffsInteriorNodes(it):
    for i in range(1,nxcv+1):
        for j in range(1,nycv+1):
            if adveScheme == 1:
                AE[i,j] = DcondX[i,j]   -  xFraSe[i]*mdotX[i,j]
                AW[i,j] = DcondX[i-1,j] + (1-xFraSe[i-1])*mdotX[i-1,j]
                AN[i,j] = DcondY[i,j]   -  yFraSn[j]*mdotY[i,j]
                AS[i,j] = DcondY[i,j-1] + (1-yFraSn[j-1])*mdotY[i,j-1]
            elif adveScheme == 2:
                AE[i,j] = DcondX[i,j]   + max(-mdotX[i,j], 0.)
                AW[i,j] = DcondX[i-1,j] + max(mdotX[i-1,j], 0.)
                AN[i,j] = DcondY[i,j]   + max(-mdotY[i,j], 0.)
                AS[i,j] = DcondY[i,j-1] + max(mdotY[i,j-1], 0.)
            else:
                AE[i,j] = DcondX[i,j]   + \
                max(-mdotX[i,j], DcondX[i,j] - xFraSe[i]*mdotX[i,j], 0.)
                AW[i,j] = DcondX[i-1,j] + \
                max(mdotX[i-1,j],DcondX[i-1,j] + (1-xFraSe[i-1])*mdotX[i-1,j], 0.)
                AN[i,j] = DcondY[i,j]   +  \
                max(-mdotY[i,j], DcondY[i,j] - yFraSn[j]*mdotY[i,j], 0.)
                AS[i,j] = DcondY[i,j-1] + \
                max(mdotY[i,j-1],DcondY[i,j-1] + (1-yFraSn[j-1])*mdotY[i,j-1], 0.)
                
            # Ap and b due to unsteady and source
            AP[i,j] = -(AW[i,j] + AE[i,j] + AS[i,j] + AN[i,j])
            SV[i,j] = 0.
            if isUnsteady == 1:
                dVrdt = DeltaV[i,j] / timeStep * RHO[i,j]
                if timeScheme == 1:
                    AP[i,j] = -(AW[i,j] + AE[i,j] + AS[i,j] + AN[i,j]) \
                              - dVrdt + SP[i,j] * DeltaV[i,j]
                    SV[i,j] = -SC[i,j] * DeltaV[i,j] - dVrdt * FI0[i,j]
                elif timeScheme == 2 and it > 1:
                    AW[i,j] *= 0.5
                    AE[i,j] *= 0.5
                    AS[i,j] *= 0.5
                    AN[i,j] *= 0.5
                    AP[i,j] = -(AW[i,j] + AE[i,j] + AS[i,j] + AN[i,j]) \
                              - dVrdt + SP[i,j] * DeltaV[i,j]
                    SV[i,j] = -SC[i,j] * DeltaV[i,j] \
                   +(AW[i,j] + AE[i,j] + AN[i,j] + AS[i,j] - dVrdt) * FI0[i,j] \
                   -(AW[i,j] * FI0[i-1,j] + AE[i,j] * FI0[i+1,j] + \
                     AS[i,j] * FI0[i,j-1] + AN[i,j] * FI0[i,j+1])
                else:
                    AP[i,j] = -(AW[i,j] + AE[i,j] + AS[i,j] + AN[i,j]) \
                              - dVrdt + SP[i,j] * DeltaV[i,j]
                    SV[i,j] = -SC[i,j] * DeltaV[i,j] - dVrdt * FI0[i,j]
                    
def set1stBC():
    FI[0,:] = 1.0 - yNode   # west, 1st BC
    FI[1:nxcv+1,-1] = 0.0  # north, 1st BC
    
def modifyCoeffsBCNeighborNodes():
    SV[1,:] -= np.multiply(AW[1,:],FI[0,:]) # west
    AW[1,:] = 0.0
    AP[-2,:] += AE[-2,:] # east
    AE[-2,:] = 0.0    
    AP[:,1] += AS[:,1] # south
    AS[:,1] = 0.0 
    SV[:,-2] -= np.multiply(AN[:,-2],FI[:,-1]) # north
    AN[:,-2] = 0.0  
    
def updateBC():
    FI[-1,:] = FI[-2,:] # east
    FI[:,0] = FI[:,1] # south

def saveFi(it):
    if (isUnsteady == 1): 
        figtitle = "time=" + "{0:8.4f}".format(curTime)
        fname = "_CVx%04dCVy%04d_%04d" %(nxcv, nycv, it)
    else:
        figtitle = " "
        fname = "_CVx%04dCVy%04d_steady" %(nxcv, nycv)
    resfullpath = subFolder + problemTitle + '_field' + fname
    figfullpath = subFolder + problemTitle + '_contour' + fname + '.jpg'
    scio.savemat(resfullpath, dict([('x', xNode), ('y', yNode), ('fi', FI)]))
    userContour2D(xNode, yNode, FI, figfullpath, figtitle, [8,6.5])
    
def monitorFaceFluxBoundary():
    # west and east boundary
    Fc_w = 0.
    Fc_e = 0.
    Fd_w = 0.
    Fd_e = 0.
    Fc_w = np.dot(mdotX[0,1:nycv+1],FI[0,1:nycv+1])
    Fc_e = - np.dot(mdotX[nxcv,1:nycv+1],FI[nxcv+1,1:nycv+1])
    Fd_w = np.dot(DcondX[0,1:nycv+1],FI[0,1:nycv+1]-FI[1,1:nycv+1])
    Fd_e = - np.dot(DcondX[nxcv,1:nycv+1],FI[nxcv,1:nycv+1]-FI[nxcv+1,1:nycv+1])
    # south and north boundary
    Fc_s = 0.
    Fc_n = 0.
    Fd_s = 0.
    Fd_n = 0.
    Fc_s = np.dot(mdotY[1:nxcv+1,0],FI[1:nxcv+1,0])
    Fc_n = - np.dot(mdotY[1:nxcv+1,nycv],FI[1:nxcv+1,nycv+1])
    Fd_s = np.dot(DcondY[1:nxcv+1,0],FI[1:nxcv+1,0]-FI[1:nxcv+1,1])
    Fd_n = - np.dot(DcondY[1:nxcv+1,nycv],FI[1:nxcv+1,nycv]-FI[1:nxcv+1,nycv+1])
    #
    #return [Fc_w, Fc_e, Fc_s, Fc_n, Fd_w, Fd_e, Fd_s, Fd_n]
    return Fc_w + Fc_e + Fc_s + Fc_n + Fd_w + Fd_e + Fd_s + Fd_n

#--------------------------------main program----------------------------------
if __name__ == "__main__":
    ### initial setting
    initializeFlowField()
    if not os.path.exists(subFolder):
        os.makedirs(subFolder)
    #saveFi(0)

    ### begin of time loop
    timeList = []
    FluxBoundaryNET = []
    for it in range(1,ntmax+1):
        curTime += timeStep
        print ("outiter = %4d, time=%8.5f" %(it,curTime))
        FI0[:,:] = FI[:,:]
        setupMaterialPropertiesAndSource()
        set1stBC()
        calcFaceFlux()
        assemblyCoeffsInteriorNodes(it)
        modifyCoeffsBCNeighborNodes()
        resor = SIPSOL2D(AE, AW, AN, AS, AP, SV, FI, nxcv, nycv, epsi)
        updateBC()
        timeList.append(curTime)
        FluxBoundaryNET.append(monitorFaceFluxBoundary())
        if it%ntsave == 0:
            saveFi(it)

    ### after time loop
    resultsFileName = "fluxBCwithTime.csv"
    resultsFileName = subFolder + resultsFileName
    fluxMat = np.vstack((np.array(timeList),np.array(FluxBoundaryNET)))
    np.savetxt(fname = resultsFileName, X = fluxMat.transpose(), encoding='utf-8')
    
    fig = plt.figure(figsize = (6, 4.5))
    plt.plot(timeList, FluxBoundaryNET, 'b', linewidth=2.0)
    plt.xlabel('time (s)', fontsize=18)
    plt.ylabel('net flux (kg/s)', fontsize=18)
    figname = subFolder + 'flux_vs_time.jpg'
    plt.savefig(figname, format='jpg', dpi=600, bbox_inches='tight')
    plt.show()
    
    print('Program Complete')
