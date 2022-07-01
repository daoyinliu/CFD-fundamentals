"""
2D (Un)steady incompressible laminar flow
SIMPLE Algorithm on Uniform Colocated Cartesian Grids
Ref: 计算流体力学基础与应用, 东南大学出版社, 2021.
# CAES description:  PIPE flow
"""

import sys, os, time, platform
sys.path.append("..")
if (platform.system()=='Linux'):
    import matplotlib
    matplotlib.use('Agg')
from cfdbooktools import \
    meshing1dUniform, calcGeometry, SIPSOL2D, exportToTecplot, userContour2D
import numpy as np
from math import sqrt as sqrt
import matplotlib.pyplot as plt
import scipy.io as scio
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lgmres as linsolver
# NOTE: the solver "lgmres" is faster than the solver "SIPSOL2D"

# ------------------------------ global variables------------------------------
### problme state
subFolder = "results_case2/"
problemTitle = "pipe_smiplecol"
isAxisXSymmetry = 1
isUnsteady = 0

### geometry setting
xwest = 0.; xeast = 0.5; ysouth = 0.; ynorth = 0.1
nxcv = int(input("Please enter number of CV x direction (nxcv=50): "))
nycv = int(input("Please enter number of CV y direction (nycv=10): "))

### physical setting
RE = 10.; rho_ref = 1.; uIN = 1.
vis_ref = rho_ref*uIN*(xeast-xwest)/RE
nu = vis_ref/rho_ref
global flowIN, flowOUT

### numerical setting
if (isUnsteady == 1):
    totalTimeSteps = 20; ntSave = 1; timeStep = 0.1; maxOutIter = 100
else:
    totalTimeSteps = 1; ntSave = 1; timeStep = 1.0e20; maxOutIter = 1000
curTime = 0.
iUEqu = 0; iVEqu = 1; iPEqu = 2
blendCDS = [1, 1, 1]
urf = [0.8, 0.8, 0.2] # relaxation factors for U-V-P
epsi = [1e-3, 1e-3, 1e-3] 
sourceMAX = 1.0e-4
resor = np.zeros((4))
idMon = [5,5]
idPreRef = [1,1]
numUDS = 5
titleUDS = ("stream","vortex","temp","k","epsi")
global tecFileID

### CREATE mesh array
xh, xFace = meshing1dUniform(xwest,xeast,nxcv)
yh, yFace = meshing1dUniform(ysouth,ynorth,nycv)
xNode,yNode,xFraSe,yFraSn,DeltaV,Rf,Rp = \
    calcGeometry(nxcv,xFace,nycv,yFace,isAxisXSymmetry)

### CREATE filed variable array
UX = np.zeros((nxcv+2,nycv+2))
UY = np.zeros((nxcv+2,nycv+2))
P = np.zeros((nxcv+2,nycv+2))
UX0 = np.zeros((nxcv+2,nycv+2))
UY0 = np.zeros((nxcv+2,nycv+2))
P0 = np.zeros((nxcv+2,nycv+2))
Pcor = np.zeros((nxcv+2,nycv+2))  # pressure correction
UDS = np.zeros((nxcv+2,nycv+2,numUDS))
RHO = np.zeros((nxcv+2,nycv+2))
VIS = np.zeros((nxcv+2,nycv+2))

### CREATE coefficients of linear algebraic matrix
AW = np.zeros((nxcv+2,nycv+2))
AP = np.zeros((nxcv+2,nycv+2))
AE = np.zeros((nxcv+2,nycv+2))
AS = np.zeros((nxcv+2,nycv+2))
AN = np.zeros((nxcv+2,nycv+2))
SC = np.zeros((nxcv+2,nycv+2))
SU = np.zeros((nxcv+2,nycv+2))
SV = np.zeros((nxcv+2,nycv+2))
APU = np.zeros((nxcv+2,nycv+2))
APV = np.zeros((nxcv+2,nycv+2))
DPX = np.zeros((nxcv+2,nycv+2))
DPY = np.zeros((nxcv+2,nycv+2))
mdotX = np.zeros((nxcv+2,nycv+2))
mdotY = np.zeros((nxcv+2,nycv+2))

### for the sparse solver: AX = B
tmpA = np.zeros((nxcv*nycv,nxcv*nycv))
tmpX0 = np.zeros((nxcv*nycv,1))
tmpB = np.zeros((nxcv*nycv,1))
RES = np.zeros((nxcv*nycv,1))

# -------------------------------- functions-----------------------------------
def initializeFlowField():
    RHO[:,:] = rho_ref
    VIS[:,:] = vis_ref
    UX[:,:] = 0.
    UY[:,:] = 0.
    P[:,:] = 0.
    
def openAndSaveResultsHead():
    global tecFileID    
    filename = "RE%04dNX%04dNY%04d_tec.dat" %(int(RE), nxcv, nycv)
    if (isUnsteady == 1): filename = subFolder + problemTitle + "_transt_" + filename
    else: filename = subFolder + problemTitle + "_steady_" + filename 
    if not os.path.exists(subFolder):
        os.makedirs(subFolder)  
    tecFileID = open(filename, mode='w')
    exportToTecplot(tecFileID,curTime,xNode,yNode,UX,UY,P,\
                    UDS,titleUDS,numUDS,nxcv,nycv)

def set1stBC():
    global flowIN
    # west
    UX[0,1:-1] = uIN
    UY[0,1:-1] = 0.
    flowIN = 0.
    for j in range(1,nycv+1): 
        mdotX[0,j] = RHO[0,j]*UX[0,j]*(yFace[j]-yFace[j-1])*Rp[j]
        flowIN += mdotX[0,j]

def modifyUVEquCoefsBCneighbour():
    # west, IN, (mass flux <> F, D <>0 )            
    for j in range(1,nycv+1):
        D = VIS[0][j]*(yFace[j]-yFace[j-1])*Rp[j]/(xNode[1]-xNode[0])    
        awc = D + mdotX[0][j]
        APU[1][j] += awc
        APV[1][j] += awc
        SU[1][j] += awc*UX[0][j]
        SV[1][j] += awc*UY[0][j]                    
    # east, OUT
    AE[-2,1:-1] = 0.           
    # south, Symmetry (mass flux = 0, D <> 0)      
    for i in range(1,nxcv+1): 
        D = VIS[i][0]*(xFace[i]-xFace[i-1])*Rf[0]/(yNode[1]-yNode[0])
        APV[i][1] += D
    # north, no-slip wall (mass flux = 0, D <> 0)         
    for i in range(1,nxcv+1): 
        D = VIS[i][-1]*(xFace[i]-xFace[i-1])*Rf[-2]/(yNode[-1]-yNode[-2])
        APU[i][-2] += D
        SU[i][-2] += D*UX[i][-1]

def updateUVBC():
    # east, OUT
    global flowOUT        
    flowOUT = 0.
    for j in range(1,nycv+1):
        mdotX[-2,j] = RHO[-2,j]*UX[-2,j]*(yFace[j]-yFace[j-1])*Rp[j]
        flowOUT += mdotX[-2,j]
    fac = flowIN/(flowOUT+1.0e-30)
    mdotX[-2,1:-1] *= fac
    UX[-1,1:-1] = UX[-2,1:-1] * fac
    print('   flowIN=%6.3e, flowOUT=%6.3e' %(flowIN,flowOUT))     
    # south, Symmetry
    UX[:,0] = UX[:,1]

def extrapolatPressureBC(pb):
    # set boundary pressure (linear extrapolation from inside)
    for i in range(1,nxcv+1):
        pb [i][0] = pb [i][1]+(pb [i][1]-pb [i][2])*yFraSn[1]
        pb[i][-1] = pb[i][-2]+(pb[i][-2]-pb[i][-3])*(1-yFraSn[-3])
    for j in range(1,nycv+1):
        pb [0][j] = pb [1][j]+(pb [1][j]-pb [2][j])*xFraSe[1]
        pb[-1][j] = pb[-2][j]+(pb[-2][j]-pb[-3][j])*(1-xFraSe[-3])
        
def assemblyAndSolveUVstarEqu():
    urfu = 1./urf[iUEqu]
    urfv = 1./urf[iVEqu]

    extrapolatPressureBC(P)
    
    SU[:,:] = 0.
    SV[:,:] = 0.
    SC[:,:] = 0.
    APU[:,:] = 0.
    APV[:,:] = 0.
                    
    # coefficient contributed by flux through CV faces: east/west
    for i in range(1,nxcv):
        fxe = xFraSe[i]
        fxP = 1. - fxe
        for j in range(1,nycv+1):
            De = VIS[i][j]*((yFace[j]-yFace[j-1])*Rp[j])/(xNode[i+1] - xNode[i])
            CE = min(mdotX[i][j],0.)
            CP = max(mdotX[i][j],0.)
            AE[i][j] =    CE - De
            AW[i+1][j] = -CP - De
            # source term contributed at P and E due to deferred correction 
            fuuds = CP*UX[i][j] + CE*UX[i+1][j]
            fvuds = CP*UY[i][j] + CE*UY[i+1][j]
            fucds = mdotX[i][j] * (fxe*UX[i+1][j] + fxP*UX[i][j])
            fvcds = mdotX[i][j] * (fxe*UY[i+1][j] + fxP*UY[i][j])
            SU[i][j] += blendCDS[iUEqu]*(fuuds-fucds)
            SU[i+1][j] -= blendCDS[iUEqu]*(fuuds-fucds)
            SV[i][j] += blendCDS[iUEqu]*(fvuds-fvcds)
            SV[i+1][j] -= blendCDS[iUEqu]*(fvuds-fvcds)

    # coefficient contributed by flux through CV faces: north/south
    for j in range(1,nycv):
        fyn = yFraSn[j]
        fyP = 1. - fyn
        for i in range(1,nxcv+1):
            Dn = VIS[i][j]*(xFace[i]-xFace[i-1])*Rf[j]/(yNode[j+1] - yNode[j])
            CN = min(mdotY[i][j],0.)
            CP = max(mdotY[i][j],0.)
            AN[i][j] =    CN - Dn
            AS[i][j+1] = -CP - Dn
            # source term contributed at P and N due to deferred correction  
            fuuds = CP*UX[i][j] + CN*UX[i][j+1]
            fvuds = CP*UY[i][j] + CN*UY[i][j+1]
            fucds = mdotY[i][j] * (fyn*UX[i][j+1] + fyP*UX[i][j])
            fvcds = mdotY[i][j] * (fyn*UY[i][j+1] + fyP*UY[i][j])
            SU[i][j] += blendCDS[iVEqu]*(fuuds-fucds)
            SU[i][j+1] -= blendCDS[iVEqu]*(fuuds-fucds)
            SV[i][j] += blendCDS[iVEqu]*(fvuds-fvcds)
            SV[i][j+1] -= blendCDS[iVEqu]*(fvuds-fvcds)
                    
    # source terms contributed by pressure gradient, unsteady term
    for i in range(1,nxcv+1):
        for j in range(1,nycv+1):
            pe = P[i+1][j]*xFraSe[i] + P[i][j]*(1.-xFraSe[i])
            pw = P[i][j]*xFraSe[i-1] + P[i-1][j]*(1.-xFraSe[i-1])
            pn = P[i][j+1]*yFraSn[j] + P[i][j]*(1.-yFraSn[j])
            ps = P[i][j]*yFraSn[j-1] + P[i][j-1]*(1.-yFraSn[j-1])
            DPX[i][j] = (pe-pw)/(xNode[i] - xNode[i-1])
            DPY[i][j] = (pn-ps)/(yNode[j] - yNode[j-1])
            SU[i][j] -= DPX[i][j]*DeltaV[i][j]
            SV[i][j] -= DPY[i][j]*DeltaV[i][j]
            if (isAxisXSymmetry==1):
                APV[i][j] += VIS[i][j]*DeltaV[i][j]/Rp[j]**2
            if (isUnsteady==1):
                apt = RHO[i][j]*DeltaV[i][j]*(1./timeStep)
                SU[i][j] += apt*UX0[i][j]
                SV[i][j] += apt*UY0[i][j]
                APU[i][j] += apt
                APV[i][j] += apt

    # boundary conditions
    modifyUVEquCoefsBCneighbour()
    
    # under-relaxation for u-velocity
    for i in range(1,nxcv+1):
        for j in range(1,nycv+1):
                AP[i][j] = (-AE[i][j]-AN[i][j]-AW[i][j]-AS[i][j]+APU[i][j])*urfu
                SC[i][j] = SU[i][j] + (1-urf[iUEqu])*AP[i][j]*UX[i][j]
                APU[i][j] = 1.0/AP[i][j]
                        
    resor[iUEqu] = SIPSOL2D(AE,AW,AN,AS,AP,SC,UX,nxcv,nycv,epsi[iUEqu])
    #resor[iUEqu] = CALL_LINEARSOLVE(AE,AW,AN,AS,AP,SC,UX,nxcv,nycv)
        
    # under-relaxation for v-velocity
    for i in range(1,nxcv+1):
        for j in range(1,nycv+1):
                AP[i][j] = (-AE[i][j]-AN[i][j]-AW[i][j]-AS[i][j]+APV[i][j])*urfv
                SC[i][j] = SV[i][j] +  (1-urf[iVEqu])*AP[i][j]*UY[i][j]
                APV[i][j] = 1.0/AP[i][j]
                
    resor[iVEqu] = SIPSOL2D(AE,AW,AN,AS,AP,SC,UY,nxcv,nycv,epsi[iVEqu])
    #resor[iVEqu] = CALL_LINEARSOLVE(AE,AW,AN,AS,AP,SC,UY,nxcv,nycv)

    updateUVBC()

def assemblyAndSolvePCorEqu():
    for i in range(1,nxcv):
        for j in range(1,nycv+1):
            Se = (yFace[j]-yFace[j-1])*Rp[j]
            vole = (xNode[i+1]-xNode[i])*Se
            apue = APU[i+1][j]*xFraSe[i] + APU[i][j]*(1.0-xFraSe[i])
            dpxe = (P[i+1][j]-P[i][j])/(xNode[i+1]-xNode[i])
            ue = UX[i+1][j]*xFraSe[i] + UX[i][j]*(1.0-xFraSe[i]) \
               - apue*vole*(dpxe-0.5*(DPX[i+1][j]+DPX[i][j]))
            mdotX[i][j] = RHO[i][j]*Se*ue
            AE[i][j] = -RHO[i][j]*Se*Se*apue
            AW[i+1][j] = AE[i][j]
            
    for i in range(1,nxcv+1):
        for j in range(1,nycv):
            Sn = (xFace[i]-xFace[i-1])*Rf[j]
            voln = (yNode[j+1]-yNode[j])*Sn
            apvn = APV[i][j+1]*yFraSn[j] + APV[i][j]*(1.-yFraSn[j])
            dpyn = (P[i][j+1]-P[i][j])/(yNode[j+1]-yNode[j])
            vn = UY[i][j+1]*yFraSn[j] + UY[i][j]*(1.-yFraSn[j]) \
               - apvn*voln*(dpyn-0.5*(DPY[i][j+1]+DPY[i][j]))
            mdotY[i][j] = RHO[i][j]*Sn*vn 
            AN[i][j] = -RHO[i][j]*Sn*Sn*apvn
            AS[i][j+1] = AN[i][j]
    
    # boundary conditions
    # no special treatment required
        
    # source term and coefficient of node P
    massErr = 0.
    for i in range(1,nxcv+1):
        for j in range(1,nycv+1):
                AP[i][j] = (-AE[i][j]-AN[i][j]-AW[i][j]-AS[i][j])
                SC[i][j] = mdotX[i-1][j] - mdotX[i][j] \
                          + mdotY[i][j-1] - mdotY[i][j] 
                Pcor[i][j] = 0.                
                massErr += abs(SC[i][j])
                
    resor[iPEqu] = SIPSOL2D(AE,AW,AN,AS,AP,SC,Pcor,nxcv,nycv,epsi[iPEqu])
    #resor[iPEqu] = CALL_LINEARSOLVE(AE,AW,AN,AS,AP,SC,Pcor,nxcv,nycv)

    # set boundary pressure correction (linear extrapolation from inside)
    extrapolatPressureBC(Pcor)
    
    return massErr

def correctUVPandMassFlux():
    for i in range(1,nxcv):
        for j in range(1,nycv+1):
            mdotX[i][j] += AE[i][j]*(Pcor[i+1][j]-Pcor[i][j])
    for i in range(1,nxcv+1):
        for j in range(1,nycv):
            mdotY[i][j] += AN[i][j]*(Pcor[i][j+1]-Pcor[i][j])
                    
    pre_ref = Pcor[idPreRef[0]][idPreRef[1]]    
    for i in range(1,nxcv+1):
        for j in range(1,nycv+1):
            ppe = Pcor[i+1][j]*xFraSe[i] + Pcor[i][j]*(1.-xFraSe[i])
            ppw = Pcor[i][j]*xFraSe[i-1] + Pcor[i-1][j]*(1.-xFraSe[i-1])
            ppn = Pcor[i][j+1]*yFraSn[j] + Pcor[i][j]*(1.-yFraSn[j])
            pps = Pcor[i][j]*yFraSn[j-1] + Pcor[i][j-1]*(1.-yFraSn[j-1])
            UX[i][j] -= (ppe-ppw)*(yNode[j]-yNode[j-1])*Rp[j]*APU[i][j]
            UY[i][j] -= (ppn-pps)*(xNode[i]-xNode[i-1])*Rp[j]*APV[i][j]
            P[i][j] += urf[iPEqu] * (Pcor[i][j]-pre_ref)

def calcStreamVortex():    
    # stream function 
    UDS[0,0,0] = 0.
    for i in range(0,nxcv+2):
        if (i>0):
            UDS[i,0,0] = UDS[i-1,0,0] + RHO[i][0]*UY[i-1][0]*(xNode[i]-xNode[i-1])
        for j in range(1,nycv+2):
            UDS[i,j,0] = UDS[i,j-1,0] - RHO[i][j]*UX[i][j-1]*(yNode[j]-yNode[j-1])
    # vortex
    for i in range(1,nxcv+1):
      for j in range(1,nycv+1):
        UDS[i,j,1] = 0.5*(UY[i+1,j+1]-UY[i,j+1]+UY[i+1,j]-UY[i,j])/(xNode[i+1]-xNode[i]) \
                    - 0.5*(UX[i+1,j+1]-UX[i+1,j]+UX[i,j+1]-UX[i,j])/(yNode[j+1]-yNode[j])

def CALL_LINEARSOLVE(AE,AW,AN,AS,AP,SC,X0,nxcv,nycv):
    for i in range(1,nxcv+1):
        for j in range(1,nycv+1):
            ij = (j-1) + (i-1)*nycv
            #print(i,j,ij)
            ijS = ij - 1
            ijN = ij + 1
            ijW = ij - nycv
            ijE = ij + nycv
            if j>1: tmpA[ij,ijS] = AS[i,j]
            if j<nycv: tmpA[ij,ijN] = AN[i,j]
            if i>1: tmpA[ij,ijW] = AW[i,j]
            if i<nxcv: tmpA[ij,ijE] = AE[i,j]
            tmpA[ij,ij] = AP[i,j]
            tmpB[ij] = SC[i,j]
            tmpX0[ij] = X0[i,j]
    csr = csr_matrix(tmpA)
    #indices = csr.indices
    #indptr = csr.indptr
    #val = csr.data
    tmpX, exitCode = linsolver(csr, tmpB, tmpX0, atol=1.0e-7)
    for ij in range(0,nxcv*nycv):
        i = (ij) // nycv + 1
        j = (ij) % nycv + 1
        #print(i,j,ij)
        X0[i,j] = tmpX[ij]

    res0 = 0.
    for ij in range(0,nxcv*nycv):
        RES[ij] = np.dot(tmpA[ij,:],tmpX0) - tmpB[ij]
        res0 += np.abs(RES[ij])
    return res0

#--------------------------------main program----------------------------------
if __name__ == "__main__":
    startCPUTime = time.time()
    global tecFileID
    ### initial setting
    initializeFlowField()
    openAndSaveResultsHead()
    
    ### begin of time loop
    totIter = 0
    iterHistCount = []
    iterHistMassErr = []
    iterHistUxErr = []
    iterHistUyErr = []
    for timeIter in range(1,totalTimeSteps+1):
        curTime += timeStep
        print ("time=%6.3e,pres=%6.3e" %(curTime,P[idMon[0]][idMon[1]]))
        if (isUnsteady==1):
            UX0[:,:] = UX[:,:]
            UY0[:,:] = UY[:,:]
            P0[:,:] = P[:,:]
        set1stBC()
        # outer iteration
        for outIter in range(1,maxOutIter+1):
            assemblyAndSolveUVstarEqu()
            massErr = assemblyAndSolvePCorEqu()
            correctUVPandMassFlux()
            # check convergence
            totIter += 1
            print ("it=%3d, massErr=%6.2e, res(V,p)=%4.2e,%4.2e,%4.2e" \
                  %(outIter,massErr,resor[0],resor[1],resor[2]))
            maxerr = max(massErr,max(resor))
            iterHistCount.append(totIter)
            iterHistMassErr.append(massErr)
            iterHistUxErr.append(resor[0])
            iterHistUyErr.append(resor[1])
            if (massErr<sourceMAX):
                print ("converged at iter: %d, massErr=%8.5e" %(outIter,massErr))
                break        
        # converged
        if ((isUnsteady==1 and timeIter%ntSave == 0) or (isUnsteady==0)):
            calcStreamVortex()
            exportToTecplot(tecFileID,curTime,xNode,yNode,UX,UY,P,\
                            UDS,titleUDS,numUDS,nxcv,nycv)

    ### after time loop
    tecFileID.close()
    endCPUTime = time.time()
# plot and save residual error
    fig = plt.figure(figsize = (6, 4.5))
    plt.plot(iterHistCount, iterHistMassErr, linestyle='-', color='b', label = "continuity")
    plt.plot(iterHistCount, iterHistUxErr, linestyle='--', color='r', label = "mom x")
    plt.plot(iterHistCount, iterHistUyErr, linestyle='-.', color='c', label = "mom y")
    plt.gca().legend(loc=0, numpoints=1)
    plt.xlabel('Iter', fontsize=18)
    plt.ylabel('Residual error', fontsize=18)
    figname = subFolder + 'residual_error.jpg'
    plt.savefig(figname, format='jpg', dpi=600, bbox_inches='tight')
    plt.show()
    errorHisFileName = "error_history.csv"
    errorHisFileName = subFolder + problemTitle + errorHisFileName
    errMat = np.vstack((np.array(iterHistCount), np.array(iterHistMassErr)))
    errMat = np.vstack((errMat, np.array(iterHistUxErr)))
    errMat = np.vstack((errMat, np.array(iterHistUyErr)))
    np.savetxt(fname = errorHisFileName, X = errMat.transpose(), encoding='utf-8')
# save y-ux profile
    uxfilename = "Re%04d_CV%04d.csv" %(int(RE), nycv)
    fullname = subFolder + problemTitle + "_ux_" + uxfilename
    tmpData = np.hstack((yNode.reshape(-1, 1),UX[int(nxcv*1/5)].reshape(-1, 1)))
    tmpData = np.hstack((tmpData,UX[int(nxcv*2/5)].reshape(-1, 1)))
    tmpData = np.hstack((tmpData,UX[int(nxcv*3/5)].reshape(-1, 1)))
    tmpData = np.hstack((tmpData,UX[int(nxcv*4/5)].reshape(-1, 1)))
    np.savetxt(fname = fullname, X=tmpData, encoding='utf-8')
# save x-uy profile
    uyfilename = "Re%04d_CV%04d.csv" %(int(RE), nycv)
    fullname = subFolder + problemTitle + "_uy_center_y_" + uyfilename
    tmpData = np.hstack((xNode.reshape(-1, 1),UY[:,int(nycv/2)].reshape(-1, 1)))
    np.savetxt(fname = fullname, X=tmpData, encoding='utf-8')
# save 2d field results: PSI
    contourfigname = "_stream_Re%04d_CVx%04dCVy%04d.jpg" %(int(RE), nxcv, nycv)
    fullname = subFolder + problemTitle + contourfigname
    figtitle = "stream function"
    userContour2D(xNode, yNode, UDS[:,:,0], fullname, figtitle, [10,2])
    resultsFileName = "_stream_Re%04d_CVx%04dCVy%04d.mat" %(int(RE), nxcv, nycv)
    resultsFileName = subFolder + problemTitle + resultsFileName
    scio.savemat(resultsFileName, dict([('x', xNode), ('y', yNode), ('fi', UDS[:,:,0])]))
# save 2d field results: velMag
    for i in range(0,nxcv+2):
        for j in range(0,nycv+2):
            UDS[i,j,0] = np.sqrt(UX[i,j]**2 + UY[i,j]**2)
    contourfigname = "_velmag_Re%04d_CVx%04dCVy%04d.jpg" %(int(RE), nxcv, nycv)
    fullname = subFolder + problemTitle + contourfigname
    figtitle = "velocity magnitude"
    userContour2D(xNode, yNode, UDS[:,:,0], fullname, figtitle, [10,2])
    resultsFileName = "_velmag_Re%04d_CVx%04dCVy%04d.mat" %(int(RE), nxcv, nycv)
    resultsFileName = subFolder + problemTitle + resultsFileName
    scio.savemat(resultsFileName, dict([('x', xNode), ('y', yNode), ('fi', UDS[:,:,0])]))
    
    print('Program Complete; solver using %.1f seconds' % (endCPUTime - startCPUTime))