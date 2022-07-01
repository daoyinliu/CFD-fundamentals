"""
Common CFD tools for Finite Volumn Method
Ref: 计算流体力学基础与应用, 东南大学出版社, 2021.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt as sqrt

# discretize 1d domain into uniform CVs, and clac. geometry variables
def meshing1dUniform(xmin,xmax,ncv):
    xFace = np.zeros((ncv+2))
    xh = (xmax - xmin) / ncv
    xFace[0] = xmin
    for i in range(1,ncv+1):
        xFace[i] = xFace[i-1] + xh
    xFace[ncv+1] = xFace[ncv]
    return xh, xFace

# calcualte geometry variables for control volumes
def calcGeometry1d(nxcv,xFace):
    xNode = np.zeros((nxcv+2))
    xFraSe = np.zeros((nxcv+2))
    volCV = np.zeros((nxcv+2))
    for i in range(1,nxcv+1):
        xNode[i] = 0.5 * (xFace[i-1] + xFace[i])
    xNode[0] = xFace[0]
    xNode[nxcv+1] = xFace[nxcv]
    for i in range(0,nxcv+1):
        xFraSe[i] = (xFace[i]-xNode[i])/(xNode[i+1]-xNode[i])
        volCV[i] = (xFace[i]-xFace[i-1])
    return xNode,xFraSe,volCV

# calcualte geometry variables for control volumes
def calcGeometry(nxcv,xFace,nycv,yFace,isAxis):
    xNode = np.zeros((nxcv+2))
    yNode = np.zeros((nycv+2))
    xFraSe = np.zeros((nxcv+2))
    yFraSn = np.zeros((nycv+2))
    R = np.zeros((nycv+2))
    Rp = np.zeros((nycv+2))     
    volCV = np.zeros((nxcv+2,nycv+2))
    for i in range(1,nxcv+1):
        xNode[i] = 0.5 * (xFace[i-1] + xFace[i])
    xNode[0] = xFace[0]
    xNode[nxcv+1] = xFace[nxcv]
    for j in range(1,nycv+1):
        yNode[j] = 0.5 * (yFace[j-1] + yFace[j])
    yNode[0] = yFace[0]
    yNode[nycv+1] = yFace[nycv]
    for i in range(0,nxcv+1):
        xFraSe[i] = (xFace[i]-xNode[i])/(xNode[i+1]-xNode[i])      
    for j in range(0,nycv+1):
        yFraSn[j] = (yFace[j]-yNode[j])/(yNode[j+1]-yNode[j])    
    if (isAxis==1): 
        R[:] = yFace[:]
        Rp[:] = yNode[:]
    else:
        R[:] = 1.
        Rp[:] = 1.
    for j in range(1,nycv+1):      
        for i in range(1,nxcv+1):                    
            volCV[i][j] = (yFace[j]-yFace[j-1])*(xFace[i]-xFace[i-1])*Rp[j]        
    return xNode,yNode,xFraSe,yFraSn,volCV,R,Rp
    
# TDMA for linear algebraic equation
# index of nodes (including boundaries): i=0:N+1  
# sovle values of fi at interior nodes: i=1:N
# mAp[i]*fi[i] + mAw[i]*fi[i-1] + mAe[i]*fi[i+1] = Q[i]
def TDMA1d(mAe, mAp, mAw, Q, fi, N):
    APt = np.zeros((N+2))
    Qt = np.zeros((N+2))

    # forward: modify mAp and Q, store modified values in APt and Qt
    APt[1] = mAp[1]
    Qt[1] = Q[1]
    for i in range(2,N+1):	
        t = - mAw[i] / APt[i-1]
        APt[i] = mAp[i] + t*mAe[i-1]
        if (abs(APt[i])<1e-30): print ("TDMA coef failed.")
        Qt[i] = Q[i] + t*Qt[i-1]

    # backwork: calculate fi
    fi[N] = Qt[N] / APt[N]
    for i in range(N-1,0,-1):	
        fi[i] = (Qt[i] - mAe[i]*fi[i+1]) / APt[i]

def SIPSOL2D(mAe, mAw, mAn, mAs, mAp, mSv, mFi, nxcv, nycv, epsi0):
    MAXINNERITER = 9999
    alfa = 0.92
    LW = np.zeros((nxcv+2,nycv+2)) 
    UE = np.zeros((nxcv+2,nycv+2)) 
    LS = np.zeros((nxcv+2,nycv+2)) 
    UN = np.zeros((nxcv+2,nycv+2))
    LPR = np.zeros((nxcv+2,nycv+2))
    RES = np.zeros((nxcv+2,nycv+2)) 

    P1 = 0.0
    P2 = 0.0
    for i in range(1,nxcv+1):
        for j in range(1,nycv+1):
            LW[i][j] = mAw[i][j] / (1. + alfa*UN[i-1][j])
            LS[i][j] = mAs[i][j] / (1. + alfa*UE[i][j-1])
            P1 = alfa*LW[i][j]*UN[i-1][j]
            P2 = alfa*LS[i][j]*UE[i][j-1]
            LPR[i][j] = 1. / (mAp[i][j]+P1+P2-LW[i][j]*UE[i-1][j] \
                             -LS[i][j]*UN[i][j-1]+1.0E-20)
            UN[i][j] = (mAn[i][j]-P1)*LPR[i][j]
            UE[i][j] = (mAe[i][j]-P2)*LPR[i][j]

    it = 0
    while (it < MAXINNERITER):
        it += 1
        resn = 0.0
        for i in range(1,nxcv+1):
            for j in range(1,nycv+1):
                RES[i][j] = mSv[i][j] \
                          - mAn[i][j]*mFi[i][j+1] - mAs[i][j]*mFi[i][j-1] \
                          - mAe[i][j]*mFi[i+1][j] - mAw[i][j]*mFi[i-1][j] \
                          - mAp[i][j]*mFi[i][j]  
                resn = resn + abs(RES[i][j]) 
                RES[i][j] = (RES[i][j]-LS[i][j]*RES[i][j-1] \
                            -LW[i][j]*RES[i-1][j])*LPR[i][j]
        if(it == 1): res0 = resn 
        rsm = resn / (res0 + 1.0e-20)
        # CALCULATE INCREMENT AND CORRECT VARIABLE		
        for i in range(nxcv,0,-1):
            for j in range(nycv,0,-1):
                RES[i][j] = RES[i][j]-UN[i][j]*RES[i][j+1]-UE[i][j]*RES[i+1][j] 
                mFi[i][j] = mFi[i][j]+RES[i][j]
        #print ("  inner it: %4d \trsm:%e" %(it, rsm))
        if(rsm < epsi0): break

#    if (it >= MAXINNERITER):
#        print ("SIPSOL2D solver reached MAXINNERITER.")
#    else:
#        print ("SIPSOL2D solver: converged at = %4d \trsm:%e" %(it,rsm))

    del LW, UE, LS, UN, LPR, RES
    return res0
    
# plot y=f(x) curve
def plotCurve(x, y, axisrange, figname, figtitle, xlabel, ylabel):
    fig = plt.figure(figsize = (10, 7))
    #plt.plot(x, y, marker = 'o', lw = 2)
    plt.plot(x, y, 'b', linewidth=2.0)
    plt.xlim(axisrange[0], axisrange[1])
    plt.ylim(axisrange[2], axisrange[3])
    plt.title(figtitle,fontsize=20)
    plt.legend(fontsize=20)
    plt.xlabel(xlabel,fontsize=20)
    plt.ylabel(ylabel,fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20) 
    figname += ".jpg"
    plt.savefig(figname, dpi=600, format='jpg', bbox_inches='tight')
#    plt.show()
#    plt.close(fig)  
    print ("saved = %s" %(figname))

# plot values of fi along x direction
def plot1dCompareFi(xNode,fi,fi_exact,axisrange,figname,figtitle):
    fig = plt.figure(figsize = (6, 4))
    plt.plot(xNode, fi, marker = 'o', lw = 2, label = "solution")
    plt.plot(xNode, fi_exact, 'r', label = "exact")
    plt.xlim(axisrange[0], axisrange[1])
    plt.ylim(axisrange[2], axisrange[3])
    plt.title(figtitle,fontsize=15)
    plt.legend(fontsize=15)
    plt.xlabel('x',fontsize=15)
    plt.ylabel('$\phi$',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    figname += ".jpg"
    plt.savefig(figname, dpi=600, format='jpg', bbox_inches='tight')
    plt.show()
    plt.close(fig)  
    print ("saved at = %s" %(figname))

# plot p=f(x,y) countor
def userContour2D(x, y, p, figname, figtitle, size):
    X, Y = np.meshgrid(x, y)
    colorinterpolation = 10
    fig = plt.figure(figsize=size)
#    fig = plt.figure(figsize=(10,3))
    
    C = plt.contour(X, Y, np.flipud(np.rot90(p)), colors='black', linewidths=0.75)
    plt.clabel(C, inline_spacing=5, fmt='%.1f', fontsize=15)
    # cmap = 'Pastel1', 'rainbow'
    plt.contourf(X, Y, np.flipud(np.rot90(p)), colorinterpolation, cmap='rainbow')
    plt.colorbar()

#    plt.figure('Contour', facecolor='lightgray')
#    plt.contour(X, Y, np.flipud(np.rot90(p)), colorinterpolation, colors='black', linewidths=0.75)
#    plt.imshow(np.flipud(np.rot90(p)), cmap='jet', origin='lower')
#    plt.colorbar()
    
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
#    plt.xlabel('x',fontsize=15)
#    plt.ylabel('y',fontsize=15)
    plt.title(figtitle,fontsize=18)
    plt.savefig(figname, dpi=600, format='jpg', bbox_inches='tight')
#    plt.show()
#    plt.close(fig)     
    print ("saved = %s" %(figname))
        
# plot p=f(x,y) 3D surface
def userSurface3D(x, y, p, figname, figtitle):
    fig = plt.figure(figsize=(8,7))
    ax3d = Axes3D(fig)
    X, Y = np.meshgrid(x, y)
    ax3d.plot_surface(X, Y, np.flipud(np.rot90(p)), cmap=cm.coolwarm, edgecolor='none')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('x',fontsize=15)
    plt.ylabel('y',fontsize=15)
    plt.title(figtitle)
    plt.savefig(figname, dpi=600, format='jpg', bbox_inches='tight')
#    plt.show()
#    plt.close(fig)
    print ("saved = %s" %(figname)) 
    
# save field data as Tecplot format
def exportToTecplot(tecfile,curTime,xNode,yNode,mVelX,mVelY,mPre,\
                    UDS,titleUDS,numUDS,nxcv,nycv):    
    if (curTime<=0):        
        tecfile.write('TITLE=flow field\n');
        variableFormat = 'VARIABLES=X,Y,U,V,Vmag,Pres'
        for i in range(0,numUDS):
            variableFormat = variableFormat + ',' + titleUDS[i]
        tecfile.write(variableFormat)
        tecfile.write('\n')
    # write data
    tecfile.write('ZONE T=\"%6.3e s\", I=%d, J=%d, DATAPACKING=BLOCK, \
      VARLOCATION=([1-%d]=NODAL), STRANDID=%d, SOLUTIONTIME=%6.3e\n' \
         %(curTime, nxcv+2, nycv+2, 6+numUDS, 1, curTime))
    for j in range(0,nycv+2):
        for i in range(0,nxcv+2):
            tecfile.write('%12.3e\t' %(xNode[i]))
        tecfile.write('\n')
    tecfile.write('\n')     
    for j in range(0,nycv+2):
        for i in range(0,nxcv+2):
            tecfile.write('%12.3e\t' %(yNode[j]))
        tecfile.write('\n')
    tecfile.write('\n')
    for j in range(0,nycv+2):
        for i in range(0,nxcv+2):
            tecfile.write('%12.3e\t' %(mVelX[i][j]))
        tecfile.write('\n')
    tecfile.write('\n')             
    for j in range(0,nycv+2):
        for i in range(0,nxcv+2):
            tecfile.write('%12.3e\t' %(mVelY[i][j]))
        tecfile.write('\n')
    tecfile.write('\n')     
    for j in range(0,nycv+2):
        for i in range(0,nxcv+2):
            velMag = sqrt(mVelX[i][j]**2 + mVelY[i][j]**2)
            tecfile.write('%12.3e\t' %(velMag))
        tecfile.write('\n')
    tecfile.write('\n')                
    for j in range(0,nycv+2):
        for i in range(0,nxcv+2):
            tecfile.write('%12.3e\t' %(mPre[i][j]))
        tecfile.write('\n')
    tecfile.write('\n') 
    for n in range(0,numUDS):
        for j in range(0,nycv+2):
            for i in range(0,nxcv+2):
                tecfile.write('%12.3e\t' %(UDS[i][j][n]))
            tecfile.write('\n')
        tecfile.write('\n')            

# construct coefficients based on convection-diffusion
def setCoeffsFromFlowDiffFlux(Fe,Fw,Fn,Fs,De,Dw,Dn,Ds,adveScheme):
    Ae = 0.
    An = 0.
    Aw = 0.
    As = 0.
    if(adveScheme == 1):
        Ae = De - Fe/2
        An = Dn - Fn/2
        Aw = Dw + Fw/2
        As = Ds + Fs/2
    elif(adveScheme == 2):
        Ae = De + max(-Fe,0.)
        An = Dn + max(-Fn,0.)
        Aw = Dw + max(Fw,0.)
        As = Ds + max(Fs,0.)
    elif(adveScheme == 3):
        Ae = De + max(-Fe,De-Fe/2,0.)
        An = Dn + max(-Fn,Dn-Fn/2,0.)
        Aw = Dw + max(Fw,Dw+Fw/2,0.)
        As = Ds + max(Fs,Ds+Fs/2,0.)
    elif(adveScheme == 4):
        s = pow(1-0.1*abs(Fe)/De, 5.)
        Ae = De*max(s,0) + max(-Fe,0.)
        s = pow(1-0.1*abs(Fn)/Dn, 5)
        An = Dn*max(s,0) + max(-Fn,0.)
        s = pow(1-0.1*abs(Fw)/Dw, 5.)
        Aw = Dw*max(s,0) + max(Fw,0.)
        s = pow(1-0.1*abs(Fs)/Ds, 5.)
        As = Ds*max(s,0) + max(Fs,0.)
    else:
        Ae = De - Fe/2
        An = Dn - Fn/2
        Aw = Dw + Fw/2
        As = Ds + Fs/2
    return Ae, An, Aw, As    

# discretize 1d domain into non-uniform CVs, and clac. geometry variables
def meshing1Dnonuniform():
    # generate grid lines
    xFace = np.zeros((10000))
    numSub = int(input("NUMBER OF SUBDOMAINS: "))
    for iSub in range(0,numSub):
        if (iSub==0):
            xBegin, xhStart, xhExpasion, xhLimit, xEnd = \
            input("ENTER 1st Subdomain > xBegin, xhStart, xhExpasion, xhLimit, xEnd: ").split(' ')
            xBegin = float(xBegin)
            xhStart = float(xhStart)
            xhExpasion = float(xhExpasion)
            xhLimit = float(xhLimit)
            xEnd = float(xEnd)
            xid = 0
            xFace[xid] = xBegin
            xh = xhStart
            while True:
                xh *= xhExpasion
                if (xhExpasion<=1): 
                    if (xh<xhLimit): xh = xhLimit
                else: 
                    if (xh>xhLimit): xh = xhLimit                   
                xid += 1
                xFace[xid] = xFace[xid-1] + xh
                if (xFace[xid]>=xEnd): break
            #correction
            xOver = xFace[xid] - xEnd
            if (abs(xhExpasion-1)>1e-10):
                xd = xOver*(1-xhExpasion)/(1-xhExpasion**xid)
                for i in range(1,xid+1):
                    xFace[i] = xFace[i] - xd*(1-xhExpasion**i)/(1-xhExpasion)                
            else:                
                xOverEach = xOver/xid 
                for i in range(1,xid+1):
                    xFace[i] = xFace[i] - i*xOverEach
        else:
            xhExpasion, xhLimit, xEnd = \
            input("ENTER next Subdomain > xhExpasion, xhLimit, xEnd: ").split(' ')
            xhExpasion = float(xhExpasion)
            xhLimit = float(xhLimit)
            xEnd = float(xEnd)
            xidPreSub = xid
            xid += 1
            xFace[xid] = xFace[xid-1] + xh            
            while True:
                xh *= xhExpasion
                if (xhExpasion<=1): 
                    if (xh<xhLimit): xh = xhLimit
                else: 
                    if (xh>xhLimit): xh = xhLimit
                xid += 1
                xFace[xid] = xFace[xid-1] + xh
                if (xFace[xid]>xEnd): break
            #correction
            xOver = xFace[xid] - xEnd
            if (abs(xhExpasion-1)>1e-10):
                xd = xOver*(1-xhExpasion)/(1-xhExpasion**xid)
                for i in range(1,xid+1):
                    xFace[i] = xFace[i] - xd*(1-xhExpasion**i)/(1-xhExpasion)                
            else:                
                xOverEach = xOver/xid 
                for i in range(1,xid+1):
                    xFace[i] = xFace[i] - i*xOverEach
    # CVs and Nodes        
    ncv = xid 
    xFace = xFace[0:xid+1]
    xRatio = np.zeros((ncv+1))
    for i in range(1,ncv):
        xRatio[i] = (xFace[i+1]-xFace[i]) /(xFace[i]-xFace[i-1])    
    xNode = np.zeros((ncv+2))
    xFraSe = np.zeros((ncv+1))
    for i in [1,-1]:
        xNode[i] = xFace[i]
    for i in range(1,ncv+1):
        xNode[i] = 0.5 * (xFace[i-1] + xFace[i])
    for i in range(0,ncv+1):
        xFraSe[i] = (xFace[i] - xNode[i]) / (xNode[i+1] - xNode[i])
    return ncv, xFace, xNode, xFraSe

def showgrid2D(nxcv, xFace, xNode, nycv, yFace, yNode, figname):
    fig = plt.figure(figsize=(9, 7))
    # grid lines
    for i in range(0,nxcv+1):
        pt1x = xFace[i]
        pt1y = yFace[0]   
        pt2x = xFace[i]
        pt2y = yFace[nycv]
        plt.plot([pt1x,pt2x],[pt1y,pt2y],'b')
    for j in range(0,nycv+1):
        pt1x = xFace[0]
        pt1y = yFace[j]   
        pt2x = xFace[nxcv]
        pt2y = yFace[j]
        plt.plot([pt1x,pt2x],[pt1y,pt2y],'b')
    # cv nodes    
    for i in range(0,nxcv+2):
        for j in range(0,nycv+2):    
            pt1x = xNode[i]
            pt1y = yNode[j]   
            pt2x = xNode[i]
            pt2y = yNode[j]
            plt.plot([pt1x,pt2x],[pt1y,pt2y],'ro')      
    plt.savefig(figname, dpi=600, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print ("saved = %s" %(figname))   
