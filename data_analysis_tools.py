"""
  Tools for a data analysis (depot)
    usage:
         python data.py <switch> TODO
"""


import sys
import os
import numpy as np
from scipy import optimize
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from data import *


def fitLineAndLine( x, y, startTime = 1.0, endTime = 24.0 ):
    fph = FRAMES_PER_H
    startDp = int( round(startTime*fph) )
    endPoint = int( round(endTime*fph) )
    for dp in xrange( startDp, endPoint + 1 ):
        # a1*x + b1; a2*x + b2
        # a1*xp + b1 = a2*xp + b2
        xp = x[dp]
        x1 = x[:dp]
        x2 = x[dp:]
        y1 = y[:dp]
        y2 = y[dp:]
        A = np.zeros( [4,4] )
        sx21 = np.sum( np.power( x1, 2 ) )
        sx11 = np.sum( x1 )
        sx22 = np.sum( np.power( x2, 2 ) )
        sx12 = np.sum( x2 )
        A[0,:] = [ xp, 1, -xp, -1 ]
        A[1,:] = [ sx21, sx11, 0, 0 ]
        A[2,:] = [ sx11, dp, 0, 0 ]
        A[3,:] = [ 0, 0, sx22, sx12 ]
        B = np.zeros((4,1))
        syx1 = np.sum( y1*x1 )
        sy1 = np.sum( y1 )
        syx2 = np.sum( y2*x2 )
        B[:,0] = [ 0, syx1, sy1, syx2 ]
        
        cp = np.linalg.norm(A)*np.linalg.norm( np.linalg.inv(A) )
        print "Cp: ", cp
        
        X = np.linalg.solve(A, B)
        print A
        print B
        print X
        a1, b1, a2, b2 = np.reshape(X,-1)
        print a1, b1, a2, b2
        
        yp = np.zeros(len(y))
        pl1 = np.poly1d([a1, b1])
        pl2 = np.poly1d([a2, b2])
        yp[:dp] = pl1( x[:dp] )
        yp[dp:] = pl2( x[dp:] )
        ymean = np.sum(y)/len(y)
        ssreg = np.sum((yp-ymean)**2)
        sstot = np.sum((y - ymean)**2)
        rSquared = ssreg/sstot
        
        normdiffYYp = np.linalg.norm(y-yp)
        
        plt.figure()
        plt.plot(x,y,"k")
        plt.plot(x[:dp],yp[:dp],"r")
        plt.plot(x[dp:],yp[dp:],"g")
        plt.text(20, 8500, str(normdiffYYp), fontsize=12)
        #plt.show()
        plt.savefig("logs2/area/tmp/area_Plot_%003d" %dp, dpi=100)
        plt.clf
        plt.close()
        #sys.exit()


def fitPolyAndLine( x, y, line1, dPoint1, endTime, ver = "ver1" ):
    #a2*x^4 + b2*x^3 + c2*x^2 + d2*x +e
    #a3*x + b3
    fph = FRAMES_PER_H
    coeffsList = []
    rSquaredList = []
    dPoitList = []
    startD2Point = dPoint1 + 10
    endpoint = int( endTime * fph )
    a1 = line1[0]
    if ver == "ver1":
        xp1 = x[dPoint1]
    elif ver == "ver2":
        xp1 = 0
    
    for d2p in xrange( startD2Point, endpoint + 1 ):
        yl3 = y[d2p:]
        xl3 = x[d2p:]
        a3, b3 = np.polyfit(xl3, yl3, 1)
        
        xp2 = x[d2p]
        xp = x[ dPoint1 : d2p ]
        yp = y[ dPoint1 : d2p ]
        
        if ver == "ver1":
            A = np.zeros( (5,5) )
            A[0,:] = [ xp1**4, xp1**3, xp1**2, xp1, 1 ]
            A[1,:] = [ 4*xp1**3, 3*xp1**2, 2*xp1, 1, 0 ]
            A[2,:] = [ xp2**4, xp2**3, xp2**2, xp2, 1 ]
            A[3,:] = [ 4*xp2**3, 3*xp2**2, 2*xp2, 1, 0 ]
            #for nn in range(5):
            #    A[4,nn] = np.sum( np.power( xp, 8 - nn ) )
            for nn in range(5):
                A[4,nn] = np.sum( np.power( xp, 4 - nn ) )
            A[4,:] = A[4,:]/1000.0
            B = np.zeros((5,1))
            #sumX4Y = np.sum( np.power( xp, 4 ) * yp )
            sumX4Y = np.sum( yp )
            B[:,0] = [ a1*xp1, a1, a3*xp2 + b3, a3, sumX4Y/1000.0 ]
            
            #A = np.zeros( (4,4) )
            #A[0,:] = [ xp1**3, xp1**2, xp1, 1 ]
            #A[1,:] = [ 3*xp1**2, 2*xp1, 1, 0 ]
            #A[2,:] = [ xp2**3, xp2**2, xp2, 1 ]
            #A[3,:] = [ 3*xp2**2, 2*xp2, 1, 0 ]
            #B = np.zeros((4,1))
            #B[:,0] = [ a1*xp1, a1, a3*xp2 + b3, a3 ]
            
        elif ver == "ver2":
            A = np.zeros( (3,3) )
            A[0,:] = [ xp2**4, xp2**3, xp2**2 ]
            A[1,:] = [ 4*xp2**3, 3*xp2**2, 2*xp2 ]
            for nn in range(3):
                A[2,nn] = np.sum( np.power( xp, 6 - nn ) )
        B = np.zeros((3,1))
        sumX2Y = np.sum( np.power( xp, 2 ) * yp )
        B[:,0] = [ a3*xp2 + b3, a3, sumX2Y ]
        
        cp = np.linalg.norm(A)*np.linalg.norm( np.linalg.inv(A) )
        print "Cp: ", cp
        
        X = np.linalg.solve(A, B)
        print A
        print B
        print X
        coeffsP = np.reshape(X,-1)
        print coeffsP
        #sys.exit()
        plt.figure()
        if ver == "ver1":
            pl1 = np.poly1d( line1 )
            ypl1 = pl1( x[:dPoint1] )
            plt.plot(x[:dPoint1],ypl1,"b")
            pp2 = np.poly1d( coeffsP )
        elif ver == "ver2":
            a2, b2, c2 = coeffsP
            pp2 = np.poly1d( [a2, b2, c2, 0, 0] )
        
        pl3 = np.poly1d( (a3,b3) )
        ypp2 = pp2(xp)
        ypl3 = pl3( x[d2p:] )
        #ybar = np.sum(y)/len(y)
        #ssreg = np.sum((yp-ybar)**2)
        #sstot = np.sum((y - ybar)**2)
        #rSquared = ssreg/sstot
        #rSquaredList.append(rSquared)
        plt.plot(x,y,"k")
        plt.plot(xp,ypp2,"r")
        plt.plot(x[d2p:],ypl3,"g")
        #plt.show()
        plt.savefig("logs2/area/tmp/area_Plot_%003d" %d2p, dpi=100)
        plt.clf
        plt.close()
    
    sys.exit()
    #return coeffsPoly, coeffsLine2, dPoint2, rSquaredPL


def myFunctionFit( x, y ):
    #y = a*x^3 + b*x^2
    ATA = np.zeros((2,2))
    K = np.zeros((2,1))
    ATA[0,0] = np.sum( np.power(x, 6) )
    ATA[0,1] = np.sum( np.power(x, 5) )
    ATA[1,0] = np.sum( np.power(x, 5) )
    ATA[1,1] = np.sum( np.power(x, 4) )
    yx3 = np.multiply( y, np.power(x, 3) )
    yx2 = np.multiply( y, np.power(x, 2) )
    K[0,0] = np.sum( yx3 )
    K[1,0] = np.sum( yx2 )
    ab = np.linalg.solve(ATA, K)
    
    return ab


def searchLine( x, y, minTime = 0.5, constantB = None, timeLimit = 24.0 ):
    searchContinue = True
    widthR = 10
    fph = FRAMES_PER_H
    inp = int( round(minTime * fph) ) #initial number of point
    rSquaredList = []
    coeffsList = []
    ii = 0
    while searchContinue == True:
        inputX = x[ : inp + ii ]
        inputY = y[ : inp + ii ]
        if constantB == None:
            yp, rSquared, coeffs = dataPolyFit( inputX, inputY, 1, returnCoeffs = True )
            a, b = coeffs
            
        else:
            #constantB = 0
            b = 0
            sumXsq = np.sum( np.power( inputX, 2 ) )
            sumXY = np.sum( np.multiply( inputX, inputY ) )
            a = sumXY/sumXsq
            
            yp = np.zeros(len(inputY))
            p = np.poly1d( (a,b) )
            yp = p(inputX)
            yMean = np.mean(inputY)
            ssreg = np.sum((yp-yMean)**2)
            sstot = np.sum((inputY - yMean)**2)
            rSquared = ssreg/sstot
            
        rSquaredList.append(rSquared)
        coeffsList.append([a, b])
        
        if ii > timeLimit * fph:
            searchContinue = False
            print "ii > timeLimit * fph" #TODO
        
        if ii >= widthR - 1:
            ar, br = np.polyfit( np.arange(widthR), rSquaredList[-widthR:] , 1)
            #print ar
            if ar < 0:
                coeffsFinal = coeffsList[ -widthR ]
                rSquaredF = rSquaredList[ -widthR ]
                point = inp + ii - widthR
                endTime = float( point )/fph
                searchContinue = False
            
        ii += 1
        
    #print rSquaredList
    return coeffsFinal, endTime, rSquaredF, point


def extremsPoly3Type( x, coeffs ):
    a, b, c, d = coeffs
    yd2 = 6*a*x + 2*b
    if yd2 > 0:
        extremTyp = "min"
    elif yd2 < 0:
        extremTyp = "max"
    else:
        extremTyp = None
        
    return extremTyp


def extremsFromPoly3( coeffs ):
    a, b, c, d = coeffs
    #first derivative: y' = 3ax^2 + 2bx + c
    #formula: x1,2 = ( -2b +- sqrt( 4b^2 - 12ac ) ) / 6a
    #second derivative: y" = 6ax + 2b
    #formula: x = -b/(3a)
    
    if a == 0:
        x1 = -c/( 2*b )
        x2 = None
        xInf = None
        
    else:
        D = 4*b**2 - 12*a*c
        
        if D > 0: 
            x1 = ( -2*b + np.sqrt( D ) ) / (6*a)
            x2 = ( -2*b - np.sqrt( D ) ) / (6*a)
            
        elif D == 0:
            x1 = ( -2*b ) / (6*a)
            x2 = None
            
        else:
            x1 = None
            x2 = None
            
        xInf = -b/( 3*a )
    
    if x1:
        exTyp1 = extremsPoly3Type( x1, coeffs )
        if exTyp1:
            ex1F = [exTyp1, x1]
        else:
            ex1F = None
    else:
        ex1F = None
            
    if x2:
        exTyp2 = extremsPoly3Type( x2, coeffs )
        if exTyp2:
            ex2F = [exTyp2, x2]
        else:
            ex2F = None
    else:
        ex2F = None
    
    return ex1F, ex2F, xInf
    




if __name__ == "__main__": 
    if len(sys.argv) < 2:
        print __doc__
        sys.exit()
    
    print "TODO"
