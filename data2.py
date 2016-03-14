"""
  Tools for analyse of logsfiles
    usage:
         python data.py <switch> <directory> <measurementLabel>
            switch: efd
"""

import sys
import os
import numpy as np
from scipy import optimize, interpolate
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from data import *
from data_analysis_tools import *


IGNORE_LIST = []
EFD_COEFFICIENTS = [7, 11, 12, 15, 19, 20]
#EFD_COEFFICIENTS = [7]
DATA_OUTPUT = [ "plots_1d", "plots_2d", "plots_3a", "plots_3d", "plots_4d", "plots_5a" ]
DEGREE = 3
FRAMES_PER_H = 30.0
SMOOTH_NUM_AREA = 10
SMOOTH_NUM_AREA_DETAIL = 4
DETAIL_AREA = 6 #time in hours
OFFSET = 120
LEN_AREA_G1 = 40
LEN_AREA_G2 = 20 #30
LEN_AREA_G3 = 10

LEN_EFD_G1 = 120
LEN_EFD_G2 = 60
LEN_EFD_G3 = 30

#Fit model 5
GRID0 = ( -0.199, 0.201, 21 )
GRID1 = ( 0, 6.0, 301 )
GRID2 = ( 0, 2.0, 101 )
GRIDF = 101


createAreaPlots = False
createAreaPlotsGrad = False
createAreaDetail = False
areaAnalyse2 = True
createEfdsPlotsOrig = True


#def model5( x, S0, a, k1, k2, d, a0, k0 = 0.001 ):
#    return S0 + a + a/( k1 - k2 ) * ( ( k2 - k1*d )* np.exp(-k1*x ) - k1*( 1 - d )* np.exp(-k2*x ) ) + a0*( np.exp( k0*x ) -1 )
  
def model5( x, S0, a, k1, k2, d, a0 ):
    return S0 + a + a/( k1 - k2 ) * ( ( k2 - k1*d )* np.exp(-k1*x ) - k1*( 1 - d )* np.exp(-k2*x ) ) + a0*x
    
def model55( x, S0, a, k1, k2, d, a0, k0 ):
    return S0 + a + a/( k1 - k2 ) * ( ( k2 - k1*d )* np.exp(-k1*x ) - k1*( 1 - d )* np.exp(-k2*x ) ) + a0*( np.exp( -k0*x ) -1 )

def model5a( x, a, k1, k2 ):
    #print "model5a x: ", x
    return a*( 1 + 1/( k1-k2 ) * ( np.exp(-k1*x )*k2 - k1*np.exp(-k2*x ) ) )
    
def model5ad( x, ad, k1, k2 ):
    return ad*k1/( k1-k2 ) * ( np.exp(-k2*x ) - np.exp(-k1*x ) )
    
def model55a0( x, a0, k0 ):
    return a0*( np.exp( -k0*x ) -1 )
    
def model5a0( x, a0 ):
    return a0*x
    
def model5DI( x, a, k1, k2, d, a0 ):
    return -a*k1/( k1 -k2 ) * ( ( k2 - k1*d )* np.exp(-k1*x ) - k2*( 1 - d )* np.exp(-k2*x ) ) + a0
    
def model55DI( x, a, k1, k2, d, a0, k0 ):
    return -a*k1/( k1 -k2 ) * ( ( k2 - k1*d )* np.exp(-k1*x ) - k2*( 1 - d )* np.exp(-k2*x ) ) - a0*k0*np.exp(-k0*x )
    
def model5DII( x, a, k1, k2, d ):
    return -a*k1/( k1 -k2 ) * ( k1*( -k2 + k1*d )* np.exp(-k1*x ) + k2**2 *( 1 - d )* np.exp(-k2*x ) )

def model55DII( x, a, k1, k2, d, a0, k0 ):
    return -a*k1/( k1 -k2 ) * ( k1*( -k2 + k1*d )* np.exp(-k1*x ) + k2**2 *( 1 - d )* np.exp(-k2*x ) ) + a0*k0**2 *np.exp(-k0*x )


def removeNan( x, y ):
    newX = []
    newY = []
    for ii in xrange(len(y)):
        if np.isnan(y[ii]):
            #print y[ii]
            continue
        else:
            newX.append(x[ii])
            newY.append(y[ii])
            
    x = np.array(newX)
    y = np.array(newY)
    
    return x, y


def divideMatrix( myArr ):
    arrA = myArr.copy()
    arrB = myArr.copy()
    for ii in xrange( myArr.shape[0] ):
        arrA[ ii, ii:] = np.nan
        arrB[ ii, :ii] = np.nan

    return arrA, arrB
    

def getLocalMin( myAr ):
    r, c = myAr.shape
    localMinList = []
    minValueList = []
    step = 0.1
    for rr in xrange(r):
        if rr <= 1 or rr >= ( r - 2 ):
            continue
            
        for cc in xrange(c):
            if cc <= 1 or cc >= ( c - 2 ):
                continue
                
            x00 = myAr[rr,cc]
            if np.isnan(x00):
                continue
            localGrup = myAr[ rr-2:r+2, cc-2:cc+2 ]
            if x00 == np.nanmin( localGrup ):
                localMinList.append( [rr,cc] )
                minValueList.append( x00 )
        
        counter = rr/float(r)
        if counter >= step:
            print "Local min: ", round(step * 100), "%"
            step += 0.1
        
    if len( localMinList ) > 0:
        absMin = localMinList[ np.argmin( minValueList ) ]
        return localMinList, tuple( absMin ), minValueList
    else:
        return None, None, None
    
    


def fitModel5( x, y, seedId, gridK1, gridK2, sigma = None, oneSide = False ):
    k1f = gridK1[0]
    k1l = gridK1[1]
    k2f = gridK2[0]
    k2l = gridK2[1]
    k1n = gridK1[2]
    k2n = gridK2[2]
    k1Ar = np.linspace( k1f, k1l, k1n )
    k2Ar = np.linspace( k2f, k2l, k2n )
    diffYYpNormAr = np.zeros( ( k1n, k2n ) )
    coeffAr = np.zeros( ( k1n, k2n, 6 ) )
    solutionLog = None
    #newX = []
    #newY = []
    #for ii in xrange(len(y)):
        #if np.isnan(y[ii]):
            ##print y[ii]
            #continue
        #else:
            #newX.append(x[ii])
            #newY.append(y[ii])
            
    #x = np.array(newX)
    #y = np.array(newY)
    
    x, y = removeNan( x, y )
    
    ii = 0
    step = 0.1
    for k1 in k1Ar:
        jj = 0
        
        for k2 in k2Ar:
            a, d, a0, S0 = None, None, None, None
            diffYYpNorm = None
            if k2 <= 0 or k1 <= 0:
                diffYYpNormAr[ii,jj] = None
                coeffAr[ii,jj] = None
                jj += 1
                continue
                
            if k1 == k2:
                diffYYpNormAr[ii,jj] = None
                coeffAr[ii,jj] = None
                jj += 1
                continue
            
            if oneSide == True and k1 < k2:
                diffYYpNormAr[ii,jj] = None
                coeffAr[ii,jj] = None
                jj += 1
                continue
                
            partA = model5a( x, 1, k1, k2 )
            partAD = model5ad( x, 1, k1, k2 )
            #partA0 = model5a0( x, 1, k0 )
            partA0 = model5a0( x, 1 )
            partList = [ partA, partAD, partA0, np.ones( len(x) ) ]
            A = np.zeros( [4,4] )
            B = np.zeros((4,1))
            for rr in range(4):
                for cc in range(4):
                    A[rr, cc] =  np.sum( np.multiply( partList[rr], partList[cc] ) )
            for rr in range(4):
                B[rr, 0] = np.sum( np.multiply( y, partList[rr] ) )
            
            try:
                X = np.linalg.solve(A, B)
            except:
                if solutionLog is None:
                    slogName = "logs2/area/solutinLog"+str(seedId)+".txt"
                    solutionLog = open( slogName, "w")
                    solutionLog.write( "ii, jj, k1, k2\r\n" )
                    print "The solutinLog was created!"
                    
                solutionLog.write("%d, %d, %f, %f\r\n" % ( ii, jj, k1, k2 ) )
                diffYYpNormAr[ii,jj] = None
                coeffAr[ii,jj] = None
                jj += 1
                continue
                
            #print X
            a, ad, a0, S0 = X
            d = ad/a
            #coeffAr[ii,jj] = S0, a, k1, k2, d, a0, k0
            #yp = model5( x, S0, a, k1, k2, d, a0, k0 )
            coeffAr[ii,jj] = S0, a, k1, k2, d, a0
            yp = model5( x, S0, a, k1, k2, d, a0 )
            diff = y - yp
            if sigma is not None:
                diff = diff/sigma
            diffYYpNorm = np.linalg.norm( diff )
            #print diffYYpNorm
            diffYYpNormAr[ii,jj] = diffYYpNorm
            
            jj += 1
            #if ii > 20:
            #    sys.exit()
            
        ii += 1
        counter = ii/float(k1n)
        if counter >= step:
            print "Fit model: ", round(step * 100), "%"
            step += 0.1
    
    if solutionLog is not None:
        solutionLog.close()
    
    return diffYYpNormAr, coeffAr
    

def fitModel55( x, y, seedId, gridK1, gridK2, gridK0, sigma = None, oneSide = False ):#TODO
    k1f = gridK1[0]
    k1l = gridK1[1]
    k2f = gridK2[0]
    k2l = gridK2[1]
    k1n = gridK1[2]
    k2n = gridK2[2]
    k0f = gridK0[0]
    k0l = gridK0[1]
    k0n = gridK0[2]
    k1Ar = np.linspace( k1f, k1l, k1n )
    k2Ar = np.linspace( k2f, k2l, k2n )
    k0Ar = np.linspace( k0f, k0l, k0n )
    diffYYpNormAr = np.zeros( ( k0n, k1n, k2n ) )
    coeffAr = np.zeros( ( k0n, k1n, k2n, 7 ) )
    solutionLog = None
    x, y = removeNan( x, y )
    
    kk = 0
    for k0 in k0Ar:
        print ""
        print "k0 = ", k0
        ii = 0
        step = 0.1
        for k1 in k1Ar:
            jj = 0
            
            for k2 in k2Ar:
                a, d, a0, S0 = None, None, None, None
                diffYYpNorm = None
                if k2 <= 0 or k1 <= 0:
                    diffYYpNormAr[kk,ii,jj] = None
                    coeffAr[kk,ii,jj] = None
                    jj += 1
                    continue
                    
                if k1 == k2:
                    diffYYpNormAr[kk,ii,jj] = None
                    coeffAr[kk,ii,jj] = None
                    jj += 1
                    continue
                
                if oneSide == True and k1 < k2:
                    diffYYpNormAr[kk,ii,jj] = None
                    coeffAr[kk,ii,jj] = None
                    jj += 1
                    continue
                    
                partA = model5a( x, 1, k1, k2 )
                partAD = model5ad( x, 1, k1, k2 )
                partA0 = model55a0( x, 1, k0 )
                partList = [ partA, partAD, partA0, np.ones( len(x) ) ]
                A = np.zeros( [4,4] )
                B = np.zeros([4,1])
                for rr in range(4):
                    for cc in range(4):
                        A[rr, cc] =  np.sum( np.multiply( partList[rr], partList[cc] ) )
                for rr in range(4):
                    B[rr, 0] = np.sum( np.multiply( y, partList[rr] ) )
                
                try:
                    X = np.linalg.solve(A, B)
                except:
                    if solutionLog is None:
                        slogName = "logs2/area/solutinLog"+str(seedId)+".txt"
                        solutionLog = open( slogName, "w")
                        solutionLog.write( "ii, jj, k1, k2, k0\r\n" )
                        print "The solutinLog was created!"
                        
                    solutionLog.write("%d, %d, %f, %f, %f\r\n" % ( ii, jj, k1, k2, k0 ) )
                    diffYYpNormAr[kk,ii,jj] = None
                    coeffAr[kk,ii,jj] = None
                    jj += 1
                    continue
                    
                #print X
                a, ad, a0, S0 = X
                d = ad/a
                coeffAr[kk,ii,jj] = S0, a, k1, k2, d, a0, k0
                yp = model55( x, S0, a, k1, k2, d, a0, k0 )
                diff = y - yp
                if sigma is not None:
                    diff = diff/sigma
                diffYYpNorm = np.linalg.norm( diff )
                #print diffYYpNorm
                diffYYpNormAr[kk,ii,jj] = diffYYpNorm
                
                jj += 1
                #if ii > 20:
                #    sys.exit()
                
            ii += 1
            counter = ii/float(k1n)
            if counter >= step:
                print "Fit model: ", round(step * 100), "%"
                step += 0.1
        kk += 1
    
    if solutionLog is not None:
        solutionLog.close()
    
    return diffYYpNormAr, coeffAr


def findY(x, y, xi):
    xDiff = abs(x - xi)
    idx = np.argmin(xDiff)
    
    return y[idx]


def getExtremes( x, y, xd, yd ):
    result = []
    for ii in xrange( len(yd) - 1 ):
        y0 = yd[ii]
        y1 = yd[ii+1]
        if (y0 * y1) > 0:
            continue
            
        if y0 > y1: #maximum
            xi = ( xd[ii] + xd[ii] ) /2.0
            yi = findY(x, y, xi)
            result.append(["max", xi, yi])
            
        else: #minimum
            xi = ( xd[ii] + xd[ii] ) /2.0
            yi = findY(x, y, xi)
            result.append(["min", xi, yi])
        
    return result



def dataPolyFit( x, y, degree, returnCoeffs = False ):
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    yp = p(x)
    #calculaton of  the r-squared (coefficient of determination)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yp-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    rSquared = ssreg/sstot
    
    if returnCoeffs == True:
        return yp, rSquared, coeffs
    else:
        return yp, rSquared


def myGradient( x, y, length1, length2 = None, length3 = None ):
    length1 = int(length1)
    n = len(y)
    
    yOut1 = np.zeros( n-length1 )
    xOut1 = np.zeros( n-length1 )
    yOut2 = None
    xOut2 = None
    yOut3 = None
    xOut3 = None
    for ii in range( n-length1 ):
        item = y[ ii : ii + length1 ]
        xItem = x[ ii : ii + length1 ]
        a, b = np.polyfit( xItem, item, 1 )
        yOut1[ii] = a
        xOut1[ii] = ( x[ ii + length1/2 ] + x[ ii + length1/2 + 1 ] ) / 2.0
        
    if length2:
        length2 = int(length2)
        n2 = len(yOut1)
        yOut2 = np.zeros( n2-length2 )
        xOut2 = np.zeros( n2-length2 )
        for ii in range(n2-length2):
            item = yOut1[ ii : ii + length2 ]
            xItem = xOut1[ii : ii + length2]
            a, b = np.polyfit( xItem, item, 1 )
            yOut2[ii] = a
            xOut2[ii] = ( xOut1[ ii + length2/2 ] + xOut1[ ii + length2/2 + 1 ] ) / 2.0
        
    if length2 and length3:
        length3 = int(length3)
        n3 = len(yOut2)
        yOut3 = np.zeros( n3-length3 )
        xOut3 = np.zeros( n3-length3 )
        for ii in range(n3-length3):
            item = yOut2[ ii : ii + length3 ]
            xItem = xOut1[ii : ii + length3]
            a, b = np.polyfit( xItem, item, 1 )
            yOut3[ii] = a
            xOut3[ii] = ( xOut2[ ii + length3/2 ] + xOut2[ ii + length3/2 + 1 ] ) / 2.0
        
    return xOut1, yOut1, xOut2, yOut2, xOut3, yOut3
    


def getSmoothData( y, smoothNum ):
    window = np.ones( smoothNum ) / smoothNum
    yOut = np.zeros( len(y) )
    yCon = np.convolve( y, window, "valid")
    #yOut = np.convolve( y, window, "same")
    yOut[ smoothNum -1 : ] = yCon
    yOut[ 0:smoothNum-1 ] = yCon[0]
    return yOut
    

def getSmoothData2( x, y, smoothNum, outlier = True ):
    n = len(y) - smoothNum
    yOut = []
    xOut = x[ smoothNum/2 : n + smoothNum/2 ]
    
    for ii in xrange(n):
        itemY = y[ ii : smoothNum + ii ]
        itemX = x[ ii : smoothNum + ii ]
        if (outlier == True) and (itemX[0] >= 0):
            coeffs = np.polyfit( itemX, itemY, 1)
            p = np.poly1d(coeffs)
            itemYp = p(itemX)
            diff = itemYp - itemY
            stdDiff = np.std(diff)
            sumItemY = 0
            num = 0
            numOfOutlier = 0
            
            for jj in range( len(diff) ):
                if abs( diff[jj] ) < 1.0 * stdDiff: # sigma limit
                    sumItemY += itemY[jj]
                    num += 1
                else:
                    #print "outlier"
                    #print stdDiff, diff, jj
                    numOfOutlier += 1
                
            if numOfOutlier > 1:
                print "numOfOutlier", numOfOutlier
                
            if num == 0:
                print "diff", diff
                print "stdDiff", stdDiff
                sys.exit()
            print "num", num
            meanItemY = sumItemY/num
            
        else:
            meanItemY = np.mean(itemY)
                
        yOut.append(meanItemY)
        
    yOut = np.array(yOut)
    
    return xOut, yOut


def getIdNums( idNumsIn, ignorList ):
    idNumsOut = []
    offset = 0
    for item in idNumsIn:
        if item in ignorList:
            offset += 1
            
        idNumsOut.append(item + offset )
        
    return idNumsOut


def efdAnalyse1( directory, measurementLabel ):
    subject = "efd"
    subject_area = "area"
    ignorList = IGNORE_LIST
    efdsCoeff = EFD_COEFFICIENTS
    dataOutPut = DATA_OUTPUT
    fph = FRAMES_PER_H
    detailA = DETAIL_AREA
    smoothNumA = SMOOTH_NUM_AREA
    smoothNumAD = SMOOTH_NUM_AREA_DETAIL
    lenAG1 = LEN_AREA_G1
    lenAG2 = LEN_AREA_G2
    lenAG3 = LEN_AREA_G3
    lenG1 = LEN_EFD_G1
    lenG2 = LEN_EFD_G2
    lenG3 = LEN_EFD_G3
    offset = OFFSET
    
    fileList = getLogs( directory )
    efds = readData( fileList, directory, subject, ignorList )
    areaAr = readData( fileList, directory, subject_area, ignorList )
    #np.savetxt("area_data", areaAr)
    #sys.exit()
    
    d1, d2, d3 = efds.shape
    print d1, d2, d3
    x = ( np.arange(d1) ) * 1/fph
    idNums = getIdNums( range(d2), ignorList )
    
    if createAreaPlotsGrad == True:
        log = open("logs2/area/extremes_area_"+measurementLabel+".txt", "w")
        for ii in xrange(d2):
            y = areaAr[:, ii]
            xn2a, yn2a = getSmoothData2( x, y, smoothNumA, outlier = True )
            x2, y2, x3, y3, x4, y4 = myGradient(xn2a, yn2a, lenAG1, lenAG2, lenAG3 )
            plt.figure()
            plt.plot(x2, y2, "ro-" )
            plt.plot(x3, y3, "b-" )
            plt.plot( [0,0], [0, max(y2)], "g-" )
            plt.plot([0, x[-1] ], [0, 0], "g-")
            plt.savefig("logs2/area/area_Plot_%003d_grad" %ii, dpi=100)
            plt.close()
            plt.figure()
            plt.plot(x3, y3, "bo-" )
            plt.plot(x4, y4, "k-" )
            plt.plot([0, x[-1] ], [0, 0], "g-")
            plt.savefig("logs2/area/area_Plot_%003d_grad2" %ii, dpi=100)
            plt.close()
            
            log.write(str(ii)+"\r\n")
            extremes0 = getExtremes( xn2, yn2, x2, y2 )
            log.write( "orig. data: "+str(extremes0)+"\r\n" )
            extremes1 = getExtremes( x2, y2, x3, y3 )
            log.write( "first der.: "+str(extremes1)+"\r\n" )
            extremes2 = getExtremes( x3, y3, x4, y4 )
            log.write( "second der.: "+str(extremes2)+"\r\n" )
            log.write("\r\n")
            
        log.close()
    
    if createAreaPlots == True:
        print areaAr.shape
        for ii in xrange(d2):
            y = areaAr[:, ii]
            x1, y1 = getSmoothData2(x, y, smoothNumA, outlier = True )
            plt.figure()
            plt.plot(x, y, "ko-" )
            plt.plot(x1, y1, "b-" )
            plt.savefig("logs2/area/area_Plot_%003d" %ii, dpi=100)
            plt.clf
            plt.close()
            
    if createAreaDetail == True:
        logD = open("logs2/area/extremes_areaD_"+measurementLabel+".txt", "w")
        for ii in xrange(d2):
            detailId = detailA * fph
            y = areaAr[0:detailId, ii]
            xad = x[0:detailId]
            x1, y1 = getSmoothData2( xad, y, smoothNumAD, outlier = False )
            yp, rSquared, coeffs = dataPolyFit( x1, y1, 3, returnCoeffs = True )
            ex1, ex2, xInf = extremsFromPoly3( coeffs )
            exLog = []
            if ex1:
                #if (ex1[1] > 0.5) and (ex1[1] < 2.0):
                if (ex1[1] > 0.5) and (ex1[1] < 5.0):
                    exLog.append(ex1)
            if ex2:
                #if (ex2[1] > 0.5) and (ex2[1] < 2.0):
                if (ex2[1] > 0.5) and (ex2[1] < 5.0):
                    exLog.append(ex2)
            if xInf:
                #if xInf > 0.5 and xInf < 2.0:
                if xInf > 0.5 and xInf < 5.0:
                    exLog.append(xInf)
                
            logD.write(str(ii)+"\r\n")
            logD.write(str(rSquared)+"\r\n")
            logD.write(str(exLog)+"\r\n")
            plt.figure()
            plt.plot(xad, y, "ko-" )
            plt.plot(x1, y1, "b-" )
            plt.plot(x1, yp, "r-" )
            plt.savefig("logs2/area/area_Plot_%003d_det" %ii, dpi=100)
            plt.clf
            plt.close()
            
        logD.close()
        
    if areaAnalyse2 == True:
        #gridK1 = GRID1
        gridK2 = GRID2
        gridK0 = GRID0
        gridF = GRIDF
        logFit = open("logs2/area/log_"+measurementLabel+"_fit.txt", "w")
        logFit.write("number seed, value, S0, a, k1, k2, d, a0, k0\r\n")
        logFitF = open("logs2/area/log_"+measurementLabel+"_fitF.txt", "w")
        logFitF.write("number seed, value, S0, a, k1, k2, d, a0, k0\r\n")
        logDI = open("logs2/area/log_"+measurementLabel+"_DI.txt", "w")
        logDI.write("number seed, y(0), x_max, y(x_max), y(12)\r\n")
        logDII = open("logs2/area/log_"+measurementLabel+"_DII.txt", "w")
        logDII.write("number seed, y(0), x_min, y(x_min), y(12)\r\n")
        logDIN = open("logs2/area/log_"+measurementLabel+"_DIN.txt", "w")
        logDIN.write("number seed, y(0), x_max, y(x_max), y(12)\r\n")
        logDIIN = open("logs2/area/log_"+measurementLabel+"_DIIN.txt", "w")
        logDIIN.write("number seed, y(0), x_min, y(x_min), y(12)\r\n")
        print areaAr.shape
            
        for ii in xrange(d2):
            if ii not in [ 1, 12, 17 ]:
                pass
                ##gridK1 = ( 0, 10.0, 10001 )
                ##gridK2 = ( 0, 10.0, 10001 )
                #continue
            y = areaAr[:, ii]
            #sigma = np.ones( y.shape )[300:] = 2.0
            #y = y[0:420]
            #x = x[0:420]
            print "len(x)", len(x)
            
            print "--------------------------------------------------------"
            print "Seed number: ", ii
            
            gridK1 = GRID1
            while True:
                diffYYpNormAr, coeffAr = fitModel55( x, y, ii, gridK1, gridK2, gridK0, sigma = None, oneSide = True )
            
                idCoeff = np.nanargmin( diffYYpNormAr )
                idCoeff = np.unravel_index( idCoeff, diffYYpNormAr.shape )
                if idCoeff[1] > gridK1[2] - 10:
                    gd = ( ( gridK1[1] - gridK1[0] ) / ( gridK1[2] - 1 ) )
                    gridK1 = ( gridK1[1] - 10*gd, gridK1[1] + 1, int(1/gd + 1 + 10) )
                    print "gd", gd
                    print "New gridK1: ", gridK1
                else:
                    break
                
            print "abs min: ", idCoeff, diffYYpNormAr[idCoeff]
            
            for jj in range( gridK0[2] ):
                arr = diffYYpNormAr[jj]
                plt.figure()
                xAr = np.linspace( gridK2[0], gridK2[1], gridK2[2] )
                yAr = np.linspace( gridK1[0], gridK1[1], gridK1[2] )
                plt.contourf( xAr, yAr, arr, 200 )
                plt.savefig("logs2/area/tmp/diffYYp_"+measurementLabel+"_%003d_%02d" %(ii, jj), dpi=500)
                plt.clf
                plt.close()
            
                maxValueIm = diffYYpNormAr[idCoeff] + ( np.nanmax(diffYYpNormAr) - diffYYpNormAr[idCoeff] )*0.1
                dataForDetail = arr
                dataForDetail[ dataForDetail > maxValueIm ] = np.nan
                plt.figure()
                plt.contourf( xAr, yAr, dataForDetail, 100 )
                plt.savefig("logs2/area/tmp/diffYYp_"+measurementLabel+"_%003d_%02d_D" %(ii, jj), dpi=500)
                plt.clf
                plt.close()
            
            if idCoeff[2] == 1:
                print "The abs min is dubious!"
                S0, a, k1, k2, d, a0, k0 = coeffAr[idCoeff]
                logFit.write("%d, %f, %f, %f, %f, %f, %f, %f, %f\r\n" %( ii, diffYYpNormAr[idCoeff], S0, a, k1, k2, d, a0, k0 ) )
                logFit.flush()
                
                plt.figure()
                plt.plot(x,y,"k")
                plt.savefig( "logs2/area/area_"+measurementLabel+"_Plot_fit_%003d" %ii, dpi=200 )
                plt.clf
                plt.close()
                
                plt.figure()
                plt.plot(x,y/S0,"k")
                plt.savefig( "logs2/area/area_"+measurementLabel+"_Plot_fit_%003dN" %ii, dpi=200 )
                plt.clf
                plt.close()
                
                continue
            
            S0, a, k1, k2, d, a0, k0 = coeffAr[idCoeff]
            logFit.write("%d, %f, %f, %f, %f, %f, %f, %f, %f\r\n" %( ii, diffYYpNormAr[idCoeff], S0, a, k1, k2, d, a0, k0 ) )
            logFit.flush()
            
            # + or - ?????
            gridD1F = 5.0*( ( gridK1[1] - gridK1[0] ) / ( gridK1[2] - 1 ) )
            gridD2F = 5.0*( ( gridK2[1] - gridK2[0] ) / ( gridK2[2] - 1 ) )
            gridD0F = 2.0*( ( gridK0[1] - gridK0[0] ) / ( gridK0[2] - 1 ) )
            
            gridK1F = ( k1 - gridD1F, k1 + gridD1F, gridF )
            gridK2F = ( k2 - gridD2F, k2 + gridD2F, gridF )
            gridK0F = ( k0 - gridD0F, k0 + gridD0F, 9 )
            
            print "Dedail"
            diffYYpNormArF, coeffArF = fitModel55( x, y, ii, gridK1F, gridK2F, gridK0F, sigma = None, oneSide = True )
            idCoeffF = np.nanargmin(diffYYpNormArF)
            idCoeffF = np.unravel_index( idCoeffF, diffYYpNormArF.shape )
            print "Final result: ", idCoeffF
            S0, a, k1, k2, d, a0, k0 = coeffArF[idCoeffF]
            logFitF.write("%d, %f, %f, %f, %f, %f, %f, %f, %f\r\n" %( ii, diffYYpNormArF[idCoeffF], S0, a, k1, k2, d, a0, k0 ) )
            print "Seed number, diffYYpNormF", ii, diffYYpNormArF[idCoeffF]
            logFitF.flush()
            
            yp = model55( x, S0, a, k1, k2, d, a0, k0 )
            ypDI = model55DI( x, a, k1, k2, d, a0, k0 )
            ypDII = model55DII( x, a, k1, k2, d, a0, k0 )
            
            yN = y/S0
            ypN = model55( x, 1, a/S0, k1, k2, d, a0/S0, k0 )
            ypDIN = model55DI( x, a/S0, k1, k2, d, a0/S0, k0 )
            ypDIIN = model55DII( x, a/S0, k1, k2, d, a0/S0, k0 )
            
            #if k2 <= k1*d:
                #xMaxDI = -1
                #xMinDII = -1
                #yXmaxDI = -1
                #yXminDII = -1
                #yXmaxDIN = -1
                #yXminDIIN = -1
            #else:
                #xMaxDI = np.log( ( -k2**2 * ( 1 - d ) ) / ( k1*(k1*d - k2) ) ) / ( k2 - k1 )
                #yXmaxDI = model5DI(xMaxDI, a, k1, k2, d, a0)
                #yXmaxDIN = model5DI(xMaxDI, a/S0, k1, k2, d, a0/S0)
                ##print yXmaxDI
                
                #xMinDII = np.log( ( k2**3 * ( 1 - d ) ) / ( k1**2 *(k2 - k1*d) ) ) / ( k2 - k1 )
                #yXminDII = model5DII(xMinDII, a, k1, k2, d )
                #yXminDIIN = model5DII(xMinDII, a/S0, k1, k2, d )
                ##print yXminDII
            
            xMaxDI = np.argmax(ypDI)/fph
            xMinDII = np.argmin(ypDII)/fph
            yXmaxDI = np.max(ypDI)
            yXminDII = np.min(ypDII)
            yXmaxDIN = np.max(ypDIN)
            yXminDIIN = np.min(ypDIIN)
            if xMaxDI == 0:
                extremI = 0
            else:
                extremI = 1
                
            if xMinDII == 0:
                extremII = 0
            else:
                extremII = 1
                            
            logDI.write("%d, %f, %f, %f, %f, %d\r\n" %(ii, ypDI[0], xMaxDI, yXmaxDI, ypDI[-1], extremI ) )
            logDI.flush()
            logDII.write("%d, %f, %f, %f, %f, %d\r\n" %(ii, ypDII[0], xMinDII, yXminDII, ypDII[-1], extremII ) )
            logDII.flush()
            logDIN.write("%d, %f, %f, %f, %f, %d\r\n" %(ii, ypDIN[0], xMaxDI, yXmaxDIN, ypDIN[-1], extremI ) )
            logDIN.flush()
            logDIIN.write("%d, %f, %f, %f, %f, %d\r\n" %(ii, ypDIIN[0], xMinDII, yXminDIIN, ypDIIN[-1], extremII ) )
            logDIIN.flush()
            
            plt.figure()
            plt.plot(x,y,"k")
            plt.plot(x,yp,"r")
            if xMinDII != 0:
                plt.plot( [xMaxDI, xMaxDI], [np.nanmin(y), np.nanmax(y)], "b")
                plt.plot( [xMinDII, xMinDII], [np.nanmin(y), np.nanmax(y)], "g")
            plt.savefig( "logs2/area/area_"+measurementLabel+"_Plot_fit_%003d-F" %ii, dpi=200 )
            plt.clf
            plt.close()
            
            plt.figure()
            plt.plot(x,ypDI,"b")
            plt.plot(x,ypDII,"g")
            plt.plot( [0, 12], [0, 0], "k")
            if xMinDII != 0:
                plt.plot( [xMaxDI, xMaxDI], [0, yXmaxDI], "b")
                plt.plot( [xMinDII, xMinDII], [0, yXminDII], "g")
            plt.savefig( "logs2/area/area_"+measurementLabel+"_Plot_fit_%003d-FD" %ii, dpi=200 )
            plt.clf
            plt.close()
            
            plt.figure()
            plt.plot(x,yN,"k")
            plt.plot(x,ypN,"r")
            if xMinDII != 0:
                plt.plot( [xMaxDI, xMaxDI], [np.nanmin(yN), np.nanmax(yN)], "b")
                plt.plot( [xMinDII, xMinDII], [np.nanmin(yN), np.nanmax(yN)], "g")
            plt.savefig( "logs2/area/area_"+measurementLabel+"_Plot_fit_%003d-FN" %ii, dpi=200 )
            plt.clf
            plt.close()
            
            plt.figure()
            plt.plot(x,ypDIN,"b")
            plt.plot(x,ypDIIN,"g")
            plt.plot( [0, 12], [0, 0], "k")
            if xMinDII != 0:
                plt.plot( [xMaxDI, xMaxDI], [0, yXmaxDIN], "b")
                plt.plot( [xMinDII, xMinDII], [0, yXminDIIN], "g")
            plt.savefig( "logs2/area/area_"+measurementLabel+"_Plot_fit_%003d-FDN" %ii, dpi=200 )
            plt.clf
            plt.close()
            
            
            
            #initEs5 = np.array([ y[0], 2000.0, 0.2, 0.1, 0.0, 1000.0 ])
            ##sigma = None
            #sigma = np.linspace(1.0, 2.0, len(y) )
            #try:
                #popt5, pcov5 = optimize.curve_fit( model5, x, y, initEs5 )
            #except:
                #print ii
                #print "problem... TODO"
                #continue
                
            #print ii
            
            ##model5
            #S0, a, k1, k2, d, a0 = popt5
            #print "S0, a, k1, k2, d, a0, k0", S0, a, k1, k2, d, a0
            #yp = model5( x, S0, a, k1, k2, d, a0 )
            ##ys1 = model5S1( x, S0, a, k1, k2, d )
            ##ys2 = model5S2( x, S0, a, k1, k2, d)
            ##ypI = model5DI( x, a, k1, k2, d )
            ##ypII = model4DII( x, a, k1, c, k2 )
            
            #plt.figure()
            #plt.plot(x[:], y[:], "k-" )
            #plt.plot( x, yp, "r-" )
            #plt.savefig("logs2/area/area_Plot_%003d_exp1" %ii, dpi=100)
            #plt.clf
            #plt.close()
            
            
    
    kk = 0
    for item in efdsCoeff:
        plotLimits = [ np.nanmin( efds[:, :, item] ), np.nanmax( efds[:, :, item] ) ]
        dataOutP = dataOutPut[kk]
        spl = interpolate.UnivariateSpline
        
        logResults = open("logs2/"+dataOutP+"/log_"+measurementLabel+"_results.txt", "w")
        logResults.write("Seed ID, y(0), y(12), Line a, Line b, Line R2, X_max, Y_max, X_min, Y_min\r\n")
        logExtrems = open("logs2/"+dataOutP+"/log_"+measurementLabel+"_Extrems.txt", "w")
        
        if createEfdsPlotsOrig == True:
            for ii in xrange(d2):
                #print "len(x)", len(x)
                y = efds[:, ii, item]
                x = ( np.arange(d1) ) * 1/fph
                y = y[0:360]
                x = x[0:360]
                x, y = removeNan( x, y )
                
                ys = spl(x, y, k = 4)
                ysDII = ys.derivative(2)
                xD0 = ys.derivative().roots()
                yD0 = ys(xD0)
                extremType = ysDII(xD0) #x<0 - max; x>0 - min
                minimums = []
                maximums = []
                for jj in range(len(xD0)):
                    if extremType[jj] < 0:
                        maximums.append( [ xD0[jj], yD0[jj] ] )
                    else:
                        minimums.append( [ xD0[jj], yD0[jj] ] )
                
                yp, rSquared, coeffs = dataPolyFit( x, y, 1, returnCoeffs = True )
                yData = ys(x)
                xMin = ( np.argmin(yData) + 1 )/fph
                yMin = np.min(yData)
                xMax = ( np.argmax(yData) + 1 )/fph
                yMax = np.max(yData)
                
                logResults.write("%d, %f, %f, %f, %f, %f, %f, %f, %f, %f\r\n" %( ii, yData[0], yData[-1], coeffs[0], coeffs[1], rSquared, xMax, yMax, xMin, yMin ) )
                logResults.flush()
                logExtrems.write(str(ii)+", maximums, "+str(maximums)+", minimums, "+str(minimums)+"\r\n")
                logExtrems.flush()
                
                plt.figure()
                plt.plot( x, y, "ko-")
                plt.plot( x, ys(x), "r-")
                plt.plot( x, yp, "m-" )
                for extrem in maximums:
                    plt.plot(extrem[0], extrem[1], "bo")
                for extrem in minimums:
                    plt.plot(extrem[0], extrem[1], "go")
                plt.ylim( plotLimits[0], plotLimits[1] )
                plt.title("item num.: "+str(ii))
                plt.savefig("logs2/"+dataOutP+"/efds_"+measurementLabel+"_Plot_%003d" %ii, dpi=100)
                plt.clf
                plt.close()
                #sys.exit()
                
        
        
        kk += 1
            

if __name__ == "__main__": 
    if len(sys.argv) < 2:
        print __doc__
        sys.exit(1)
    
    print "ignore list", IGNORE_LIST
    measurementLabel = sys.argv[3]
    directory = sys.argv[2]
    switch = sys.argv[1]
    if switch == "efd":
        efdAnalyse1( directory, measurementLabel )
        
    print "END"
