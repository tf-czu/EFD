"""
  Tools for analyse of logsfiles
    usage:
         python data.py <switch> <directory>
            switch: a - area
            switch: efd
"""

import sys
import os
import numpy as np
from data2 import *
from matplotlib import pyplot as plt
from elliptic_fourier_descriptors import *


IGNORE_LIST = [ 96, 116 ]
FRAMES_PER_H = 12.0


def getEfdCoefficient(num):
    n = num / 4
    l = num % 4
    if l == 0:
        letter = "a"
    elif l == 1:
        letter = "b"
    elif l == 2:
        letter = "c"
    elif l == 3:
        letter = "d"
        
    return str(n) + letter
    

def readData( fileList, directory, subject, ignorList ):
    #fileList = fileList.sort()
    if subject == "area":
        fileIdList = []
        areaList = []
        for fileN in fileList:
            if fileN == "black_images_log.txt":
                print fileN
                continue
            #print fileN
            fId = fileN.split("_")[1]
            #fId = fileN.split("_")[2] # for m_tae2016 only!!
            fId = int(fId) - 1
            fileIdList.append(fId)
            
            areaL = []
            f = open( directory+fileN, "r" )
            label = None
            for line in f:
                line = line.split(":")
                if line[0] == label:
                    label = int(line[1])
                    continue
                
                if line[0] == "area (mm2)" and label not in ignorList:
                    area = float(line[1])
                    areaL.append(area)
            
            f.close()
            areaList.append(areaL)
        
        #print fileIdList
        areaList = np.array(areaList)
        maxFId = max(fileIdList) +1
        s1, s2 = areaList.shape
        areaAr = np.empty( [ maxFId, s2 ] )
        areaAr.fill(np.nan)
        
        ii = 0
        for fId in fileIdList:
            areaAr[fId] = areaList[ii]
            ii += 1
        
        return areaAr
        
    elif subject == "efd":
        fileIdList = []
        totalEfdList = []
        for fileN in fileList:
            if fileN == "black_images_log.txt":
                print fileN
                continue
            fId = fileN.split("_")[1]
            #fId = fileN.split("_")[2] #only for m_tae2016!!
            fId = int(fId) - 1
            fileIdList.append(fId)
            
            efdList = []
            efdL = []
            f = open( directory+fileN, "r" )
            read = False
            label = None
            for line in f:
                lineSplit = line.split(":")
                if line[0] == label:
                    label = int(line[1])
                    continue
                
                if lineSplit[0] == "efd" and label not in ignorList:
                    read = True
                    continue
                    
                if len(line) < 4:
                    read = False
                    efdList.append(efdL)
                    efdL = []
                    continue
                
                if read == True:
                    line = line[1:-3]
                    #print line
                    numFromLine = map(float, line.split() )
                    for item in numFromLine:
                        efdL.append(item)
            
            f.close()
            
            totalEfdList.append(efdList)
            
        totalEfdList = np.array(totalEfdList)
        maxFId = max(fileIdList) +1
        s1, s2, s3 = totalEfdList.shape
        totalEfdArS = np.empty( [ maxFId, s2, s3 ] )
        totalEfdArS.fill(np.nan)
        ii = 0
        for fId in fileIdList:
            totalEfdArS[fId] = totalEfdList[ii]
            ii += 1
        
        #print totalEfdList.shape
        #print totalEfdArS[144, 80, :]
        
        return totalEfdArS


def getLogs( directory ):
    fileList = os.listdir( directory )
    #print fileList
    
    cleanFileL = []
    for fileN in fileList:
        if fileN.split(".")[-1] == "txt":
            cleanFileL.append( fileN )
    
    print len(cleanFileL)
    return cleanFileL


def areaAnalyse(directory):
    ignorList = IGNORE_LIST
    fph = FRAMES_PER_H 
    subject = "area"
    
    fileList = getLogs( directory )
    areaAr = readData( fileList, directory, subject, ignorList )
    #print "{0}".format(areaAr[0])
    print areaAr.shape
    
    normAreaAr = areaAr.copy()
    initArea = areaAr[0]
    #print initArea
    for ii in xrange( len(areaAr) ):
        normAreaAr[ii] = areaAr[ii] / initArea
        
    print "{0}".format(normAreaAr)
    
    n = len(normAreaAr)
    meanNormArea = np.zeros( n )
    sdNormArea = np.zeros( n )
    for ii in xrange( n ):
        meanNormArea[ii] = np.mean( normAreaAr[ii] )
        sdNormArea[ii] = np.std( normAreaAr[ii] )
    
    print meanNormArea, sdNormArea
    x = ( np.array(range( n )) + 1 ) * 1/fph
    #plt.figure()
    plt.subplots(figsize=(20,6))
    plt.errorbar( x, meanNormArea, yerr=sdNormArea )
    plt.show()
    plt.savefig("logs/area/area_PlotE", dpi=300)
    
    results = open("logs/area/result_area.txt", "w")
    results.write("note:\r\n")
    results.write("Ignore list: "+str(ignorList)+"\r\n")
    results.write("meanNormArea\r\n")
    results.write(str( list( meanNormArea ) )+"\r\n")
    results.write("sdNormArea\r\n")
    results.write(str( list( sdNormArea ) )+"\r\n")
    results.write("normAreaAr\r\n")
    for item in normAreaAr:
        results.write( str( list(item) ) )
        results.write("\r\n")
        
    results.write("areaAr\r\n")
    for item in areaAr:
        results.write( str(list(item)) )
        results.write("\r\n")

    results.close()


def efdAnalyse( directory ):
    ignorList = IGNORE_LIST
    fph = FRAMES_PER_H
    subject = "efd"
    
    fileList = getLogs( directory )
    efds = readData( fileList, directory, subject, ignorList )
    meanEfds = np.zeros( ( efds.shape[0], efds.shape[2] ) )
    sdEfds = np.zeros( meanEfds.shape )
    varCoEfds = np.zeros( meanEfds.shape )
    ii = 0
    for efd in efds:
        meanEfds[ii] = np.mean( efd, axis = 0 )
        sdEfds[ii] = np.std( efd, axis = 0 )
        varCoEfds[ii] = sdEfds[ii] / abs(meanEfds[ii])
        ii += 1
    
    print meanEfds[0]
    print sdEfds[0]
    print varCoEfds[0]
    print meanEfds.shape
    
    #plt.hist(efds[0, :, 8])
    #plt.title("koeficient: "+str(ii))
    #plt.savefig('hists_test/Hist'+str(ii), dpi=50)
    
    for jj in range(80):
        plt.figure()
        plt.hist(efds[0, :, jj])
        print jj
        plt.title("koeficient: "+str(jj))
        plt.savefig('hists_test/Hist'+str(jj), dpi=100)
        plt.close()
    
    n = meanEfds.shape[0]
    for ii in xrange( meanEfds.shape[1] ):
        coefficient = getEfdCoefficient(ii)
        plt.figure()
        x = ( np.array(range( n )) + 1 ) * 1/fph
        plt.errorbar( x, meanEfds[:, ii], yerr=sdEfds[:, ii] )
        plt.title("coefficient: "+coefficient)
        plt.savefig("efdsPlots/efds_Plot_"+coefficient, dpi=100)
        plt.clf
        #plt.show()
        plt.close()
        
    x = range(44)
    for jj in xrange( meanEfds.shape[0] ):
        plt.figure()
        plt.plot( x, abs(meanEfds[jj, 0:44]), "ro-" )
        plt.grid( True )
        plt.axis([0, 44, 10e-6, 1.1])
        plt.yscale('log')
        time = (jj + 1) *1/fph
        plt.title("time: "+str(time) )
        plt.savefig("efdsPlots2/efds_Plot2_%003d" %jj, dpi=100)
        plt.close()
        

def efdAnalyse2( directory ):
    ignorList = IGNORE_LIST
    fph = FRAMES_PER_H
    subject = "efd"
    
    fileList = getLogs( directory )
    efds = readData( fileList, directory, subject, ignorList )
    
    efdErrL = []
    print efds.shape
    d1 ,d2, d3 = efds.shape
    for nn in xrange(d2):
        print nn
        for ii in xrange(d1):
            efdR = efds[ii, nn, :]
            efdR[:4] = 0
            efdR = np.resize(efdR,(20,4))
            #print efdR
            #print len(efdR)
            recR = reconstruct(efdR, T = 2810.5, K = None)
            errL = []
            for jj in xrange(2,20+1):
                efd = efds[ii, nn, :]
                efd[:4] = 0
                efd = np.resize(efd,(20,4))
                efd = efd[:jj]
                #print efd
                rec = reconstruct(efd, T = 2810.5, K = None)
                err = np.linalg.norm(recR-rec)/np.linalg.norm(recR)
                errL.append(err)
                #np.savetxt("recR", recR)
                #np.savetxt("rec", rec)
                #np.savetxt("recR-rec", recR-rec)
                #print "{0}".format(recR)
                #print "{0}".format(rec)
                #print "{0}".format(recR - rec)
                #sys.exit()
            efdErrL.append(errL)
    
    efdErr = np.array(efdErrL)
    efdErrMean = np.mean(efdErr,0)
    efdErrSD = np.std(efdErr,0)
    print efdErrMean
    print efdErrSD
    plt.boxplot(efdErr)
    plt.savefig("logs/boxplot", dpi=300)
    
    x = range(1,20)
    plt.subplots(figsize=(20,10))
    plt.plot(x, efdErrMean, "ko", label = "Mean value")
    plt.errorbar( x, efdErrMean, yerr=efdErrSD,fmt='ko', ecolor = "k", label = "Standard deviation", capthick=2, markersize=10 )
    #plt.legend( loc = 1, borderaxespad = 0.2 )
    plt.axis([ 0, 20, -0.01, 0.08])
    plt.xlabel("Harmonic content N", fontsize = 25)
    plt.ylabel("Relative error RE", fontsize = 25)
    plt.yticks( size = 20)
    plt.xticks( x, size = 20)
    plt.grid( True )
    plt.savefig("logs/ErrorPlot", dpi=300)
    plt.close()
    

def efdAnalyse3( directory ):
    ignorList = IGNORE_LIST
    fph = FRAMES_PER_H
    subject = "efd"
    
    fileList = getLogs( directory )
    efds = readData( fileList, directory, subject, ignorList )
    
    print efds.shape
    d1 ,d2, d3 = efds.shape
    efds = np.reshape(efds, ( d1*d2, d3 ) )
    efds = np.abs(efds)
    meanEfds = np.mean(efds,0)[4:24]
    sdEfds = np.std(efds,0)[4:24]
    print meanEfds
    maxEfds = np.max(efds,0)[4:24]
    minEfds = np.min(efds,0)[4:24]
    print sdEfds
    
    x = np.arange(4,24)
    xLabel = [ getEfdCoefficient(xx) for xx in x ]
    plt.figure()
    plt.subplots(figsize=(20,6))
    plt.subplots_adjust(bottom=0.15)
    plt.plot(x, meanEfds, "ko", markersize=10, label = "Mean value")
    plt.plot(x, maxEfds, "k_", markersize=10, mew=2.0, label = "Mean value")
    plt.plot(x, minEfds, "k_", markersize=10, mew=2.0, label = "Mean value")
    #plt.errorbar( x, meanEfds, yerr=sdEfds,fmt='ko', ecolor = "k", label = "Standard deviation", capthick=2 )
    plt.grid( True )
    plt.axis([3, 24, 10e-8, 1.2])
    plt.yscale('log')
    #plt.legend( loc = 1, borderaxespad = 0.2 )
    plt.xlabel("Elliptic Fourier descriptors", fontsize = 25)
    plt.ylabel("Magnitude", fontsize = 25)
    plt.yticks( size = 20)
    plt.xticks( x, xLabel, size = 20)
    #plt.ylim(10e-6, 1.2)
    plt.savefig("logs/EFDSPlot", dpi=300)
    plt.close()


def efdAnalyse4( directory ):
    ignorList = IGNORE_LIST
    fph = FRAMES_PER_H
    subject = "efd"
    
    fileList = getLogs( directory )
    efds = readData( fileList, directory, subject, ignorList )
    
    efdErrL = []
    print efds.shape
    d1 ,d2, d3 = efds.shape
    for nn in xrange(d2):
        print nn
        for ii in xrange(d1):
            efdR = efds[ii, nn, :]
            efdR[:4] = 0
            efdR = np.resize(efdR,(20,4))
            #print efdR
            #print len(efdR)
            recR = reconstruct(efdR, T = 2810.5, K = None)
            efd = np.zeros([4,4])
            efd[1,0] = efdR[1,0]
            efd[1,3] = efdR[1,3]
            efd[3,0] = efdR[3,0]
            efd[3,3] = efdR[3,3]
            rec = reconstruct(efd, T = 2810.5, K = None)
            err = np.linalg.norm(recR-rec)/np.linalg.norm(recR)
            efdErrL.append(err)
    
    efdErr = np.array(efdErrL)
    efdErrMean = np.mean(efdErr)
    efdErrSD = np.std(efdErr)
    print efdErrMean
    print efdErrSD
    plt.boxplot(efdErr)
    plt.savefig("logs/boxplot2", dpi=300)
    f=open("logs/efdAnalyse4_log.txt", "w")
    f.write("efdErrMean %f\r\n" %efdErrMean)
    f.write("efdErrSD %f\r\n" %efdErrSD)
    f.close()
    

def efdAreaPlot( directory ):
    ignorList = IGNORE_LIST
    fph = FRAMES_PER_H
    subject = "efd"
    
    fileList = getLogs( directory )
    efds = readData( fileList, directory, subject, ignorList )
    area = readData( fileList, directory, "area", ignorList )
    
    d1, d2, d3 = efds.shape
    seeds = range(20)
    x = ( np.arange(d1) ) * 1/fph
    for ss in seeds:
        sArea = area[:,ss]
        x2, y2, x3, y3, x4, y4 = myGradient(x, sArea, 12, 12, 12 )
        
        efd1d = efds[:, ss, 7]
        efd2d = efds[ :, ss, 11 ]
        efd3a = efds[ :, ss, 12 ]
        efd3d = efds[ :, ss, 15 ]
        
        fig = plt.figure(figsize=(15,20))
        ax = fig.add_subplot(211)
        p0, = ax.plot( x, sArea, "ko-", label = "Seed area" )
        ax2 = ax.twinx()
        p1, = ax2.plot( x2, y2, "k-", label = "Seed area rate" )
        ax.grid()
        ax.set_xlabel("Time (h)", fontsize = 20)
        ax.set_ylabel("Seed area ($mm^{2}$)", fontsize = 20)
        ax2.set_ylabel("Seed area rate ($mm^{2} h^{-1}$)", fontsize = 20)
        #ax2.set_ylim(0, 35)
        #ax.set_ylim(-20,100)
        ax.tick_params(labelsize=15)
        ax2.tick_params(labelsize=15)
        l = [ p0, p1 ]
        ax.legend( l, [ll.get_label() for ll in l], loc=5 )
        
        ax = fig.add_subplot(212)
        p0, = ax.plot( x, efd1d, "ko-", label = "Coefficient $d_{1}$" )
        ax2 = ax.twinx()
        p1, = ax2.plot( x, efd2d, "k+-", label = "Coefficient $d_{2}$" )
        p2, = ax2.plot( x, efd3a, "kv-", label = "Coefficient $a_{3}$" )
        p3, = ax2.plot( x, efd3d, "k^-", label = "Coefficient $d_{3}$" )
        ax.grid()
        ax.set_xlabel("Time (h)", fontsize = 20)
        ax.set_ylabel("Coefficient $d_{1}$ (-)", fontsize = 20)
        ax2.set_ylabel("Coefficients $d_{2}$, $a_{3}$ and $d_{3}$ (-)", fontsize = 20)
        #ax2.set_ylim(0, 35)
        #ax.set_ylim(-20,100)
        ax.tick_params(labelsize=15)
        ax2.tick_params(labelsize=15)
        l = [ p0, p1, p2, p3 ]
        ax.legend( l, [ll.get_label() for ll in l], loc = 5 )
        
        plt.savefig("logs/fig_%d" %ss, dpi=300)
        


if __name__ == "__main__": 
    if len(sys.argv) < 2:
        print __doc__
        print "ignore list", IGNORE_LIST
        sys.exit(1)
    
    directory = sys.argv[2]
    switch = sys.argv[1]
    if switch == "a":
        areaAnalyse( directory )
        
    elif switch == "efd":
        efdAnalyse( directory )
    
    elif switch == "efd2":
        efdAnalyse2( directory )
        
    elif switch == "efd3":
        efdAnalyse3( directory )
        
    elif switch == "efd4":
        efdAnalyse4( directory )
        
    elif switch == "plot":
        efdAreaPlot( directory )
        
    print "END"
