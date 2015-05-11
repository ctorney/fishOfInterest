
import numpy as np
import Scientific.IO.NetCDF as Dataset
import cv2
import sys, os

def main():
    results = np.zeros( shape=(0, 4) )
    dataDir = '/home/ctorney/data/fishPredation/'

    allTrials = np.loadtxt('trialList.txt',delimiter=',',dtype=int)
    for trial in allTrials:
        trialName = "MVI_" +  str(trial[3])
        NUMFISH = int(trial[2])
        for tr in range(NUMFISH):
            directory = dataDir + '/process/' + trialName + '/FR_ID' + str(tr) + '/'
            files = [name for name in os.listdir(directory)]
            thisData = np.zeros((len(files)))
            i = 0
            for imName in files:
                sample = cv2.imread(directory + imName)
                thisIm = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(thisIm,20,255,cv2.THRESH_BINARY)
                cv2.imshow('frame',thresh)
                cv2.waitKey(30)
                thisData[i]=0
                
                i = i + 1

        

    
    
            t2 = np.hstack((trial[0:3],0))
            print t2
            results = np.vstack((results, t2))
    #print results
    return results
    

if __name__ == "__main__":
    allData = main()
    #plt.boxplot([allData[np.logical_and(allData[:,1]==0 , allData[:,0]==0 ),3],allData[np.logical_and(allData[:,1]==0 , allData[:,0]==0 ),3],allData[np.logical_and(allData[:,1]==0 , allData[:,0]==0 ),3]],50,'')
    #plt.boxplot([allData[allData[:,0]==0,3],allData[allData[:,0]==1,3],allData[allData[:,0]==2,3]],50,'')
    #means = ([np.mean(allData[allData[:,0]==i,3]) for i in range(3)])
    #plt.scatter([1, 2, 3], means)
    #plt.show()
    np.savetxt('sizeData.txt',allData, delimiter=',', fmt=['%d','%d','%d','%f'])
