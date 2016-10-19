
import numpy as np

import scipy.io.netcdf as Dataset
import cv2
import sys


dataDir = '/home/ctorney/data/fishPredation/'

allTrials = np.loadtxt('trialList.txt',delimiter=',',dtype=int)
counts = np.zeros(len(allTrials))
for t, trial in enumerate(allTrials):
    trialName = "MVI_" +  str(trial[3])
    NUMFISH = int(trial[2])
    ncFileName = dataDir + "tracked/linked" + trialName + ".nc"
    f = Dataset.NetCDFFile(ncFileName, 'r')
    

    # get the positions variable
    trXY = f.variables['trXY']
    trackList = []
    trackList = np.empty_like (trXY.data)
    np.copyto(trackList,trXY.data)
    
    tCount = np.shape(trackList)[0]
    
    for track in range(tCount):
        # delete anything less than a second
        [vals] = np.nonzero(trackList[track,:,0])
        if len(vals)<24:
            trackList[track,vals,0]=0
            trackList[track,vals,1]=0

    [trackIndex,timeIndex]=np.nonzero(trackList[:,:,0])
    trXY=None
    f.close()
    counts[t]= len(np.unique(trackIndex))
    print(trialName,counts[t])
        
    
np.savetxt('relinkList.txt', allTrials[np.argsort(counts)[::-1]],delimiter=',',fmt='%d')
