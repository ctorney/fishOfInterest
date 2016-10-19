
import numpy as np

import scipy.io.netcdf as Dataset
import cv2
import sys,os
import pandas as pd

dataDir = '/media/ctorney/SAMSUNG/data/fishPredation/'
outDir = '/home/ctorney/Dropbox/fishPredation/csvfiles/'
outDir = './'
arms0 = cv2.imread('arm0.png',0)
arms1 = cv2.imread('arm1.png',0)
arms2 = cv2.imread('arm2.png',0)

arms = np.zeros_like(arms0, dtype=int)-1
#arms = arms + arms0 + 2*arms1 + 3*arms2
arms = arms.astype(int) + (arms0.astype(float)/255.0).astype(int) + (2*arms1.astype(float)/255.0).astype(int) + (3*arms2.astype(float)/255.0).astype(int)

allTrials = np.loadtxt('trialList.txt',delimiter=',',dtype=int)
for trial in allTrials:
    trialName = "MVI_" +  str(trial[4])
    
    if trialName!="MVI_3464":
        continue
    NUMFISH = int(trial[3])
    ncFileName = dataDir + "tracked/linked" + trialName + ".nc"
    #print(ncFileName)
    f = Dataset.NetCDFFile(ncFileName, 'r', mmap=False)


    # get the positions variable
    trXY = f.variables['trXY']
    trackList = []
    trackList = np.empty_like (trXY.data)
    np.copyto(trackList,trXY.data)
    [trackIndex,timeIndex]=np.nonzero(trackList[:,:,0])
    print(trialName,trackList.shape[0])
    #continue
    # get the movie frame numbers
    fid = f.variables['fid']
    fishIDs = []
    fishIDs = np.empty_like(fid.data)
    np.copyto(fishIDs, fid.data)
    cid = f.variables['certID']
    certIDs = []
    certIDs = np.empty_like(cid.data)
    np.copyto(certIDs, cid.data)




    # these variables store the index values when a track is present
    [trackIndex,timeIndex]=np.nonzero(trackList[:,:,0])

    #print 'TIME:F0 (arm,uncert):F1 (arm,uncert):F2 (arm,uncert):F3 (arm,uncert)'

    fishArm = np.zeros((NUMFISH,np.max(timeIndex)))-1
    fishCert = np.zeros((NUMFISH,np.max(timeIndex)))
    fishTrackNum = np.zeros((NUMFISH,np.max(timeIndex)))


    columns = ['frame', 'x', 'y', 'fid','tid','arm','cert']
    df = pd.DataFrame(columns=columns) 
    

    for t in range(np.min(timeIndex),np.max(timeIndex)):
        liveTracks = trackIndex[timeIndex[:]==t]

        for tr in liveTracks:
            xp = trackList[tr, t,0]
            yp = trackList[tr, t,1]


            df.loc[len(df)] = [t,xp,yp,fishIDs[tr,t],tr,arms[((yp,xp))],certIDs[tr,t]]
            
    dfx = df[['x','tid']]
    dfy = df[['y','tid']]
    df.to_csv(outDir + trialName + '.csv')
    dfx.to_csv(outDir + trialName + 'X.csv')
    dfy.to_csv(outDir + trialName + 'Y.csv')
