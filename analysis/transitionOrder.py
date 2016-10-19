
import numpy as np

import scipy.io.netcdf as Dataset
import cv2
import sys,os

dataDir = '/home/ctorney/data/fishPredation/'
dataDir = '/media/ctorney/SAMSUNG/data/fishPredation/'
outDir = '/home/ctorney/Dropbox/fishPredation/csvfiles/'
outDir = './'


def main():
    results = np.zeros( shape=(0, 6) )
    trialRes = np.zeros( shape=(0, 6) )
        
    arms0 = cv2.imread('arm0.png',0)
    arms1 = cv2.imread('arm1.png',0)
    arms2 = cv2.imread('arm2.png',0)

    arms = np.zeros_like(arms0, dtype=int)-1
#arms = arms + arms0 + 2*arms1 + 3*arms2
    arms = arms.astype(int) + (arms0.astype(float)/255.0).astype(int) + (2*arms1.astype(float)/255.0).astype(int) + (3*arms2.astype(float)/255.0).astype(int)

    
    allTrials = np.loadtxt('trialList.txt',delimiter=',',dtype=int)
    for trial in allTrials:
        trialName = "MVI_" +  str(trial[4])
        #if trialName!="MVI_3464":
        #    continue
        NUMFISH = int(trial[3])
        ncFileName = dataDir + "tracked/linked" + trialName + ".nc"
        print(ncFileName)
        f = Dataset.NetCDFFile(ncFileName, 'r', mmap=False)
        
        trialRes = np.zeros( shape=(0, 6) )
        # get the positions variable
        trXY = f.variables['trXY']
        trackList = []
        trackList = np.empty_like (trXY.data)
        np.copyto(trackList,trXY.data)
        [trackIndex,timeIndex]=np.nonzero(trackList[:,:,0])
        
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
    
        for t in range(np.min(timeIndex),np.max(timeIndex)):
            liveTracks = trackIndex[timeIndex[:]==t]
         
            for tr in liveTracks:
                xp = trackList[tr, t,0]
                yp = trackList[tr, t,1]
            
     
            
                fishCert[fishIDs[tr,t],t] = min( certIDs[tr,0],1.0)
                fishArm[fishIDs[tr,t],t] = arms[((int(yp),int(xp)))] # int(round(float(arms[((yp,xp))])/(0.5*255.0)))
                fishTrackNum[fishIDs[tr,t],t] = tr

        for thisFish in range(NUMFISH):
            soloTrans = 0
            followTrans = 0
            currentArms = fishArm[thisFish,0:6750].copy()
            currentTracks = fishTrackNum[thisFish,0:6750].copy()
            neighbours = np.zeros_like(currentArms)
            nArms = len(currentArms)
            for aaa in range(1,nArms):
                if currentArms[aaa]<0:
                    currentArms[aaa]=currentArms[aaa-1]
                for leadFish in range(NUMFISH):
                    if (thisFish==leadFish):
                        continue
                    if fishArm[leadFish,aaa]==currentArms[aaa]:
                        neighbours[aaa]+=1
            # points where the sum of the track IDs change are when a track is lost or found
            d= np.diff(currentArms)
            d2= np.diff(currentTracks)
            idx, = ((d!=0)&(d2==0)).nonzero()
            NSW = np.size(idx)
            #print( idx)
    
            for id in idx:
                myArm = currentArms[id + 1]
                empty = True
                followed = False
                for leadFish in range(NUMFISH):
                    if (thisFish==leadFish):
                        continue
                    
                
                    if fishArm[leadFish,id+1]==myArm:
                        empty = False
                    for pt in range(2,50):
                        if (id+pt)==nArms:
                            break
                        if fishArm[leadFish,id+pt]==myArm:
                            followed=True 
                            break
                if empty:
                    soloTrans+=1
                t2 = np.hstack((trial[4], thisFish, id, currentArms[id],currentArms[id+1], fishCert[thisFish,id]))
                trialRes = np.vstack((trialRes, t2))
#if NSW>0:
#                t2 = np.hstack((trial[0:3],soloTrans/float( NSW)))
#            else:
##                t2 = np.hstack((trial[0:3],0))
        results = np.vstack((results, trialRes[trialRes[:,2].argsort()] ))
        f.close()
    #print results
    return results#[results[:,2].argsort()]
    

if __name__ == "__main__":
    allData = main()
    #plt.boxplot([allData[np.logical_and(allData[:,1]==0 , allData[:,0]==0 ),3],allData[np.logical_and(allData[:,1]==0 , allData[:,0]==0 ),3],allData[np.logical_and(allData[:,1]==0 , allData[:,0]==0 ),3]],50,'')
 #   plt.boxplot([allData[allData[:,1]==0,4],allData[allData[:,1]==1,4],allData[allData[:,1]==2,4]],50,'')
 #   means = ([np.mean(allData[allData[:,1]==i,4]) for i in range(3)])
  #  plt.scatter([1, 2, 3], means)
  #  plt.show()
    np.savetxt('orderAnalysis.txt',allData, delimiter=',', fmt=['%d','%d','%d','%d','%d','%f'])
