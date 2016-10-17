import numpy as np
import pandas as pd
import os
import scipy.io.netcdf as Dataset

import cv2


dataDir = '/home/ctorney/data/fishPredation/'
allTrials = np.loadtxt('relinkList.txt',delimiter=',',dtype=int)

for trial in allTrials:
    trialName = "MVI_" +  str(trial[3])
    movieName = dataDir + "allVideos/" + trialName + ".MOV";
    NUMFISH = int(trial[2])
    ncFileName = dataDir + "tracked/linked" + trialName + ".nc"
    ncOutName = dataDir + "tracked/relinked" + trialName + ".nc"

    f = Dataset.NetCDFFile(ncFileName, 'r', mmap=False)
    

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
    
    tCount = np.shape(trackList)[0]
    
    #ids = np.unique(posDF['c_id'].values)
    newids = range(0,tCount)#np.unique(posDF['c_id'].values)
    trackVals = np.zeros((tCount,6)) # start stop xstart ystart xstop ystop
    
    for f in range(tCount):
        [vals] = np.nonzero(trackList[f,:,0])
        if len(vals):
            trackVals[f,0]=vals[0]
            trackVals[f,1]=vals[-1]
            trackVals[f,2]=trackList[f,vals,0][0]
            trackVals[f,3]=trackList[f,vals,1][0]
            trackVals[f,4]=trackList[f,vals,0][-1]
            trackVals[f,5]=trackList[f,vals,1][-1]
        

    #break
    timediff = 1200 # 10 seconds
    dist = 100
    cap = cv2.VideoCapture(movieName)
    box_dim=256
    nx = 1920
    ny = 1080
    sz=16
    frName = 'is this the same caribou? y or n'
    cv2.destroyAllWindows()
    cv2.namedWindow(frName, flags =  cv2.WINDOW_NORMAL)
    escaped=False
    for i in range(tCount):
        if trackVals[i,1]==0: continue # empty
        found = False
        skip=False
        for j in range(i+1,tCount):
            
            if trackVals[j,1]==0: continue # empty
            if trackVals[j,0]<(trackVals[i,1]-120): continue # track j starts more than 2 secs before i finishes
            if trackVals[j,0]>(trackVals[i,1]+timediff): continue # track j starts too long after i finishes
            distance = ((trackVals[j,2]-trackVals[i,4])**2+(trackVals[j,3]-trackVals[i,5])**2)**0.5
            print(distance)
            if distance>500: continue
            print(j)
            
            
            startFrame = max(0,trackVals[i,1]-120)
            stopFrame = trackVals[j,0]+120
            cap.set(cv2.CAP_PROP_POS_FRAMES,startFrame)
            #ipos = posDF[posDF['c_id']==i]
            #ipos = ipos[ipos['frame']>startFrame]
            #jpos = posDF[posDF['c_id']==j]
            #jpos = jpos[jpos['frame']<stopFrame]
            ix = trackVals[i,4]
            iy = trackVals[i,5]
            
            
            while True:
                thisFrame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if thisFrame>stopFrame:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,startFrame)
                _, frame = cap.read()
                
                
                ix = int(trackList[i,thisFrame,0])
                iy = int(trackList[i,thisFrame,1])
                jx = int(trackList[j,thisFrame,0])
                jy = int(trackList[j,thisFrame,1])
                
                cv2.circle(frame, (ix,iy),3,(0,0,255),-2)
                cv2.circle(frame, (jx,jy),4,(155,255,155),3)
                 
                tmpImg = frame[max(0,iy-box_dim/2):min(ny,iy+box_dim/2), max(0,ix-box_dim/2):min(nx,ix+box_dim/2)]
                
                
       
                cv2.imshow(frName,frame)
                k = cv2.waitKey(10)
                
                if k==ord('y'):
                    found=True
                    break
                if k==ord('n'):
                    break
                if k==ord('s'):
                    skip=True
                    break
                
                if k==27:    # Esc key to stop
                    escaped=True
                    break 
            if found:
                print(str(j)+' changed to '+str(i))
                newids[j]=newids[i]
                break
            if escaped:
                break
            if skip:
                break
        if escaped:
            break
            
            
          
      

    cv2.destroyAllWindows()
    
#    for thisID in ids:
#        posDF.loc[posDF['c_id']==thisID,'c_id']=newids[ids==thisID]
#    postmean = 0.0
#    countID=0
#    for cnum, cpos in posDF.groupby('c_id'):
#        postmean = postmean +  (cpos['frame'].iloc[-1]-cpos['frame'].iloc[0])
#        countID=countID+1
#
#    postmean = postmean/(countID)
#    print(str(premean/60)+" before, now: "+str(postmean/60))
#    #posDF.to_csv(outfilename)
    if escaped:
        break

#    break
