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

    ncfile = Dataset.NetCDFFile(ncFileName, 'a', mmap=False)
    

    # get the positions variable
    trXY = ncfile.variables['trXY']
    trackList = []
    trackList = np.empty_like (trXY.data)
    np.copyto(trackList,trXY.data)
    [trackIndex,timeIndex]=np.nonzero(trackList[:,:,0])

    
    # get the movie frame numbers
    # get the movie frame numbers
    frNum = ncfile.variables['frNum']
    trFrames = []
    trFrames = np.empty_like(frNum.data)
    np.copyto(trFrames,frNum.data)
    
    fid = ncfile.variables['fid']
    fishIDs = []
    fishIDs = np.empty_like(fid.data)
    np.copyto(fishIDs, fid.data)
    cid = ncfile.variables['certID']
    certIDs = []
    certIDs = np.empty_like(cid.data)
    np.copyto(certIDs, cid.data)
    



    # these variables store the index values when a track is present
    
    tCount = np.shape(trackList)[0]
    
    for f in range(tCount):
        # delete anything less than a second
        [vals] = np.nonzero(trackList[f,:,0])
        if len(vals)<24:
            trackList[f,vals,0]=0
            trackList[f,vals,1]=0
    [trackIndex,timeIndex]=np.nonzero(trackList[:,:,0])
    realCount = len(np.unique(trackIndex))
    
    print(str(realCount) + ' tracks left' )
    #ids = np.unique(posDF['c_id'].values)
    newids = np.arange(0,tCount)#np.unique(posDF['c_id'].values)
    trackVals = np.zeros((tCount,7)) # start stop xstart ystart xstop ystop length
    
    for f in range(tCount):
        [vals] = np.nonzero(trackList[f,:,0])
        if len(vals):
            trackVals[f,0]=vals[0]
            trackVals[f,1]=vals[-1]
            trackVals[f,2]=trackList[f,vals,0][0]
            trackVals[f,3]=trackList[f,vals,1][0]
            trackVals[f,4]=trackList[f,vals,0][-1]
            trackVals[f,5]=trackList[f,vals,1][-1]
            trackVals[f,6]=vals[-1]-vals[0]
        
    tListDesc = np.argsort(trackVals[:,6])[::-1]
    #break
    timediff = 120 # 10 seconds
    dist = 20
    cap = cv2.VideoCapture(movieName)
    box_dim=512
    nx = 1920
    ny = 1080
    sz=16
    frName = 'is this the same individual? y or n'
    cv2.destroyAllWindows()
    cv2.namedWindow(frName, flags =  cv2.WINDOW_NORMAL)
    escaped=False
    for ipos in range(tCount):
        i = tListDesc[ipos]
        if trackVals[i,1]==0: continue # empty
        found = False
        skip=False
        brexit = False
        for j in range(i+1,tCount):
            
            if trackVals[j,1]==0: continue # empty
            if trackVals[j,0]<(trackVals[i,1]): continue # track j starts  before i finishes
            if trackVals[j,0]>(trackVals[i,1]+timediff): continue # track j starts too long after i finishes
            distance = ((trackVals[j,2]-trackVals[i,4])**2+(trackVals[j,3]-trackVals[i,5])**2)**0.5
           
            
            startFrame = int(max(0,trackVals[i,1]-25))
            stopFrame = int(min(trackVals[j,0]+25,np.shape(trackList)[1]))
            cap.set(cv2.CAP_PROP_POS_FRAMES,trFrames[startFrame])
            #ipos = posDF[posDF['c_id']==i]
            #ipos = ipos[ipos['frame']>startFrame]
            #jpos = posDF[posDF['c_id']==j]
            #jpos = jpos[jpos['frame']<stopFrame]
            cix = trackVals[i,4]
            ciy = trackVals[i,5]
            thisFrame=startFrame
            speed = 50
            while True:
                thisFrame = thisFrame + 1#cap.get(cv2.CAP_PROP_POS_FRAMES)
                if thisFrame>stopFrame:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,trFrames[startFrame])
                    thisFrame=startFrame
                _, frame = cap.read()
                
                
                ix = int(trackList[i,thisFrame,0])
                iy = int(trackList[i,thisFrame,1])
                jx = int(trackList[j,thisFrame,0])
                jy = int(trackList[j,thisFrame,1])
                
                cv2.circle(frame, (ix,iy),3,(0,0,255),-2)
                cv2.circle(frame, (jx,jy),4,(155,255,155),3)
                
                tmpImg = frame[max(0,ciy-box_dim):min(ny,ciy+box_dim), max(0,cix-box_dim):min(nx,cix+box_dim)]
                
                
       
                cv2.imshow(frName,tmpImg)
                k = cv2.waitKey(speed) & 0xFF
                
                if k==ord('y'):
                    found=True
                    break
                if k==ord('n'):
                    break
                if k==ord('s'):
                    skip=True
                    break
                if k==ord('m'):
                    speed = speed + 1
                if k==ord('p'):
                    speed = max(1,speed - 1)
                    
                
                if k==27:    # Esc key to stop
                    escaped=True
                    break 
                if k==ord('x'):
                    brexit=True
                    break
            if found:
                if newids[j]>newids[i]:
                    realCount-=1
                    print(str(j)+' changed to '+str(i) + ', ' + str(realCount) + ' tracks left' )
                    newids[j]=newids[i]
                    break
            if escaped:
                break
            if skip:
                break
            if brexit:
                break
        if escaped:
            break
        if brexit:
            break
            
            
          
      

    cv2.destroyAllWindows()
    
    for i in range(tCount):
        
        if newids[i]==i: continue
        j = newids[i]
        [valsi] = np.nonzero(trackList[i,:,0])
        [valsj] = np.nonzero(trackList[j,:,0])
        trackList[j,valsi,0]=trackList[i,valsi,0]
        trackList[j,valsi,1]=trackList[i,valsi,1]
        trackList[i,valsi,0]=0
        trackList[i,valsi,1]=0
        newids[newids==i]=j

        startOfGap=valsj[-1]
        endOfGap=valsi[0]

        for k in range(startOfGap,endOfGap):
            # linear interpolation of point
            trackList[j,k,0] = trackList[j,startOfGap,0] + (float(k-startOfGap)/float(endOfGap-startOfGap))*(trackList[j,endOfGap,0]-trackList[j,startOfGap,0])
            trackList[j,k,1] = trackList[j,startOfGap,1] + (float(k-startOfGap)/float(endOfGap-startOfGap))*(trackList[j,endOfGap,1]-trackList[j,startOfGap,1])
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
    np.copyto(trXY.data,trackList)
    
    ncfile.sync()
    ncfile.close()
    if brexit:
        break

#    break
