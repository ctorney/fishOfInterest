#import SimpleCV
from SimpleCV import Image,  VirtualCamera, Display, Color
import numpy as np
import Scientific.IO.NetCDF as Dataset
#from netCDF4 import Dataset
import cv2
import os

def createSampleImages(dataDir, trialName, mainTrackList):
   
    # movie and images
    movieName = dataDir + "sampleVideo/" + trialName + ".avi";
    bkGrnd = Image(dataDir + "bk.png")
    mask  = Image(dataDir + "maskmono.png")

    # open netcdf file
    ncFileName = dataDir + "tracked/linked" + trialName + ".nc"    
    f = Dataset.NetCDFFile(ncFileName, 'a')

    # get the positions variable
    trXY = f.variables['trXY']
    trackList = []
    trackList = np.empty_like (trXY)
    np.copyto(trackList,trXY)

    # get the movie frame numbers
    frNum = f.variables['frNum']
    trFrames = []
    trFrames = np.empty_like(frNum)
    np.copyto(trFrames,frNum)

    # get the fish identities
    fid = f.variables['fid']  
    fishIDs = -np.ones(np.shape(fid),'int16')
    
    # these variables store the index values when a track is present
    [trackIndex,timeIndex]=np.nonzero(trackList[:,:,0])

    # to find where tracks come and go we sum all the track IDs at each time point
    maxTime = np.size(trackList,1)
    trackSum = np.zeros([maxTime])
    trackCount = np.zeros([maxTime])
    for k in range(maxTime): 
        trackSum[k]=sum(trackIndex[timeIndex[:]==k])
        trackCount[k]=np.size(trackIndex[timeIndex[:]==k])

    # points where the sum of the track IDs change are when a track is lost or found
    d= np.diff(trackSum)
    # mark the start and end points
    d[0]=1
    d[-1]=1
    # now find where the tracks change
    idx, = d.nonzero()
    # because diff calculates f(n)-f(n+1) we need the next entry found for all except d[0]
    idx[idx>0]=idx[idx>0]+1
    # the difference between the indices gives the length of each block of continuous tracks
    conCounts =  idx[1:-1] - idx[0:-1-1]
   
    # array of all tracks || length of time tracks are present || number of tracks || start index || stop index || start frame of movie || stop frame of movie
    mainTrackList = np.column_stack((trackCount[idx[0:-1-1]], conCounts, idx[0:-1-1],idx[1:-1]))
    
    # now sort the array by order of length
    ind = np.lexsort((-mainTrackList[:,1],-mainTrackList[:,0]))
    mainTrackList = mainTrackList[ind,:].astype(int)

    numBlocks =  np.size(mainTrackList,0)

    # get the parameters for the longest track block 
    thisTrackCount = mainTrackList[0,0]
    indexStart = mainTrackList[0,2]

    liveTracks = trackIndex[timeIndex[:]==indexStart]

    startIndex = np.min(timeIndex[np.in1d(trackIndex,liveTracks)])
    stopIndex = np.max(timeIndex[np.in1d(trackIndex,liveTracks)])


    # now go through and store images from each track in a separate folder 
    box_dim = 50    
    vir = VirtualCamera(movieName, "video")
    for fr in range(startIndex, stopIndex):
        thisIm = vir.getFrame(trFrames[fr]).toGray()
        thisIm = Image(cv2.absdiff(thisIm.getGrayNumpyCv2(), bkGrnd.getGrayNumpyCv2()), cv2image=True)
        thisIm = thisIm.applyBinaryMask(mask)
        
        for tr in range(thisTrackCount):
            xp = trackList[liveTracks[tr], fr,0]
            yp = trackList[liveTracks[tr], fr,1]
            if xp>0:
                direct = trialName + str(tr) 
                tmpImg = thisIm.crop(round(xp), round(yp), box_dim, box_dim, centered=True)
                save_path = direct + "/img-" + str(fr) + ".png"
                tmpImg.save(save_path)
            
    # store the IDs     
    for tr in range(thisTrackCount):
        fishIDs[liveTracks[tr],:] = tr
        
    fid.assignValue( (fishIDs))
    f.sync()
    f.close()
    
    return
    



