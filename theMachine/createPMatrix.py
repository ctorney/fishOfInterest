from SimpleCV import Image,  VirtualCamera, Display, SVMClassifier
import numpy as np
import Scientific.IO.NetCDF as Dataset
import cv2


from circularHOGExtractor import circularHOGExtractor

def createPMatrix(dataDir, trialName, NUMFISH, mainTrackList):
    # movie and images
    movieName = dataDir + "sampleVideo/" + trialName + ".avi";
    bkGrnd = Image(dataDir + "bk.png")
    mask  = Image(dataDir + "maskmono.png")

    # open netcdf file
    ncFileName = dataDir + "tracked/linked" + trialName + ".nc"    
    f = Dataset.NetCDFFile(ncFileName, 'r')

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


    # these variables store the index values when a track is present
    [trackIndex,timeIndex]=np.nonzero(trackList[:,:,0])

    if np.size(mainTrackList==0):
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

    # load the classifier
    cl = SVMClassifier.load('svm' + trialName + '.xml')
    classes = []
    for tr in range(NUMFISH):
        classes.append(str(tr))

    # only do the complete segments
    numBlocks = np.sum(mainTrackList[:,0]==NUMFISH) 

    # arrays to store the scores and the track list for output
    allScores = np.zeros((numBlocks,NUMFISH,NUMFISH))
    allLiveTracks = np.zeros((numBlocks,NUMFISH))
 
    vir = VirtualCamera(movieName, "video")
    box_dim = 50    

    for nb in range( numBlocks):
        # get the parameters for the current track block 
        indexStart = mainTrackList[nb,2]
        indexStop = mainTrackList[nb,3]

        score = np.zeros((NUMFISH,NUMFISH))
        liveTracks = trackIndex[timeIndex[:]==indexStart]

        for fr in range(indexStart, indexStop):
            # extract the frame
            thisIm = vir.getFrame(trFrames[fr]).toGray()
            thisIm = Image(cv2.absdiff(thisIm.getGrayNumpyCv2(), bkGrnd.getGrayNumpyCv2()), cv2image=True)
            thisIm = thisIm.applyBinaryMask(mask)

            # look at each track
            for tr in range(NUMFISH):
                xp = trackList[liveTracks[tr], fr, 0]
                yp = trackList[liveTracks[tr], fr, 1]
                if xp>0:        
                    # extract the individual and classify
                    tmpImg = thisIm.crop(round(xp), round(yp), box_dim,box_dim, centered=True)
                    fishGuess = cl.classify(tmpImg)
                    # record the score for this track
                    score[tr,int(fishGuess)]+=1

           
        # store the total score matrix for assigning each track to each possible identity 
        allScores[nb,:,:] = score
        allLiveTracks[nb,:] = liveTracks



    f.sync()
    f.close()

    np.save("aS-" + trialName + ".npy", allScores)
    np.save("aLT-" + trialName + ".npy", allLiveTracks)
    np.save("mTL-" + trialName + ".npy", mainTrackList)
    return


