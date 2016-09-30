

import numpy as np
import time
#import Scientific.IO.NetCDF as Dataset
import scipy.io.netcdf as Dataset
import cv2
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

from circularHOGExtractor import circularHOGExtractor

def createPMatrix(dataDir, trialName, NUMFISH, mainTrackList):
    # movie and images
    movieName = dataDir + "allVideos/" + trialName + ".MOV"
    bkGrnd = cv2.imread(dataDir + "backGrounds/bk-" + trialName + ".png")
    bkGrnd = cv2.cvtColor(bkGrnd, cv2.COLOR_BGR2GRAY)
    mask = cv2.imread(dataDir + "mask.png")


    # open netcdf file
    ncFileName = dataDir + "tracked/linked" + trialName + ".nc"    
    f = Dataset.NetCDFFile(ncFileName, 'r',mmap=False)

    # get the positions variable
    trXY = f.variables['trXY']
    trackList = []
    trackList = np.empty_like (trXY.data)
    np.copyto(trackList,trXY.data)

    # get the movie frame numbers
    frNum = f.variables['frNum']
    trFrames = []
    trFrames = np.empty_like(frNum.data)
    np.copyto(trFrames,frNum.data)

    ch = circularHOGExtractor(6,4,3) 
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
        conCounts =  idx[1:] - idx[0:-1]

        # array of all tracks || length of time tracks are present || number of tracks || start index || stop index || start frame of movie || stop frame of movie
        mainTrackList = np.column_stack((trackCount[idx[0:-1]], conCounts, idx[0:-1],idx[1:]))

        # now sort the array by order of length
        ind = np.lexsort((-mainTrackList[:,1],-mainTrackList[:,0]))
        mainTrackList = mainTrackList[ind,:].astype(int)

    # load the classifier



    gnb = pickle.load( open( dataDir + '/process/' + trialName + '/boost' + trialName + '.p',"rb"))

    # only do the complete segments
    numBlocks = np.sum(mainTrackList[:,0]==NUMFISH) 

    # arrays to store the scores and the track list for output
    allScores = np.zeros((numBlocks,NUMFISH,NUMFISH))
    allLiveTracks = np.zeros((numBlocks,NUMFISH))


    

    box_dim = 50
    bd2 = int(box_dim*0.5)


    #cv2.NamedWindow("w1", cv2.WINDOW_AUTOSIZE)
    blockList = np.zeros( shape=(0, 3) )

    print(numBlocks)
    for nb in range( numBlocks):
        indexStart = mainTrackList[nb,2]
        indexStop = mainTrackList[nb,3]
        blockList= np.vstack((blockList,[indexStart,indexStop,nb]))
    
#    ind = np.lexsort(blockList[:,1])
    ind = np.argsort((blockList[:,0]))

    blockList = blockList[ind,:].astype(int)
    

    frameCounter = 0
    cap = cv2.VideoCapture(movieName)
    for block in blockList:
        
        # get the parameters for the current track block 
        indexStart = block[0]#mainTrackList[nb,2]
        indexStop = block[1]#mainTrackList[nb,3]
        nb=block[2]

        score = np.zeros((NUMFISH,NUMFISH))
        liveTracks = trackIndex[timeIndex[:]==indexStart]

        while (frameCounter<trFrames[indexStart]):
            ret, frame = cap.read()
            frameCounter+=1
        
        #cap.set(cv2.CAP_PROP_POS_FRAMES,trFrames[indexStart])
        for fr in range(indexStart, indexStop):
            # extract the frame
           
            
            ret, frame = cap.read()
            frameCounter+=1
            thisIm = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #       cv2.imshow('frame',thisIm)
    #        cv2.waitKey(0)
            thisIm = cv2.absdiff(thisIm, bkGrnd)
            
            # look at each track
            for tr in range(NUMFISH):
                
                xp = trackList[liveTracks[tr], fr, 0]
                yp = trackList[liveTracks[tr], fr, 1]
                if xp>0:
                    # extract the individual and classify
                    tmpImg = thisIm[int(round(yp))-bd2:int(round(yp))+bd2, int(round(xp))-bd2:int(round(xp))+bd2]
                    #cv2.imshow('w1',tmpImg)
                    #cv2.waitKey(1)
                    
                    if tmpImg.shape[0]==box_dim and tmpImg.shape[1]==box_dim:
                        #direct = dataDir + '/process/' + trialName + '/FR_ID_TEMP'
                        #save_path = direct + "/img-" + str(liveTracks[tr])+"-" + str(fr) + ".png"
                        #cv2.imwrite(save_path, tmpImg)
                        features =  np.hstack((ch.extract(tmpImg), np.mean(tmpImg)))
                        fishGuess = gnb.predict(features)
                        # record the score for this track
                        score[tr,int(fishGuess)]+=1

        # store the total score matrix for assigning each track to each possible identity 
        allScores[nb,:,:] = score
        allLiveTracks[nb,:] = liveTracks
    cap.release()


    
    f.sync()
    f.close()
    np.save(dataDir + '/process/' + trialName + "/aS-" + trialName + ".npy", allScores)
    np.save(dataDir + '/process/' + trialName + "/aLT-" + trialName + ".npy", allLiveTracks)
    np.save(dataDir + '/process/' + trialName + "/mTL-" + trialName + ".npy", mainTrackList)
    return


