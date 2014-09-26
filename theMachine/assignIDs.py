

import numpy as np
from munkres import Munkres
import Scientific.IO.NetCDF as Dataset

def assignIDs(dataDir, trialName, NUMFISH, mainTrackList, allScores, allLiveTracks):
    # open netcdf file
    ncFileName = dataDir + "tracked/linked" + trialName + ".nc"    
    f = Dataset.NetCDFFile(ncFileName, 'a')

    # get the positions variable
    trXY = f.variables['trXY']
    trackList = []
    trackList = np.empty_like (trXY)
    np.copyto(trackList,trXY)

    # get the movie frame numbers
    fid = f.variables['fid']
    fishIDs = []
    fishIDs = np.empty_like(fid)
    np.copyto(fishIDs, fid)



    # these variables store the index values when a track is present
    [trackIndex,timeIndex]=np.nonzero(trackList[:,:,0])
    munk = Munkres()

    # convert the scores to an accuracy matrix for the classifier
    # this accounts for the full array of probabilities so down weights
    # assignments based on classification between individuals that are
    # difficult to distinguish
    pm = np.zeros_like(allScores[0])

    for i in range(NUMFISH):
        pm[i,:] = allScores[0,fishIDs[int(allLiveTracks[0,i]),0],:]

    pm = pm/np.sum(pm,axis=1)

    # print the score matrix - this determines the accuracy of the classifier
    print trialName
    print '==========='
    print pm
    
    # introduce a small error to account for the higher accuracy of training and testing on same set
    eps = 0.50
    accMat = pm*(1.0-eps) + eps*0.25*np.ones_like(pm)


    allCosts = np.zeros_like(allScores)
    numBlocks = np.size(allScores,0)

    # now we loop through an calculate a cost value for each assignment to each track
    # the cost value is defined as the negative log likelihood, and the assignment that 
    # minimizes this value will be used
    for block in range(1,numBlocks):
        thisScore = allScores[block]
        ps = np.ones((NUMFISH,NUMFISH))
        # for each block and each track i, we find P(id = i|X) where X is the sequence of observations
        # we use the individual assignment accuracy and Bayes theorem assuming each assignment is a priori
        # equally likely. Due to numerical precision the expression is rearranged and logs taken
        for i in range(NUMFISH):
            for m in range(NUMFISH):
                denom = 0.0
                for j in range(NUMFISH):
                    d_add = 0.0
                    for k in range(NUMFISH):
                        d_add += (math.log(accMat[j,k])-math.log(accMat[m,k]))*thisScore[i,k]
                    
                    if d_add>709.0:
                        denom = inf
                    else:
                        denom += math.exp(d_add)
                
        
                ps[i,m] = math.log(denom)
    
        allCosts[block,:,:]=ps

    # now we have the negative log likelihood we find the assignment that minimizes this with the hungarian algorithm
    allAssign = np.zeros((numBlocks,NUMFISH,2))
    allMunk = np.zeros((numBlocks))
    allMunk[0]=np.nan
    thisCost = np.zeros_like(allCosts[0,:,:])
    for block in range(1,numBlocks):
        np.copyto(thisCost, allCosts[block,:,:])
        # this is the assignment
        indexes = munk.compute(thisCost)
        np.copyto(thisCost, allCosts[block,:,:])
        allAssign[block,:,:]=indexes
        # next we calculate the cost of the assignment
        allMunk[block] = 0.0
        for i in indexes:
            allMunk[block] += thisCost[i]

    # now we loop through and take the minimum cost assignment off the stack
    while np.isfinite(allMunk).any():
        block = np.nanargmin(allMunk)
        # check if it doesn't contradict an earlier assignment
        imPoss = checkSanity(allAssign[block,:,:], allLiveTracks[block,:], fishIDs, trackList)
        if (imPoss<0):
            # if it doesn't assign to fish and set the cost to nan to indicate it's done
            for i in range(NUMFISH):
                thisTrack = int(allLiveTracks[block,i])
                fishIDs[thisTrack,:] = int(allAssign[block,i,1])
            allMunk[block]=np.nan
        else:
            # if it's impossible we set the cost of the assignmet to infinity
            wrong = allAssign[block,imPoss,:]
            allCosts[block,wrong[0],wrong[1]] = np.inf
            np.copyto(thisCost, allCosts[block,:,:])
            # we make this assignment very expensive (munk can't handle inf)
            thisCost[np.isinf(thisCost)] = 1e12
            indexes = munk.compute(thisCost)
            np.copyto(thisCost, allCosts[block,:,:])
            # then store this assignment and cost - even if it's infinity, 
            # if that's the case then it will be ignored and assumed to be unclassifiable
            allAssign[block,:,:]=indexes
            allMunk[block] = 0.0
            for i in indexes:
                allMunk[block] += thisCost[i]

    fid.assignValue( (fishIDs))
    f.sync()
    f.close()
    return

def checkSanity(assignment, tracks, fishIDs, trackList):

    # check that this doesn't contradict an earlier assignment
    [trackIndex,timeIndex]=np.nonzero(trackList[:,:,0])
    for i in range(NUMFISH):
        t_id = tracks[i]
        fish = assignment[i,1]
        # we want to identify track t_id as fish
        # first find overlapping tracks
        otherTracks = np.unique(trackIndex[np.in1d(timeIndex,np.where(trackList[t_id, :, 0]>0)[0])])
        for ot in otherTracks:
            if ot!=t_id:
                if fishIDs[ot,0]==fish:
                    return i
        
    return -1

