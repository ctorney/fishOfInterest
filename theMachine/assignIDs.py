
import math
import numpy as np
from munkres import Munkres
import Scientific.IO.NetCDF as Dataset

def assignIDs(dataDir, trialName, NUMFISH):
    allScores = np.load("aS-" + trialName + ".npy")
    allLiveTracks = np.load("aLT-" + trialName + ".npy")
    mainTrackList = np.load("mTL-" + trialName + ".npy")
    # open netcdf file
    ncFileName = dataDir + "tracked/linked" + trialName + ".nc"    
    f = Dataset.NetCDFFile(ncFileName, 'a')

    # get the positions variable
    trXY = f.variables['trXY']
    trackList = []
    trackList = np.empty_like (trXY)
    np.copyto(trackList,trXY)
    [trackIndex,timeIndex]=np.nonzero(trackList[:,:,0])

    # get the movie frame numbers
    fid = f.variables['fid']
    fishIDs = []
    fishIDs = -np.ones_like(fid)

    for tr in range(NUMFISH):
        fishIDs[allLiveTracks[0,tr],:] = tr    
    
    #np.copyto(fishIDs, fid)
    cid = f.variables['certID']
    certIDs = []
    certIDs = np.empty_like(cid)

    perms = []
    permute(range(NUMFISH),perms)
    

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
    eps = 0.10
    accMat = pm*(1.0-eps) + eps*0.25*np.ones_like(pm)

    # autocorrelation parameter to discount the number of samples that are effectively pseudo-reps
    # i.e. 10 images that are assigned to a track should only count as 1 due to the high frame rate
    rho = 0.025

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
                        denom = np.inf
                    else:
                        denom += math.exp(d_add)
                
        
                ps[i,m] = math.log(denom)
    
        allCosts[block,:,:]=ps

    # now we have the negative log likelihood we find the assignment that minimizes this with the hungarian algorithm
    allAssign = np.zeros((numBlocks,NUMFISH,2))
    theStack = np.zeros((numBlocks))
    theStack[0]=np.nan
    
    # hhhheeeerrrrreeeeee!!!!!!!!!!!
    
    thisCost = np.zeros_like(allCosts[0,:,:])
    for block in range(1,numBlocks):
        np.copyto(thisCost, allCosts[block,:,:])
        # this is the assignment
        indexes = munk.compute(thisCost)
        np.copyto(thisCost, allCosts[block,:,:])
        allAssign[block,:,:]=indexes
        # next we calculate the cost of the assignment
        theStack[block] = 0.0
        for i in indexes:
            theStack[block] += thisCost[i]



    uncertainty = 0.0
    
    
    # now we loop through and take the minimum cost assignment off the stack
    while np.isfinite(theStack).any():
        calculateAllCosts(allAssign, allCosts, allLiveTracks, theStack, perms, fishIDs, trackList, trackIndex, timeIndex, NUMFISH)
        block = np.nanargmin(theStack)
        if theStack[block]>uncertainty:
            uncertainty = theStack[block]

        if block ==32:
            print block
        debugCost = allCosts[17]
        # if it doesn't assign to fish and set the cost to nan to indicate it's done
        for i in range(NUMFISH):
            thisTrack = int(allLiveTracks[block,i])
            if fishIDs[thisTrack,0]<0:
                certIDs[thisTrack,:] = float(uncertainty)
            fishIDs[thisTrack,:] = int(allAssign[block,i,1])
            # update all costs where this track is present            
            [b_inds, f_inds] = np.where(allLiveTracks==thisTrack)
            
            for j in range(np.size(b_inds)):
                oblk = b_inds[j]
                oind = f_inds[j]
                allCosts[oblk,oind,:] = np.inf
                allCosts[oblk,oind,int(allAssign[block,i,1])]=0
        print block
        print allCosts[17]
            
            
            

            
        print 'assigned fragment ' + str(block)
        #print allScores[block]
        #print allAssign[block]
        #print theStack[block]
      #  print uncertainty
     #   print mainTrackList[block,2], mainTrackList[block,3]
     #   print allLiveTracks[block]
      #  print '================'
        theStack[block]=np.nan
       


    fid.assignValue( (fishIDs))
    cid.assignValue( ( certIDs))
    f.sync()
    f.close()
    return

def calculateAllCosts(allAssign, allCosts, allLiveTracks, theStack, perms, fishIDs, trackList, trackIndex, timeIndex, NUMFISH):
    
    numBlocks = np.size(theStack,0)
    thisCost = np.zeros_like(allCosts[0,:,:])
    # first update the cost matrices based on the previous assignments
    for block in range(1,numBlocks):
        if np.isnan(theStack[block]): continue
        assignBAD = True
        while assignBAD:
            
            assignBAD = False
            np.copyto(thisCost, allCosts[block,:,:])
            # this is the assignment
            [indexes, stackCost] = findOptimalAssignment(thisCost, perms, NUMFISH)
            if np.isinf(stackCost): break
            for i in range(NUMFISH):
                t_id = allLiveTracks[block, i]
                fish = indexes[i,1]
                # we want to identify track t_id as fish
                # first find overlapping tracks
                otherTracks = np.unique(trackIndex[np.in1d(timeIndex,np.where(trackList[t_id, :, 0]>0)[0])])
                for ot in otherTracks:
                    if ot!=t_id:
                        if fishIDs[ot,0]==fish:
                            wrong =indexes[i,:]
                            allCosts[block,wrong[0],wrong[1]] = np.inf
                            assignBAD = True
            
       
        
        allAssign[block,:,:]=indexes
        theStack[block] = stackCost
        

def findOptimalAssignment(thisCost, perms, NUMFISH):

    n = np.size(perms,0)
    possCosts = np.zeros((2,math.factorial(NUMFISH)), dtype=float)
    
    for i in range(n):
        possCosts[0,i]=i
        for j in range(NUMFISH):
            possCosts[1,i]+= thisCost[j,perms[i][j]]


    # now sort the array by order of length
    
    ind = np.lexsort(possCosts)

    
    if possCosts[1,ind[1]]==0:
        return np.column_stack((range(NUMFISH),perms[ind[0]])), np.inf
    else:
        if np.isinf(possCosts[1,ind[0]]):
            return np.column_stack((range(NUMFISH),perms[ind[0]])), possCosts[1,ind[0]]
        else:
            return np.column_stack((range(NUMFISH),perms[ind[0]])), possCosts[1,ind[0]]/possCosts[1,ind[1]]
    
    
def permute(a, results):
    if len(a) == 1:
        results.insert(len(results), a)

    else:
        for i in range(0, len(a)):
            element = a[i]
            a_copy = [a[j] for j in range(0, len(a)) if j != i]
            subresults = []
            permute(a_copy, subresults)
            for subresult in subresults:
                result = [element] + subresult
                results.insert(len(results), result)
