

import SimpleCV
import time
import sympy

from SimpleCV import Image, VirtualCamera, Display, Features, TreeClassifier, ImageSet, Color
from SimpleCV import SVMClassifier, TreeClassifier, KNNClassifier
import sys
import random
import math
from munkres import Munkres

munk = Munkres()

pm = np.zeros_like(allScores[0])


for i in range(NUMFISH):
    pm[i,:] = allScores[0,fishIDs[int(allLiveTracks[0,i]),0],:]

pm = pm/np.sum(pm,axis=1)

eps = 0.50# introduce a small error to account for the higher accuracy of training and testing on same set
accMat = pm*(1.0-eps) + eps*0.25*np.ones_like(pm)

P = np.zeros_like(allScores[1])




allCosts = np.zeros_like(allScores)

numBlocks = np.size(allScores,0)

for block in range(1,numBlocks):
    thisScore = allScores[block]
    ps = np.ones((NUMFISH,NUMFISH))
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

print ps
#first vs second
allAssign = np.zeros((numBlocks,NUMFISH,2))
allMunk = np.zeros((numBlocks))
allMunk[0]=np.nan
thisCost = np.zeros_like(allCosts[0,:,:])
for block in range(1,numBlocks):
    np.copyto(thisCost, allCosts[block,:,:])
    indexes = munk.compute(thisCost)
    np.copyto(thisCost, allCosts[block,:,:])
    allAssign[block,:,:]=indexes
    allMunk[block] = 0.0
    for i in indexes:
        allMunk[block] += thisCost[i]

while np.isfinite(allMunk).any():
    block = np.nanargmin(allMunk)
    imPoss = checkSanity(allAssign[block,:,:], allLiveTracks[block,:], fishIDs, trackList)
    if (imPoss<0):
        #assign to fish --------
        for i in range(NUMFISH):
            thisTrack = int(allLiveTracks[block,i])
            fishIDs[thisTrack,:] = int(allAssign[block,i,1])
        allMunk[block]=np.nan
    else:
        wrong = allAssign[block,imPoss,:]
        allCosts[block,wrong[0],wrong[1]] = np.inf
        np.copyto(thisCost, allCosts[block,:,:])
        thisCost[np.isinf(thisCost)] = 1e12
        indexes = munk.compute(thisCost)
        np.copyto(thisCost, allCosts[block,:,:])
        allAssign[block,:,:]=indexes
        allMunk[block] = 0.0
        for i in indexes:
            allMunk[block] += thisCost[i]
    print np.sum(np.isfinite(allMunk))

# find min

# check sane    
def checkSanity(assignment, tracks, fishIDs, trackList):

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

