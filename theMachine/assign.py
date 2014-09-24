

import SimpleCV
import time
import sympy

from SimpleCV import Image, VirtualCamera, Display, Features, TreeClassifier, ImageSet, Color
from SimpleCV import SVMClassifier, TreeClassifier, KNNClassifier
import sys
import random
import math


pm = allScores[0]
pm = pm/np.sum(pm,axis=1)


eps = 0.05 # introduce a small error to account for the higher accuracy of training and testing on same set
accMat = pm*(1.0-eps) + eps*np.ones_like(pm)

P = np.zeros_like(allScores[1])
thisScore = allScores[1]

for track in range(NUMFISH):
    ss = np.sum(thisScore[track,:])
    for fish in range(NUMFISH):
       # whats the probability track is fish
       # this is prob classifier is right when track=fish and wrong the other times
       cc = thisScore[track,fish]
       pp = accMat[track,fish] #sympy.binomial(ss,thisScore[track,fish])*accMat[track,fish]**thisScore[track,fish] * (1.0-accMat[track,fish])**(ss-thisScore[track,fish])
       #pp = binomial_p(accMat[track,fish], int(ss), int(thisScore[track,fish]))
       P[track,fish] = math.log(pp/(1.0-pp))*(2.0*cc - ss)
       
       
def binomial_p(p1, n, k):

    result = 1.0
    nocount = n-k

    for i in range(1,k+1):
        result *= ((n - (k - i)) / float(i))
        result *= p1
        if nocount>0:
            result *= (1.0-p1)
            nocount -= 1

    for i in range(nocount):
        result *= (1.0-p1)  
    return float(result)