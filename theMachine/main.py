import numpy as np
import Scientific.IO.NetCDF as Dataset
import cv2
import os
from createSampleImages import createSampleImages
from trainClassifier import trainClassifier
from createPMatrix import createPMatrix
from assignIDs import assignIDs
import sys


def main():

    dataDir = '/home/ctorney/data/fishPredation/'

    trialName = "MVI_" +  str(sys.argv[1])
    NUMFISH = int(sys.argv[2])
    #trialName = "MVI_3370"
    #NUMFISH = 2
    print trialName, NUMFISH

    if not os.path.exists(dataDir + '/process/' + trialName):
            os.makedirs(dataDir + '/process/' + trialName)
    for tr in range(NUMFISH):
        direct = dataDir + '/process/' + trialName + '/FR_ID' + str(tr)
        if not os.path.exists(direct):
            os.makedirs(direct)

    mainTrackList = []
    print "creating sample images ..."
    mainTrackList = createSampleImages(dataDir, trialName )
    
    print "training the classifier ..."
    trainClassifier(dataDir, trialName, NUMFISH)

    print "creating the probability matrix for each track  ..."
    createPMatrix(dataDir, trialName, NUMFISH, mainTrackList)

    print "assign fish IDs to each track  ..."
    assignIDs(dataDir, trialName, NUMFISH)

    return
    



if __name__ == "__main__":
   main()
