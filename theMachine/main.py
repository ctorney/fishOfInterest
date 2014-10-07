from SimpleCV import Image,  VirtualCamera, Display, Color
import numpy as np
import Scientific.IO.NetCDF as Dataset
import cv2
import os
from createSampleImages import createSampleImages
from trainClassifier import trainClassifier
from createPMatrix import createPMatrix
from assignIDs import assignIDs

def main():
    dataDir = '/home/ctorney/data/fishPredation/'
    trialName = "MVI_3371"

    NUMFISH = 4

    for tr in range(NUMFISH):
        direct = trialName + str(tr)
        if not os.path.exists(direct):
            os.makedirs(direct)

    
    mainTrackList = []
    print "creating sample images ..."
    #mainTrackList = createSampleImages(dataDir, trialName )
    
    print "training the classifier ..."
#    trainClassifier(trialName, NUMFISH)

    print "creating the probability matrix for each track  ..."
    #createPMatrix(dataDir, trialName, NUMFISH, mainTrackList)

    print "assign fish IDs to each track  ..."
    assignIDs(dataDir, trialName, NUMFISH)

    return
    



if __name__ == "__main__":
   main()
