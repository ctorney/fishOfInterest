
import numpy as np
import cv2
import os
from createSampleImages import createSampleImages
from trainClassifier import trainClassifier
from createPMatrix import createPMatrix
from assignIDs import assignIDs
import sys


def main():
    dataDir = "/media/ctorney/SAMSUNG/data/fishPredation/"

 #   dataDir = '/home/ctorney/data/fishPredation/'
    allTrials = np.loadtxt('redoTrials.txt',delimiter=',',dtype=int)
    if len(np.shape(allTrials))==1:
        NUMFISH = str(allTrials[2])
        trialName = "MVI_" +  str(allTrials[3])
        os.system("python3 main.py " + trialName + " " + NUMFISH + " >> log.txt")
        
    else:
            for trial in allTrials:
                NUMFISH = str(trial[2])
                trialName = "MVI_" +  str(trial[3])
                os.system("python3 main.py " + trialName + " " + NUMFISH + " >> log.txt")

    return
        



if __name__ == "__main__":
   main()


