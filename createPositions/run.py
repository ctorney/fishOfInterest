
import numpy as np
import os


dataDir = '/home/ctorney/data/fishPredation/'
allTrials = np.loadtxt('trialList.txt',delimiter=',',dtype=int)
for trial in allTrials:
    trialName = "MVI_" +  str(trial[3])
    NUMFISH = str(trial[2])
    os.system("./build/createPositions " + trialName + " " + NUMFISH + " >> output.txt") 
    print(trialName + " completed")
