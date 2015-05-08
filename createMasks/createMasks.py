
import numpy as np
import cv2
import os


dataDir = '/home/ctorney/data/fishPredation/'
allTrials = np.loadtxt('trialList.txt',delimiter=',',dtype=int)
for trial in allTrials:
    trialName = "MVI_" +  str(trial[3])
    
    
    bkGrnd = cv2.imread(dataDir + "backGrounds/bk-" + trialName + ".png")
    bkGrnd = cv2.cvtColor(bkGrnd, cv2.COLOR_BGR2GRAY)
    mask = cv2.imread(dataDir + "maskmono.png",cv2.IMREAD_GRAYSCALE)
    #thisIm = bkGrnd.applyBinaryMask(mask)
    thisIm = cv2.bitwise_and(bkGrnd,mask)
    cv2.imshow('ok?',thisIm)  
    k = cv2.waitKey(0)
    if k==27:    # Esc key to stop
        break
    elif k==ord('y'):
        cv2.imwrite(mask, dataDir + "masks/mask-" + trialName + ".png")
    elif k==ord('n'):
        cv2.imwrite(thisIm,dataDir + "masks/mask-" + trialName + ".png")
    
    



