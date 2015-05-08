
import sys, os
import numpy as np
import cv2
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from circularHOGExtractor import circularHOGExtractor

def trainClassifier(dataDir, trialName, NUMFISH):


    
    ch = circularHOGExtractor(5,5,6) 
    nFeats = ch.getNumFields()
    trainData = np.array([])#np.zeros((len(lst0)+len(lst0c)+len(lst1),nFeats))
    targetData = np.array([])#np.hstack((np.zeros(len(lst0)+len(lst0c)),np.ones(len(lst1))))
    for tr in range(NUMFISH):
        directory = dataDir + '/process/' + trialName + '/FR_ID' + str(tr) + '/'
        files = [name for name in os.listdir(directory)]
        thisData = np.zeros((len(files),nFeats))
        thisTarget = tr*np.ones(len(files))
        i = 0
        for imName in files:
            sample = cv2.imread(directory + imName)
            thisIm = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
            
            thisData[i,:] = ch.extract(thisIm)            
            i = i + 1
        trainData = np.vstack((trainData, thisData)) if trainData.size else thisData
        targetData = np.hstack((targetData, thisTarget)) if targetData.size else thisTarget

    clf = svm.SVC()
    #gnb = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME",n_estimators=200)
    y_pred = clf.fit(trainData,targetData)
    pickle.dump(clf, open( dataDir + '/process/' + trialName + '/boost' + trialName + '.p',"wb"))
    y_pred = clf.predict(trainData)
    print("Number of mislabeled points out of a total %d points : %d" % (trainData.shape[0],(targetData != y_pred).sum()))
    #svm = SVMClassifier(extractor) # try an svm, default is an RBF kernel function
    #trainPaths = []
    #classes = []
    
            
    # train the classifier on the data
    #svm.train(trainPaths,classes,verbose=False)
    #svm.save(dataDir + '/process/' + trialName + '/svm' + trialName + '.xml')
