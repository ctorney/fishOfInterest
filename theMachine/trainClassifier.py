
from SimpleCV import SVMClassifier
from circularHOGExtractor import circularHOGExtractor

def trainClassifier(dataDir, trialName, NUMFISH):


    
    ch = circularHOGExtractor(5,5,6) 

    extractor = [ch] 
    svm = SVMClassifier(extractor) # try an svm, default is an RBF kernel function
    trainPaths = []
    classes = []
    for tr in range(NUMFISH):
        directory = dataDir + '/process/' + trialName + '/FR_ID' + str(tr) + '/' 
        trainPaths.append(directory)
        classes.append(str(tr))
            
    # train the classifier on the data
    svm.train(trainPaths,classes,verbose=False)
    svm.save(dataDir + '/process/' + trialName + '/svm' + trialName + '.xml')
