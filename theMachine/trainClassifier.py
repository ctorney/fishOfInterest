
from SimpleCV import SVMClassifier
from circularHOGExtractor import circularHOGExtractor

def trainClassifier(trialName, NUMFISH):

    ch = circularHOGExtractor(5,5,6) 

    extractor = [ch] 
    svm = SVMClassifier(extractor) # try an svm, default is an RBF kernel function
    trainPaths = []
    classes = []
    for tr in range(NUMFISH):
        directory = './' + trialName + str(tr) + '/'
        trainPaths.append(directory)
        classes.append(str(tr))
            
    # train the classifier on the data
    svm.train(trainPaths,classes,verbose=False)
    svm.save('svm' + trialName + '.xml')
