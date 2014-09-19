

import SimpleCV
import time
from SimpleCV import Image, VirtualCamera, Display, Features, TreeClassifier, ImageSet, Color
from SimpleCV import SVMClassifier, TreeClassifier
import sys
import random


from circularHOGExtractor import circularHOGExtractor

ch = circularHOGExtractor(5,5,4) 
extractor = [ch] # put these all together
svm = SVMClassifier(extractor) # try an svm, default is an RBF kernel function
tree = TreeClassifier(extractor,flavor='Boosted') # also try a decision tree
tree.mBoostedFlavorDict['NTrees'] = 10
#tree.mforestFlavorDict['NTrees'] = 200
trainPaths = ['./MVI_33710/','./MVI_33711/','./MVI_33712/','./MVI_33713/']
# # define the names of our classes
classes = ['0','1','2','3']
# # # train the data
print svm.train(trainPaths,classes,verbose=True)
print tree.train(trainPaths,classes,verbose=True)
svm.save('trainedSVM.xml')
tree.save('trainedTREE.xml')
#
#    outTest = False
#
#    if outTest:
#        cl = SVMClassifier.load('trainedSVM.xml')
#         #   tree2.mClassifier = extractor
#            testPaths = ['./trainyes/','./trainno/']
#                test = ImageSet()
#        for p in testPaths: # load the data
#                test += ImageSet(p)
#        random.shuffle(test) # shuffle it
#            test = test[0:10] # pick ten off the top
#                i = 0
#                    for t in test:
#                            finalClass = 'no'
#                                    #if className == 'yes':
#                                    finalClass = cl.classify(t) # classify them
#                                            t.drawText(finalClass,10,10,fontsize=10,color=Color.RED)
#            fname = "./timgs/classification"+str(i)+".png"
#                    t.applyLayers().resize(w=128).save(fname)
#            i = i + 1
#
#
