import SimpleCV
from SimpleCV import Image,  VirtualCamera, Display, Color
import numpy as np
import Scientific.IO.NetCDF as DataSet
import cv2
import os
from SimpleCV import SVMClassifier, TreeClassifier


from circularHOGExtractor import circularHOGExtractor

def main():
    dataDir = '/home/ctorney/data/fishPredation/'
    trialName = "MVI_3371"
    ncFileName = dataDir + "tracked/linked" + trialName + ".nc"    
    f = DataSet.NetCDFFile(ncFileName, 'r')

    cl = SVMClassifier.load('trainedSVM.xml')
    #cl = TreeClassifier.load('trainedTREE.xml')
    trXY = f.variables['trXY']
    trackList = []
    trackList = np.empty_like (trXY)
    np.copyto(trackList,trXY)

    frNum = f.variables['frNum']
    trFrames = []
    trFrames = np.empty_like(frNum)
    np.copyto(trFrames,frNum)
    
    fid = f.variables['fid']  
    fishIDs = -np.ones(np.shape(fid),'int16')
    
    print np.shape(fid)
    #for p in fid: print p
    print np.shape(trXY)
#    fid.assignValue( (np.ones(np.shape(fid),'int16')))
 #   print fid[0,0]
 #   f.sync() 

    fishIDs[237,:]=0 
    fishIDs[238,:]=1
    fishIDs[243,:]=2
    fishIDs[244,:]=3

    [trackIndex,timeIndex]=np.nonzero(trackList[:,:,0])
    
    maxTime = np.size(trackList,1)

    trackSum = np.zeros([maxTime])
    trackCount = np.zeros([maxTime])
    for k in range(maxTime): 
        trackSum[k]=sum(trackIndex[timeIndex[:]==k])
        trackCount[k]=np.size(trackIndex[timeIndex[:]==k])

    print np.shape(trackSum)

    d= np.diff(trackSum)

    d[0]=1
    d[-1]=1
    idx, = d.nonzero()    
    conCounts =  idx[1:-1] - idx[0:-1-1]
   
    # array of all tracks || length of time tracks are present || number of tracks || start index || stop index || start frame of movie || stop frame of movie
    mainTrackList = np.column_stack((trackCount[idx[0:-1-1]+1], conCounts, idx[0:-1-1],idx[1:-1], trFrames[idx[0:-1-1]], trFrames[idx[1:-1]]))
    
    ind = np.lexsort((-mainTrackList[:,1],-mainTrackList[:,0]))
    mainTrackList = mainTrackList[ind,:].astype(int)

    TRACKCOUNT = 0
    TRACKLENGTH = 1
    INDSTART = 2
    INDSTOP = 3
    FRMSTART = 4
    FRMSTOP = 5
    classes =np.array(['0','1','2','3'])
    numBlocks =  2#np.size(mainTrackList,0)

    movieName = dataDir + "sampleVideo/" + trialName + ".avi";
    bkGrnd = Image(dataDir + "bk.png")
    mask  = Image(dataDir + "maskmono.png")
 
    vir = VirtualCamera(movieName, "video")
    #display = Display()
    for nb in range(1,numBlocks):
    
        thisBlockSize = mainTrackList[nb,TRACKLENGTH]
        thisTrackCount = mainTrackList[nb,TRACKCOUNT]
        frStart = mainTrackList[nb,FRMSTART]
        indexStart = mainTrackList[nb,INDSTART]
        
        score = np.zeros((thisTrackCount,4))
        liveTracks = trackIndex[timeIndex[:]==indexStart+1]
        counter = np.zeros_like(liveTracks)
        box_dim = 50    
        for fr in range(thisBlockSize):
            thisIm = vir.getFrame(fr+frStart).toGray()
            thisIm = Image(cv2.absdiff(thisIm.getGrayNumpyCv2(), bkGrnd.getGrayNumpyCv2()), cv2image=True)
            thisIm = thisIm.applyBinaryMask(mask)
            
            for tr in range(thisTrackCount):
                
                if fishIDs[[liveTracks[tr],fr+indexStart+1]<0:
                    xp = trackList[liveTracks[tr], fr+indexStart+1,0]
                    yp = trackList[liveTracks[tr], fr+indexStart+1,1]
                
                    tmpImg = thisIm.crop(round(xp), round(yp), box_dim,box_dim, centered=True)
                    fishGuess = cl.classify(tmpImg)
                    int1, =  np.where(classes==fishGuess)
                    score[tr,int1]+=1
                else:
                    score[tr,fishIDs[[liveTracks[tr],fr+indexStart+1]]+=1
        
        assignment = np.argmax(score,axis=1)
        
        for avals in range(np.size(assignment)): 
            if np.sum(assignment==assignment[avals])>1: 
                assignment[assignment==assignment[avals]]=-1



        for tr in range(thisTrackCount):
            fishIDs[[liveTracks[tr],:] = assignment[tr]

        print score
    
        

    display.quit()
    
    return
    
    movieName = dataDir + "sampleVideo/" + trialName + ".avi";
    print movieName
 
    vir = VirtualCamera(movieName, "video")
    display = Display()
    while display.isNotDone():

        thisIm = vir.getImage()

        thisIm.save(display)
        
        

        if vir.getImage().getBitmap() == '': display.done = True
        if display.mouseRight: display.done = True

    display.quit()




if __name__ == "__main__":
   main()
