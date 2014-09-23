import SimpleCV
from SimpleCV import Image,  VirtualCamera, Display, Color
import numpy as np
import Scientific.IO.NetCDF as Dataset

import cv2
import os
from SimpleCV import SVMClassifier, TreeClassifier


from circularHOGExtractor import circularHOGExtractor

#def main():
dataDir = '/home/ctorney/data/fishPredation/'
trialName = "MVI_3371"
ncFileName = dataDir + "tracked/linked" + trialName + ".nc"    
f = Dataset.NetCDFFile(ncFileName, 'a')

NUMFISH = 4

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

fishIDs =np.empty_like (fid)
np.copyto(fishIDs,fid)
print np.shape(fid)
#for p in fid: print p
print np.shape(trXY)
#    fid.assignValue( (np.ones(np.shape(fid),'int16')))
 #   print fid[0,0]
 #   f.sync() 


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
# because diff calculates f(n)-f(n+1) we need the next entry found for all except d[0]
idx[idx>0]=idx[idx>0]+1

conCounts =  idx[1:-1] - idx[0:-1-1]
   
# array of all tracks || length of time tracks are present || number of tracks || start index || stop index || start frame of movie || stop frame of movie
mainTrackList = np.column_stack((trackCount[idx[0:-1-1]], conCounts, idx[0:-1-1],idx[1:-1], trFrames[idx[0:-1-1]], trFrames[idx[1:-1]]))

ind = np.lexsort((-mainTrackList[:,1],-mainTrackList[:,0]))
mainTrackList = mainTrackList[ind,:].astype(int)

TRACKCOUNT = 0
TRACKLENGTH = 1
INDSTART = 2
INDSTOP = 3
FRMSTART = 4
FRMSTOP = 5
classes =np.array(['0','1','2','3'])
tmpDirs =np.array([trialName + 'TMP-0', trialName + 'TMP-1', trialName + 'TMP-2', trialName + 'TMP-3'])
# only do the complete segments
numBlocks = np.sum(mainTrackList[:,TRACKCOUNT]==NUMFISH)# np.size(mainTrackList,0)
#numBlocks = 10
movieName = dataDir + "sampleVideo/" + trialName + ".avi";
bkGrnd = Image(dataDir + "bk.png")
mask  = Image(dataDir + "maskmono.png")
 
vir = VirtualCamera(movieName, "video")
#display = Display()
maxTrackCount = mainTrackList[0,TRACKCOUNT]
for nf in range(NUMFISH):
    if not os.path.exists(tmpDirs[nf]):
        os.makedirs(tmpDirs[nf])

for nb in range(1, numBlocks):
    print nb, numBlocks
    for nf in range(NUMFISH):
        #os.remove(tmpDirs[nf] + "/*")
        fileList = os.listdir(tmpDirs[nf])
        for fileName in fileList:
            os.remove(tmpDirs[nf]+"/"+fileName)
    thisBlockSize = mainTrackList[nb,TRACKLENGTH]
    thisTrackCount = mainTrackList[nb,TRACKCOUNT]
    frStart = mainTrackList[nb,FRMSTART]
    indexStart = mainTrackList[nb,INDSTART]
    indexStop = mainTrackList[nb,INDSTOP]
    
    score = np.zeros((thisTrackCount,4))
    liveTracks = trackIndex[timeIndex[:]==indexStart]
    counter = np.zeros_like(liveTracks)
    box_dim = 50    

    liveTracks = trackIndex[timeIndex[:]==indexStart]
     

    counter = np.zeros_like(liveTracks)
    box_dim = 50    
    for fr in range(indexStart, indexStop):
        thisIm = vir.getFrame(trFrames[fr]).toGray()
        thisIm = Image(cv2.absdiff(thisIm.getGrayNumpyCv2(), bkGrnd.getGrayNumpyCv2()), cv2image=True)
        thisIm = thisIm.applyBinaryMask(mask)
    
        for tr in range(thisTrackCount):
        
      
            if fishIDs[liveTracks[tr],fr]<0:        
                xp = trackList[liveTracks[tr], fr, 0]
                yp = trackList[liveTracks[tr], fr, 1]
                if xp>0:        
                    tmpImg = thisIm.crop(round(xp), round(yp), box_dim,box_dim, centered=True)
                    fishGuess = cl.classify(tmpImg)
                    int1, =  np.where(classes==fishGuess)
                    #int1 = (tr+nb)%NUMFISH
                    score[tr,int1]+=1
                    save_path = tmpDirs[tr] + "/img-" + str(fr) + "-" + str(nb) + ".png"
                    tmpImg.save(save_path)
            else:
                score[tr,fishIDs[liveTracks[tr],fr]]+=1
    
    assignment = np.argmax(score,axis=1)
    
    for avals in range(np.size(assignment)): 
        if np.sum(assignment==assignment[avals])>1: 
            assignment[assignment==assignment[avals]]=-1

    if np.all( assignment>=0):
        # all good so copy to learning folders
        for tr in range(thisTrackCount):
            sourceDir = tmpDirs[tr]
            destDir =trialName + str( assignment[tr])
            os.system("mv " + sourceDir + "/* " + destDir        )
            
        print fishIDs[trackList[:,1500,0]>0,1500]
    
        print assignment
        print score
        print liveTracks
        print indexStart, indexStop
        for tr in range(thisTrackCount):
            if fishIDs[liveTracks[tr],0]<0:
                fishIDs[liveTracks[tr],:] = assignment[tr]

#    print score
fid.assignValue( (fishIDs))
f.sync()
f.close()
#fid.assignValue( (fishIDs))

#fid.sync()         

#    display.quit()

#    return



if __name__ == "__main__":
   main()
