import SimpleCV
from SimpleCV import Image,  VirtualCamera, Display, Color
import numpy as np
import Scientific.IO.NetCDF as DataSet
import cv2
import os

def main():
    dataDir = '/home/ctorney/data/fishPredation/'
    trialName = "MVI_3371"
    ncFileName = dataDir + "tracked/linked" + trialName + ".nc"    
    f = DataSet.NetCDFFile(ncFileName, 'a')


    trXY = f.variables['trXY']
    trackList = []
    trackList = np.empty_like (trXY)
    np.copyto(trackList,trXY)

    frNum = f.variables['frNum']
    trFrames = []
    trFrames = np.empty_like(frNum)
    np.copyto(trFrames,frNum)
    
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
    
    numBlocks =  np.size(mainTrackList,0)

    movieName = dataDir + "sampleVideo/" + trialName + ".avi";
    bkGrnd = Image(dataDir + "bk.png")
    mask  = Image(dataDir + "maskmono.png")
 
    vir = VirtualCamera(movieName, "video")
    display = Display()
    nb = 0
    
    thisBlockSize = mainTrackList[nb,TRACKLENGTH]
    thisTrackCount = mainTrackList[nb,TRACKCOUNT]
    frStart = mainTrackList[nb,FRMSTART]
    indexStart = mainTrackList[nb,INDSTART]
    for tr in range(thisTrackCount):
        direct = trialName + str(tr)
        if not os.path.exists(direct):
            os.makedirs(direct)
    liveTracks = trackIndex[timeIndex[:]==indexStart+1]
    counter = np.zeros_like(liveTracks)
    box_dim = 50    
    for fr in range(1):#range(thisBlockSize):
        thisIm = vir.getFrame(fr+frStart).toGray()
        thisIm = Image(cv2.absdiff(thisIm.getGrayNumpyCv2(), bkGrnd.getGrayNumpyCv2()), cv2image=True)
        thisIm = thisIm.applyBinaryMask(mask)
        
        for tr in range(thisTrackCount):
            direct = trialName + str(tr)
            print liveTracks[tr]
            xp = trackList[liveTracks[tr], fr+indexStart+1,0]
            yp = trackList[liveTracks[tr], fr+indexStart+1,1]
           
            tmpImg = thisIm.crop(round(xp), round(yp), box_dim,box_dim, centered=True)
            save_path = direct + "/img-" + str(counter[tr]) + ".png"
            tmpImg.save(save_path)
            counter[tr]+=1
            
        
        thisIm.save(display)
    
        

            
    display.quit()
    fid = f.variables['fid']    
    print np.shape(fid)
    #for p in fid: print p
    print trXY[0,0,0]
    fid.assignValue( (np.ones(np.shape(fid),'int16')))
    print fid[0,0]
    f.sync() 

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
