
#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <netcdfcpp.h>

using namespace cv;
using namespace std;

int setUpNetCDF(NcFile* dataFile, int nFrames, int nFish);

int main( int argc, char** argv )
{
    // **************************************************************************************************
    // set up the parameters for blob detection
    // **************************************************************************************************
    SimpleBlobDetector::Params params;
    params.minDistBetweenBlobs = 1.0f;
    params.filterByInertia = false;
    params.filterByConvexity = false;
    params.filterByColor = false;
    params.filterByCircularity = false;
    params.filterByArea = true;
    params.minArea = 5.0f;
    params.maxArea = 200.0f;
    params.minThreshold = 15;
    params.maxThreshold = 255;

    // set up and create the detector using the parameters
    Ptr<FeatureDetector> blob_detector = new SimpleBlobDetector(params);
    blob_detector->create("SimpleBlob");

    vector<KeyPoint> keypoints;
    string dataDir = "/home/ctorney/data/fishPredation/";
    string bkup = dataDir + "bk-up.png";
    string bkdown = dataDir + "bk-down.png";

    // **************************************************************************************************
    // setup background images
    // **************************************************************************************************
    Mat imBkUp = imread( bkup , IMREAD_GRAYSCALE);
    Mat imBkDown = imread( bkdown, IMREAD_GRAYSCALE );

    int rows = imBkUp.rows;
    int cols = imBkUp.cols;
    int split = 0.5*rows;

    Mat imBk;

    vconcat(imBkUp(Range(0,split), Range(0, cols)), imBkDown(Range(split,rows),Range(0,cols)), imBk);
    
//    imwrite( "bk.png", imBk );


    string mask = dataDir + "mask.png";

    // **************************************************************************************************
    // create mask image
    // **************************************************************************************************
    Mat imMask = imread( mask, IMREAD_GRAYSCALE );

    // **************************************************************************************************
    // open the movie
    // **************************************************************************************************
    string trialName = "MVI_3371";

    string movie = dataDir + "sampleVideo/" + trialName + ".avi";
    VideoCapture cap(movie);
    if (!cap.isOpened())
    {
        cout << "Failed to open avi file: " << movie << endl;
        return -1;
    }

    int fCount = cap.get(CV_CAP_PROP_FRAME_COUNT );
    int fStart = 200;
    int nFrames = fCount - fStart;
    int nFish = 4;

    // **************************************************************************************************
    // create the netcdf file
    // **************************************************************************************************
    string ncFileName = dataDir + "tracked/" + trialName + ".nc";
    NcFile dataFile(ncFileName.c_str(), NcFile::Replace);

    if (!dataFile.is_valid())
    {
        cout << "Couldn't open netcdf file!\n";
        return -1;
    }

    setUpNetCDF(&dataFile, nFrames, nFish );

    // get the variable to store the positions
    NcVar* pxy = dataFile.get_var("pxy");
    NcVar* frNum = dataFile.get_var("frNum");

    // **************************************************************************************************
    // loop over all frames and record positions
    // **************************************************************************************************

    bool visuals = false;
    Mat frame, gsFrame;
    for(int f=0;f<fCount;f++)
    {
        if (!cap.read(frame))             
            break;
        if (f<fStart)
            continue;

        // convert to grayscale
        cvtColor( frame, gsFrame, CV_BGR2GRAY );
        // subtract background
        absdiff(imBk,gsFrame, gsFrame);
        // select region of interest using mask
        bitwise_and(gsFrame, imMask, gsFrame);
        // find the blobs
        blob_detector->detect(gsFrame, keypoints);
        // create array for output
        float dataOut[nFish][2];
        for (int i=0;i<nFish;i++)
            dataOut[i][0]=dataOut[i][1]=-1.0f;

        // extract the x y coordinates of the keypoints
        // use only the first 4 as keypoints are sorted according to quality 
        int foundPoints = keypoints.size();
        if (foundPoints>nFish)
            foundPoints = nFish;
        for (int i=0; i<foundPoints; i++)
        {
            dataOut[i][0] = keypoints[i].pt.x; 
            dataOut[i][1] = keypoints[i].pt.y;
            if (visuals)
                circle( frame,keypoints[i].pt, 8, Scalar( 25, 125, 125 ), -1, 8);
        }

        pxy->set_cur(f - fStart);
        frNum->set_cur(f - fStart);
        pxy->put(&dataOut[0][0], 1, nFish, 2);
        frNum->put(&f, 1);


        if (visuals)
        {
            pyrDown(frame, frame) ;
            imshow("detected individuals", frame);


            char key = cvWaitKey(10);
            if (key == 27) // ESC
                break;
        }
    }

    return 0;
}

int setUpNetCDF(NcFile* dataFile, int nFrames, int nFish)
{
    // dimension for each frame
    NcDim* frDim = dataFile->add_dim("frame", nFrames);
    // dimension for individual fish
    NcDim* iDim = dataFile->add_dim("fish", nFish);
    // xy dimension for vectors
    NcDim* xyDim = dataFile->add_dim("xy", 2);

    // define a netCDF variable for the positions of individuals
    dataFile->add_var("pxy", ncFloat, frDim, iDim, xyDim);
    // define a netCDF variable for the frame number to account 
    // for any offset needed to remove experimental set-up
    dataFile->add_var("frNum", ncInt, frDim);

    return 0;

}



