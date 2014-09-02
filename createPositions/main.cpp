
#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <netcdfcpp.h>

using namespace cv;
using namespace std;

int setUpNetCDF(NcFile* dataFile, int nFrames, int nFish);

int main( int argc, char** argv )
{
    string dataDir = "/home/ctorney/data/fishPredation/";
    string trialName = "MVI_3371";
    int nFrames = 100;
    int nFish = 4;

    string ncFileName = dataDir + "tracked/" + trialName + ".nc";
    NcFile dataFile(ncFileName.c_str(), NcFile::Replace);

    if (!dataFile.is_valid())
    {
        cout << "Couldn't open netcdf file!\n";
        return -1;
    }

    setUpNetCDF(&dataFile, nFrames, nFish );
    NcVar* pxy = dataFile.get_var("pxy");

    // This is the data array we will write. It will just be filled
    // with a progression of numbers for this example.
    float dataOut[nFish][2];

    for (int t=0;t<20;t++)
    {
    // Create some pretend data. If this wasn't an example program, we
    // would have some real data to write, for example, model output.
    for(int i = 0; i < nFish; i++)
    {
            dataOut[i][0] = i + 0.1;
            dataOut[i][1] = t + 0.125;
    }

    // Write the pretend data to the file. Although netCDF supports
    // reading and writing subsets of data, in this case we write all
    // the data in one operation.
    pxy->set_cur(t);
    pxy->put(&dataOut[0][0], 1, nFish, 2);
    }
    // The file will be automatically close when the NcFile object goes
    // out of scope. This frees up any internal netCDF resources
    // associated with the file, and flushes any buffers.



    string bkup = dataDir + "bk-up.png";
    string bkdown = dataDir + "bk-down.png";

    /// read background images
    Mat imBkUp = imread( bkup , IMREAD_GRAYSCALE);
    Mat imBkDown = imread( bkdown, IMREAD_GRAYSCALE );

    int rows = imBkUp.rows;
    int cols = imBkUp.cols;
    int split = 0.5*rows;

    Mat imBk;

    vconcat(imBkUp(Range(0,split), Range(0, cols)), imBkDown(Range(split,rows),Range(0,cols)), imBk);

    string mask = dataDir + "mask.png";

    /// read mask image
    Mat imMask = imread( mask, IMREAD_GRAYSCALE );


    // set up the parameters for blob detection
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
    string movie = dataDir + "sampleVideo/" + trialName + ".avi";
    VideoCapture cap(movie);
    if (!cap.isOpened())
    {
        std::cout << "!!! Failed to open file: " << movie << std::endl;
        return -1;
    }

    int fCount = cap.get(CV_CAP_PROP_FRAME_COUNT );
    int fStart = 200;
    Mat frame, gsFrame;
    for(int f=0;f<fCount;f++)
    {
        if (!cap.read(frame))             
            break;
        if (f<fStart)
            continue;

        cvtColor( frame, gsFrame, CV_BGR2GRAY );
        absdiff(imBk,gsFrame, gsFrame);


        bitwise_and(gsFrame, imMask, gsFrame);
        blob_detector->detect(gsFrame, keypoints);

        // extract the x y coordinates of the keypoints: 

        for (int i=0; i<keypoints.size(); i++){
            float X=keypoints[i].pt.x; 
            float Y=keypoints[i].pt.y;
            circle( frame,keypoints[i].pt, 8, Scalar( 25, 125, 125 ), -1, 8);
        }

        pyrDown(frame, frame) ;
        imshow("detected individuals", frame);

        char key = cvWaitKey(10);
        if (key == 27) // ESC
            break;
    }

    return 0;
}

int setUpNetCDF(NcFile* dataFile, int nFrames, int nFish)
{
 //   dataFile->set_fill(NcFile::NoFill);
//    return 0;
    // dimension for each frame
    NcDim* frDim = dataFile->add_dim("frame", nFrames);
    // dimension for individual fish
    NcDim* iDim = dataFile->add_dim("fish", nFish);
    // xy dimension for vectors
    NcDim* xyDim = dataFile->add_dim("xy", 2);
    // dimension for tracks (unlimited as new tracks are created when a fish is lost)
    NcDim* trDim = dataFile->add_dim("track");

    // define a netCDF variable for the positions of individuals
    dataFile->add_var("pxy", ncFloat, frDim, iDim, xyDim);
    // define a netCDF variable for the positions of linked tracks
    dataFile->add_var("trxy", ncFloat, trDim, frDim, xyDim);
    // linked tracks following smoothing
    dataFile->add_var("trxy_sm", ncFloat, trDim, frDim, xyDim);
    // velocities
    dataFile->add_var("tr_vel", ncFloat, trDim, frDim, xyDim);
    // accelerations 
    dataFile->add_var("tr_accel", ncFloat, trDim, frDim, xyDim);
    // variable for ID of fish
    dataFile->add_var("fid", ncShort, trDim, frDim);

    return 0;

}



