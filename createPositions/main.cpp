
#include <cv.h>
#include <highgui.h>
#include <iostream>

using namespace cv;
using namespace std;


int main( int argc, char** argv )
{
    string dataDir = "/home/ctorney/data/fishPredation/";

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
    
    string movie = dataDir + "sampleVideo/MVI_3371.avi";
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
