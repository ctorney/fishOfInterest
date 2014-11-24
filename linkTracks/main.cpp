
#include <iostream>
#include <netcdfcpp.h>
#include <math.h>
#include <cv.h>
#include <highgui.h>
#include <opencv2/video/tracking.hpp>

using namespace std;
using namespace cv;


// **************************************************************************************************
// structure for storing required information for a track
// **************************************************************************************************
struct track{
    int trackNo;
    int absentFrames;
    float x;
    float y;
    float xp;
    float yp;
    float vx;
    float vy;
    float ax;
    float ay;
    bool empty;
    int thisPos;
    bool duplicate;
    bool live;
    KalmanFilter* KF;
};

int setUpNetCDF(NcFile* dataFile, int nFrames);
int assignTracks(track* allTracks, float* dataXY,  int totalFish, int totalTracks, int f);



int main( int argc, char** argv )
{
    string dataDir = "/home/ctorney/data/fishPredation/";

    // **************************************************************************************************
    // open the positions netcdf file
    // **************************************************************************************************
    std::string trialName;
    if (argc > 1) 
        trialName =  argv[1];
    else
    {
        cout<<"trial name missing!"<<endl;
        return 0;
    }
    trialName =  "MVI_" + trialName;

    string ncFileName = dataDir + "tracked/linked" + trialName + ".nc";
    NcFile outputFile(ncFileName.c_str(), NcFile::Replace);
    outputFile.set_fill(NcFile::NoFill);

    if (!outputFile.is_valid())
    {
        cout << "Couldn't create netcdf file!\n";
        return -1;
    }

    // **************************************************************************************************
    // create the netcdf file for tracks
    // **************************************************************************************************
    ncFileName = dataDir + "tracked/" + trialName + ".nc";
    NcFile inputFile(ncFileName.c_str(), NcFile::ReadOnly);

    if (!inputFile.is_valid())
    {
        cout << "Couldn't open netcdf file!\n";
        return -1;
    }

    NcDim* frDim = inputFile.get_dim("frame");
    NcDim* iDim = inputFile.get_dim("fish");
    int totalFrames = frDim->size();
    int totalFish = iDim->size();
    
    setUpNetCDF(&outputFile, totalFrames);

    // **************************************************************************************************
    // get the variable that stores the positions and velocities
    // **************************************************************************************************
    NcVar* pxy = inputFile.get_var("pxy");
    NcVar* frNum = inputFile.get_var("frNum");

    NcVar* xy = outputFile.get_var("trXY");
    NcVar* v = outputFile.get_var("trVel");
    NcVar* a = outputFile.get_var("trAccel");
    NcVar* fid = outputFile.get_var("fid");
    NcVar* outFrame = outputFile.get_var("frNum");

    int totalTracks = 0;
    track *allTracks = new track[totalFish];

    // **************************************************************************************************
    // loop through the frames and link trajectories
    // **************************************************************************************************
    for (int f=0;f<totalFrames;f++)
    {
        float* dataXY = new float[totalFish*2];
        pxy->set_cur(f);
        pxy->get(dataXY, 1, totalFish, 2);
        int* curFrame = new int[1];
        frNum->set_cur(f);
        frNum->get(curFrame, 1);
        outFrame->set_cur(f);
        outFrame->put(curFrame, 1);
        
        // main call to track linking
        totalTracks = assignTracks(allTracks, dataXY, totalFish, totalTracks, f);

        // output to netcdf file
        for (int i=0;i<totalFish;i++)
            if (allTracks[i].live)
            {
                // set the current entry
                xy->set_cur(allTracks[i].trackNo, f, 0);
                v->set_cur(allTracks[i].trackNo, f, 0);
                a->set_cur(allTracks[i].trackNo, f, 0);
                // output position for the track id
                float* dataTRXY = new float[2];
                dataTRXY[0] = allTracks[i].x;
                dataTRXY[1] = allTracks[i].y;
                xy->put(dataTRXY, 1, 1, 2);
                // also record the smoothed velocity and acceleration from the Kalman filter
                dataTRXY[0] = allTracks[i].vx;
                dataTRXY[1] = allTracks[i].vy;
                v->put(dataTRXY, 1, 1, 2);
                dataTRXY[0] = allTracks[i].ax;
                dataTRXY[1] = allTracks[i].ay;
                a->put(dataTRXY, 1, 1, 2);
            }
    }
    return 0;
}

int setUpNetCDF(NcFile* dataFile, int nFrames)
{
    // dimension for each frame
    NcDim* frDim = dataFile->add_dim("frame", nFrames);
    // xy dimension for vectors
    NcDim* xyDim = dataFile->add_dim("xy", 2);
    // dimension for tracks (unlimited as new tracks are created when a fish is lost)
    NcDim* trDim = dataFile->add_dim("track");
    // define a netCDF variable for the positions of linked tracks
    dataFile->add_var("trXY", ncFloat, trDim, frDim, xyDim);
    // velocities
    dataFile->add_var("trVel", ncFloat, trDim, frDim, xyDim);
    // accelerations 
    dataFile->add_var("trAccel", ncFloat, trDim, frDim, xyDim);
    // variable for ID of fish
    dataFile->add_var("fid", ncShort, trDim, frDim);
    // variable for certainty of ID assignment
    dataFile->add_var("certID", ncFloat, trDim, frDim);
    // variable for the frame number
    dataFile->add_var("frNum", ncInt, frDim);

    return 0;

}


int assignTracks(track* allTracks, float* dataXY,  int totalFish, int totalTracks, int f)
{
    int trackCount = totalTracks;
    

    // **************************************************************************************************
    // set initial values for positions
    // **************************************************************************************************
    int matchedTrack[totalFish];
    bool doubleMatch[totalFish];
    for (int i=0;i<totalFish;i++)
    {
        matchedTrack[i]=-1;
        doubleMatch[i]=false;
    }

    // **************************************************************************************************
    // nearest neighbour assignment for live tracks based on prediction from last time step
    // **************************************************************************************************
    for (int i=0;i<totalFish;i++)
    {
        float minDist = 1e12;
        if (!allTracks[i].live)
            continue;
        allTracks[i].empty = true;
        for (int j=0;j<totalFish;j++)
            if (dataXY[j*2+0]>0) 
            {
                float dist = pow(allTracks[i].xp-dataXY[j*2+0],2) + pow(allTracks[i].yp-dataXY[j*2+1],2);
                if (dist<minDist)
                {
                    minDist = dist;
                    allTracks[i].thisPos = j;
                    allTracks[i].empty = false;
                }


            }


        if (!allTracks[i].empty)
        {
            if (matchedTrack[allTracks[i].thisPos]<0)
                matchedTrack[allTracks[i].thisPos] = i;
            else
                doubleMatch[allTracks[i].thisPos] = true;
        }


    }

    // **************************************************************************************************
    // loop through the frames and link trajectories
    // **************************************************************************************************
    for (int i=0;i<totalFish;i++)
    {
        if (!allTracks[i].live)
            continue;
        if (!allTracks[i].empty)
        {
            if (doubleMatch[allTracks[i].thisPos])
                allTracks[i].duplicate=true;
        }


    }


    // **************************************************************************************************
    // loop through the frames and link trajectories
    // **************************************************************************************************
    // now delete all empty or duplicate tracks
    for (int i=0;i<totalFish;i++)
    {
        if (!allTracks[i].live)
            continue;
        if ((allTracks[i].empty)||(allTracks[i].duplicate))
        {
            allTracks[i].live = false;
            // delete the associated Kalman filter
            delete allTracks[i].KF;
        }


    }

    // **************************************************************************************************
    // create entries for unassigned positions
    // **************************************************************************************************
    for (int i=0;i<totalFish;i++)
        if ((dataXY[i*2]>0) && (matchedTrack[i]<0))
        {
            for (int j=0;j<totalFish;j++)
            {
                if (allTracks[j].live)
                    continue;

                allTracks[j].thisPos=i;
                allTracks[j].trackNo=trackCount;
                allTracks[j].x=dataXY[i*2+0];
                allTracks[j].y=dataXY[i*2+1];
                allTracks[j].vx = 0.0f;
                allTracks[j].vy = 0.0f;
                allTracks[j].ax = 0.0f;
                allTracks[j].ay = 0.0f;
                allTracks[j].duplicate = false;
                allTracks[j].live = true;

                // initialize the Kalman filter
                allTracks[j].KF = new KalmanFilter(6, 2, 0);
                // assume dt=1 for simplicity
                allTracks[j].KF->transitionMatrix = *(Mat_<float>(6, 6) << 1,0,1,0,0.5,0, 0,1,0,1,0,0.5, 0,0,1,0,1,0, 0,0,0,1,0,1, 0,0,0,0,1,0, 0,0,0,0,0,1);

                allTracks[j].KF->statePre.at<float>(0) = allTracks[j].x;
                allTracks[j].KF->statePre.at<float>(1) = allTracks[j].y;
                allTracks[j].KF->statePre.at<float>(2) = 0;
                allTracks[j].KF->statePre.at<float>(3) = 0;
                allTracks[j].KF->statePre.at<float>(4) = 0;
                allTracks[j].KF->statePre.at<float>(5) = 0;
                allTracks[j].KF->statePost.at<float>(0) = allTracks[j].x;
                allTracks[j].KF->statePost.at<float>(1) = allTracks[j].y;
                allTracks[j].KF->statePost.at<float>(2) = 0;
                allTracks[j].KF->statePost.at<float>(3) = 0;
                allTracks[j].KF->statePost.at<float>(4) = 0;
                allTracks[j].KF->statePost.at<float>(5) = 0;
                setIdentity( allTracks[j].KF->measurementMatrix);
                setIdentity( allTracks[j].KF->processNoiseCov, Scalar::all(1e-4));
                setIdentity( allTracks[j].KF->measurementNoiseCov, Scalar::all(1e-1));
                setIdentity( allTracks[j].KF->errorCovPost, Scalar::all(.1));
                Mat prediction = allTracks[j].KF->predict();

                // keep a record of how many tracks have been created
                trackCount++;


                break;
            }
        }

    // **************************************************************************************************
    // finally make a prediction using the Kalman filter and add the current measurement
    // **************************************************************************************************
    for (int i=0;i<totalFish;i++)
    {
        if (!allTracks[i].live)
            continue;
        int j = allTracks[i].thisPos;
        allTracks[i].x=dataXY[j*2+0];
        allTracks[i].y=dataXY[j*2+1];
        // use the filters estimate of velocity and acceleration
        allTracks[i].vx = allTracks[i].KF->statePre.at<float>(2);
        allTracks[i].vy = allTracks[i].KF->statePre.at<float>(3);
        allTracks[i].ax = allTracks[i].KF->statePre.at<float>(4);
        allTracks[i].ay = allTracks[i].KF->statePre.at<float>(5);

        Mat_<float> measurement(2,1);
        measurement(0) = allTracks[i].x;
        measurement(1) = allTracks[i].y;

        allTracks[i].KF->correct(measurement);
        Mat prediction = allTracks[i].KF->predict();
        allTracks[i].xp=prediction.at<float>(0);
        allTracks[i].yp=prediction.at<float>(1);



    }
    return trackCount;
}

