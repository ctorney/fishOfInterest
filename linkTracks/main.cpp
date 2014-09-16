
#include <iostream>
#include <netcdfcpp.h>
#include <math.h>
#include <cv.h>
#include <highgui.h>
#include <opencv2/video/tracking.hpp>

using namespace std;
using namespace cv;


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
    // open the netcdf file
    // **************************************************************************************************
    string trialName = "MVI_3371";

    string ncFileName = dataDir + "tracked/linked" + trialName + ".nc";
    NcFile outputFile(ncFileName.c_str(), NcFile::Replace);
    outputFile.set_fill(NcFile::NoFill);

    if (!outputFile.is_valid())
    {
        cout << "Couldn't create netcdf file!\n";
        return -1;
    }

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
    //totalFrames=10;
    setUpNetCDF(&outputFile, totalFrames);


    // get the variable that stores the positions
    NcVar* pxy = inputFile.get_var("pxy");

    NcVar* xy = outputFile.get_var("trXY");
    NcVar* v = outputFile.get_var("trVel");
    NcVar* a = outputFile.get_var("trAccel");
    NcVar* fid = outputFile.get_var("fid");

    int totalTracks = 0;
    track *allTracks = new track[totalFish];
totalFrames=1;
    for (int f=0;f<totalFrames;f++)
    {
        cout<<f<<endl;
        float* dataXY = new float[totalFish*2];
        pxy->set_cur(f);
        pxy->get(dataXY, 1, totalFish, 2);
        
     //   cout<<f<<endl;
     //   for (int i=0;i<totalFish;i++)
     //       cout<<dataXY[i*2]<<" "<<dataXY[i*2+1]<<endl;
        totalTracks = assignTracks(allTracks, dataXY, totalFish, totalTracks, f);
        for (int i=0;i<totalFish;i++)
            if (allTracks[i].live)
            {
                xy->set_cur(allTracks[i].trackNo, f, 0);
                float* dataTRXY = new float[2];
                dataTRXY[0] = allTracks[i].x;
                dataTRXY[1] = allTracks[i].y;
     //           cout<<i<<" "<<allTracks[i].trackNo<<" "<<dataTRXY[0]<<endl;
                xy->put(dataTRXY, 1, 1, 2);
            }
        //        cout<<allTracks[i].trackNo<<" "<<allTracks[i].x<<" "<<allTracks[i].y<<endl;
      //  cout<<"~~~"<<endl;
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

    return 0;

}


int assignTracks(track* allTracks, float* dataXY,  int totalFish, int totalTracks, int f)
{
    int trackCount = totalTracks;
    int liveTracks = 0;
    int pos=0;
    
    // count how many live tracks
    // count how many positions (positions that are negative are empty)
    for (int i=0;i<totalFish;i++)
    {
        if (dataXY[i*2+0]>0) 
            pos++;
        if (allTracks[i].live)
            liveTracks++;

    }

    int matchedTrack[totalFish];
    bool doubleMatch[totalFish];
    for (int i=0;i<totalFish;i++)
    {
        matchedTrack[i]=-1;
        doubleMatch[i]=false;
    }

    // nearest neighbour assignment for live tracks
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
 //       if (f==437)
 //           cout<<i<<" "<<allTracks[i].thisPos<<" "<<minDist<<endl;
        if (!allTracks[i].empty)
        {
            if (matchedTrack[allTracks[i].thisPos]<0)
                matchedTrack[allTracks[i].thisPos] = i;
            else
                doubleMatch[allTracks[i].thisPos] = true;
        }


    }

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


   /* if (f==437)
        for (int i=0;i<totalFish;i++)
        {
            cout<<"DEBUG: "<<i<<endl;
            cout<<"DEBUG live: "<<allTracks[i].live<<endl;
            cout<<"DEBUG emp: "<<allTracks[i].empty<<endl;
            cout<<"DEBUG pos: "<<allTracks[i].thisPos<<endl;
            cout<<"DEBUG dup: "<<allTracks[i].duplicate<<endl;
            cout<<"DEBUG track: "<<allTracks[i].trackNo<<endl;
            cout<<"DEBUG xp: "<<allTracks[i].xp<<endl;
            cout<<"DEBUG yp: "<<allTracks[i].yp<<endl;
            cout<<"~~~~~~~~~~~~"<<endl;


        }

*/
    // now delete all empty or duplicate tracks
    for (int i=0;i<totalFish;i++)
    {
        if (!allTracks[i].live)
            continue;
        if ((allTracks[i].empty)||(allTracks[i].duplicate))
        {
            allTracks[i].live = false;
        cout<<"in"<<endl;
            delete allTracks[i].KF;
        cout<<"out"<<endl;
        }


    }
  /*  if (f==437)
        for (int i=0;i<totalFish;i++)
        {
            cout<<"DEBUG: "<<i<<endl;
            cout<<"DEBUG live: "<<allTracks[i].live<<endl;
            cout<<"DEBUG emp: "<<allTracks[i].empty<<endl;
            cout<<"DEBUG pos: "<<allTracks[i].thisPos<<endl;
            cout<<"DEBUG dup: "<<allTracks[i].duplicate<<endl;
            cout<<"DEBUG track: "<<allTracks[i].trackNo<<endl;
            cout<<"~~~~~~~~~~~~"<<endl;


        }
*/

    // create entries for unassigned positions
    for (int i=0;i<totalFish;i++)
        if ((dataXY[i*2]>0) && (matchedTrack[i]<0))
        {
            for (int j=0;j<totalFish;j++)
            {
                if (allTracks[j].live)
                    continue;


 //               cout<<i<<" "<<j<<endl;

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
                allTracks[j].KF = new KalmanFilter(4, 2, 0);
                allTracks[j].KF->transitionMatrix = *(Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
                 
                // init...
                 allTracks[j].KF->statePre.at<float>(0) = allTracks[j].x;
                 allTracks[j].KF->statePre.at<float>(1) = allTracks[j].y;
                 allTracks[j].KF->statePre.at<float>(2) = 0;
                 allTracks[j].KF->statePre.at<float>(3) = 0;
                setIdentity( allTracks[j].KF->measurementMatrix);
                setIdentity( allTracks[j].KF->processNoiseCov, Scalar::all(1e-4));
                setIdentity( allTracks[j].KF->measurementNoiseCov, Scalar::all(1e-1));
                setIdentity( allTracks[j].KF->errorCovPost, Scalar::all(.1));
        cout<<"unpredicted"<<endl;
               cout<<allTracks[j].KF->statePre.at<float>(0)<<" ";
               cout<<  allTracks[j].KF->statePre.at<float>(1) <<" ";
                cout<<         allTracks[j].KF->statePre.at<float>(2) <<" ";
                       cout<<  allTracks[j].KF->statePre.at<float>(3) <<endl;
                Mat prediction = allTracks[j].KF->predict();
        cout<<"predicted"<<endl;
               cout<<allTracks[j].KF->statePost.at<float>(0)<<" ";
               cout<<  allTracks[j].KF->statePost.at<float>(1) <<" ";
                cout<<         allTracks[j].KF->statePost.at<float>(2) <<" ";
                       cout<<  allTracks[j].KF->statePost.at<float>(3) <<endl;

                trackCount++;
                cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~"<<endl;
                KalmanFilter* KF = new KalmanFilter(4, 2, 0);
                Mat_<float> state(4, 1); /* (x, y, Vx, Vy) */
                KF->statePre.at<float>(0) = 100.0;
                KF->statePre.at<float>(1) = 200.0;
                KF->statePre.at<float>(2) = 0;
                KF->statePre.at<float>(3) = 0;
                KF->transitionMatrix = *(Mat_<float>(4, 4) << 1,0,0,0,   0,1,0,0,  0,0,1,0,  0,0,0,1);

                setIdentity(KF->measurementMatrix);
                setIdentity(KF->processNoiseCov, Scalar::all(1e-4));
                setIdentity(KF->measurementNoiseCov, Scalar::all(1e-1));
                setIdentity(KF->errorCovPost, Scalar::all(.1));



                    cout<<"unpredicted"<<endl;
                    cout<<KF->statePre.at<float>(0)<<" ";
                    cout<<  KF->statePre.at<float>(1) <<" ";
                    cout<<         KF->statePre.at<float>(2) <<" ";
                    cout<<  KF->statePre.at<float>(3) <<endl;
                    Mat prediction2 = KF->predict();
                    cout<<"predicted"<<endl;
                    cout<<KF->statePost.at<float>(0)<<" ";
                    cout<<  KF->statePost.at<float>(1) <<" ";
                    cout<<         KF->statePost.at<float>(2) <<" ";
                    cout<<  KF->statePost.at<float>(3) <<endl;
                    return 0;
                    break;
                }
            }

            for (int i=0;i<totalFish;i++)
            {
                if (!allTracks[i].live)
                    continue;
                int j = allTracks[i].thisPos;
                allTracks[i].x=dataXY[j*2+0];
                allTracks[i].y=dataXY[j*2+1];

                Mat_<float> measurement(2,1);
                // Get mouse point
                measurement(0) = allTracks[i].x;
                measurement(1) = allTracks[i].y;
                cout<<"uncorrected"<<endl;
                cout<<allTracks[i].KF->statePre.at<float>(0)<<" ";
                cout<<  allTracks[i].KF->statePre.at<float>(1) <<" ";
                cout<<         allTracks[i].KF->statePre.at<float>(2) <<" ";
                cout<<  allTracks[i].KF->statePre.at<float>(3) <<endl;
                allTracks[i].KF->correct(measurement);
                cout<<"corrected"<<endl;
                cout<<allTracks[i].KF->statePost.at<float>(0)<<" ";
               cout<<  allTracks[i].KF->statePost.at<float>(1) <<" ";
                cout<<         allTracks[i].KF->statePost.at<float>(2) <<" ";
                       cout<<  allTracks[i].KF->statePost.at<float>(3) <<endl;
        Mat prediction = allTracks[i].KF->predict();
        allTracks[i].xp=prediction.at<float>(0);
        allTracks[i].yp=prediction.at<float>(1);
        cout<<allTracks[i].x<<" ";
        cout<<allTracks[i].y<<" ";
        cout<<allTracks[i].xp<<" ";
        cout<<allTracks[i].yp<<endl;

       


    }
    return trackCount;
}

/*
void deleteEntry(track *deleteMe, track **listP)
{

    nodeT *currP, *prevP;

    // * For 1st node, indicate there is no previous. ///
    prevP = NULL;

    *
     * Visit each node, maintaining a pointer to
     * the previous node we just visited.
     /
    for (currP = *listP;
            currP != NULL;
            prevP = currP, currP = currP->next) {

        if (currP == deleteMe) {  * Found it. *
            if (prevP == NULL) {
                / Fix beginning pointer. *
                *listP = currP->next;
            } else {
                *
                 * Fix previous node's next to
                 * skip over the removed node.
                 /
                prevP->next = currP->next;
            }

            delete currP;

            return;
        }
    }



}*/
