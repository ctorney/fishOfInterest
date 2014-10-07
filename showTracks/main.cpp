
#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <netcdfcpp.h>

using namespace cv;
using namespace std;

int setUpNetCDF(NcFile* dataFile, int nFrames, int nFish);
void cvDrawDottedRect(cv::Mat* img, int x, int y, CvScalar color);

int main( int argc, char** argv )
{
    string dataDir = "/home/ctorney/data/fishPredation/";

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
    Size S = Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
            (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT));

    string outMovie = "./" + trialName + "_TRACKED.avi";
    int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC)); 
    VideoWriter outputVideo;      
    outputVideo.open(outMovie, ex, cap.get(CV_CAP_PROP_FPS), S, true);


    Scalar colors[nFish];
    colors[0] = Scalar(255,255,255);
    colors[1] = Scalar(0,255,255);
    colors[2] = Scalar(34,34,200);
    colors[3] = Scalar(225,50,50);

    // **************************************************************************************************
    // open the netcdf file
    // **************************************************************************************************
    string ncFileName = dataDir + "tracked/linked" + trialName + ".nc";
    NcFile dataFile(ncFileName.c_str(), NcFile::ReadOnly);

    if (!dataFile.is_valid())
    {
        cout << "Couldn't open netcdf file!\n";
        return -1;
    }

    NcDim* trDim = dataFile.get_dim("track");
    int totalTracks = trDim->size();
    float *dataOut = new float[totalTracks*2];
    float *fidOut = new float[totalTracks];
    float *certOut = new float[totalTracks];
    // get the variable to store the positions
    NcVar* xy = dataFile.get_var("trXY");
    NcVar* fid = dataFile.get_var("fid");
    NcVar* cert = dataFile.get_var("certID");

    // **************************************************************************************************
    // loop over all frames and record positions
    // **************************************************************************************************

    int offset = 25;
    Mat frame, gsFrame;
    //       for (int t=0;t<totalTracks;t++)
    //         cout<<dataOut[t*2  ]<<" "<<dataOut[t*2 + 1]<<endl;
    //   return 0;
    
    for(int f=0;f<fCount;f++)
    {
        if (!cap.read(frame))             
            break;
        if (f<fStart)
            continue;


 //       xy->set_cur(0, f - fStart, 0, 0);
        xy->set_cur(0, f-fStart, 0);
        xy->get(dataOut, totalTracks, 1, 2);
        fid->set_cur(0, f-fStart, 0);
        fid->get(fidOut, totalTracks, 1);
        cert->set_cur(0, f-fStart, 0);
        cert->get(certOut, totalTracks, 1);
 //       xy->get(&dataOut[0][0], totalTracks, 1, 2);


  //      for (int t=0;t<totalTracks;t++)
  //          if ((dataOut[t*2]>0)&&(fidOut[t]>=0))
  //              cout<<fidOut[t]<<endl;
 //               int col = (t%nFish);
        putText(frame, "uncertainty", Point2f((int)1600, (int)75), FONT_HERSHEY_PLAIN, 1.5,  0, 2);
        for (int t=0;t<totalTracks;t++)
            if ((dataOut[t*2]>0)&&(fidOut[t]>=0))
            {
                ostringstream ss;
                ss << "fish " <<fidOut[t]<< ": ";
                ss << scientific << setprecision(3) << certOut[t];
 //               cout<<certOut[t]<<endl;
                int col=fidOut[t];
                putText(frame, ss.str(), Point2f((int)1600, (int)100+offset*col), FONT_HERSHEY_PLAIN, 1.5,  colors[col], 2);
                cvDrawDottedRect(&frame, (int)dataOut[t*2], (int)dataOut[t*2 + 1], colors[col]);
   //             cout<<t<<endl;
                cout<<t<<" "<<certOut[t]<<" "<<fidOut[t]<<endl;
            }
        cout<<"~~~ "<<f<<" ~~~~~"<<endl;

        outputVideo << frame;
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
    // dimension for each frame
    NcDim* frDim = dataFile->add_dim("frame", nFrames);
    // dimension for individual fish
    NcDim* iDim = dataFile->add_dim("fish", nFish);
    // xy dimension for vectors
    NcDim* xyDim = dataFile->add_dim("xy", 2);
    // dimension for tracks (unlimited as new tracks are created when a fish is lost)
    //   NcDim* trDim = dataFile->add_dim("track");

    // define a netCDF variable for the positions of individuals
    dataFile->add_var("pxy", ncFloat, frDim, iDim, xyDim);
    // define a netCDF variable for the positions of linked tracks
    // dataFile->add_var("trxy", ncFloat, trDim, frDim, xyDim);
    // linked tracks following smoothing
    // dataFile->add_var("trxy_sm", ncFloat, trDim, frDim, xyDim);
    // velocities
    //dataFile->add_var("tr_vel", ncFloat, trDim, frDim, xyDim);
    // accelerations 
    //dataFile->add_var("tr_accel", ncFloat, trDim, frDim, xyDim);
    // variable for ID of fish
    //dataFile->add_var("fid", ncShort, trDim, frDim);

    return 0;

}


// drawDottedRect
// Maxime Tremblay, 2010, Université Laval, Québec city, QB, Canada


void cvDrawDottedLine(cv::Mat* img, Point pt1, Point pt2, CvScalar color, int thickness, int lengthDash, int lengthGap)
{
    Mat img1;
    Point pt11,pt12;
    LineIterator it(*img, pt1, pt2, 8, false);            // get a line iterator

    for(int i = 0; i < it.count; i++,it++)
        if ( (i%lengthGap) ==0 )
        {
            Point tmpStart = it.pos();
            for (int j=0;j<lengthDash;j++)
            {
                i++;
                it++;
                if (i==it.count)
                    break;
            }

            line(*img, tmpStart, it.pos(), color, thickness, 8, 0);
        }         // every 5'th pixel gets dropped, blue stipple lin

    /*   CvLineIterator iterator;
         int count = cvInitLineIterator( img, pt1, pt2, &iterator, lineType, leftToRight );
         int offset,x,y;


         for( int i = 0; i < count; i= i + (lenghOfDots*2-1) )
         {
         if(i+lenghOfDots > count)
         break;

        offset = iterator.ptr - (uchar*)(img->imageData);
        y = offset/img->widthStep;
        x = (offset - y*img->widthStep)/(3*sizeof(uchar) // * size of pixel * /);

        CvPoint lTemp1 = cvPoint(x,y);
        for(int j=0;j<lenghOfDots-1;j++) //I want to know have the last of these in the iterator
            CV_NEXT_LINE_POINT(iterator);

        offset = iterator.ptr - (uchar*)(img->imageData);
        y = offset/img->widthStep;
        x = (offset - y*img->widthStep)/(3*sizeof(uchar) // * size of pixel * /);

        CvPoint lTemp2 = cvPoint(x,y);
        line(img,lTemp1,lTemp2,color,thickness,lineType);
        for(int j=0;j<lenghOfDots;j++)
            CV_NEXT_LINE_POINT(iterator);
    }*/
}

void cvDrawDottedRect(cv::Mat* img, int x, int y, CvScalar color)
{ 
    int hwidth = 20;
    int corner = 4;
    int cross = 4;

    // corners 
    
    Point p1(x-hwidth,y-hwidth);
    Point p2(x+hwidth,y-hwidth);
    Point p3(x+hwidth,y+hwidth);
    Point p4(x-hwidth,y+hwidth);
    // draw box
    cvDrawDottedLine(img, p1, p2, color, 1, 2, 4);
    cvDrawDottedLine(img, p2, p3, color, 1, 2, 4);
    cvDrawDottedLine(img, p3, p4, color, 1, 2, 4);
    cvDrawDottedLine(img, p4, p1, color, 1, 2, 4);
    // draw corners
    Point x_off(corner, 0);
    Point y_off(0, corner);
    cvDrawDottedLine(img, p1, p1 + x_off, color, 2, 10, 10);
    cvDrawDottedLine(img, p1, p1 + y_off, color, 2, 10, 10);
    cvDrawDottedLine(img, p2, p2 - x_off, color, 2, 10, 10);
    cvDrawDottedLine(img, p2, p2 + y_off, color, 2, 10, 10);
    cvDrawDottedLine(img, p3, p3 - x_off, color, 2, 10, 10);
    cvDrawDottedLine(img, p3, p3 - y_off, color, 2, 10, 10);
    cvDrawDottedLine(img, p4, p4 + x_off, color, 2, 10, 10);
    cvDrawDottedLine(img, p4, p4 - y_off, color, 2, 10, 10);
    // draw cross
    Point x_coff(cross, 0);
    Point y_coff(0, cross);
    p1 = Point(x,y-hwidth);
    p2 = Point(x,y+hwidth);
    p3 = Point(x-hwidth,y);
    p4 = Point(x+hwidth,y);
    cvDrawDottedLine(img, p1 + y_coff, p1, color, 1, 10, 10);
    cvDrawDottedLine(img, p2, p2 - y_coff, color, 1, 10, 10);
    cvDrawDottedLine(img, p3 + x_coff, p3, color, 1, 10, 10);
    cvDrawDottedLine(img, p4 - x_coff, p4, color, 1, 10, 10);


 /*   LineIterator it(img, p1, p2, 8);            // get a line iterator
    for(int i = 0; i < it.count; i++,it++)
            if ( i%5!=0 ) {(*it)[0] = 200;}         // every 5'th pixel gets dropped, blue stipple lin

    CvPoint tempPt1 = cvPoint(pt2.x,pt1.y);
    CvPoint tempPt2 = cvPoint(pt1.x,pt2.y);
    cvDrawDottedLine(img,pt1,tempPt1,color,thickness,lenghOfDots,lineType, 0);
    cvDrawDottedLine(img,tempPt1,pt2,color,thickness,lenghOfDots,lineType, 0);
    cvDrawDottedLine(img,pt2,tempPt2,color,thickness,lenghOfDots,lineType, 1);
    cvDrawDottedLine(img,tempPt2,pt1,color,thickness,lenghOfDots,lineType, 1);*/
}

