#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <dirent.h>
#include "SupportVectorMachine.h"
#include "opencv2/ximgproc/segmentation.hpp"

using namespace std;
using namespace cv::ximgproc::segmentation;

int main()
{

//------------------------------------------------------------------
// For training the svm...

//    vector<string> vehicleFiles;
//    vector<string> noVehicleFiles;
//    DIR *dpdf;
//    struct dirent *epdf;
//    dpdf = opendir("./svm_data/vehicles/");
//    if(dpdf!= NULL)
//    {
//        while (epdf = readdir(dpdf))
//        {
//            string file = epdf->d_name;
//            vehicleFiles.push_back("./svm_data/vehicles/"+ file);
//        }
//    }
//    closedir(dpdf);
//
//    dpdf = opendir("./svm_data/vehicles_GTI_Far/");
//    if(dpdf!= NULL)
//    {
//        while (epdf = readdir(dpdf))
//        {
//            string file = epdf->d_name;
//            vehicleFiles.push_back("./svm_data/vehicles_GTI_Far/"+ file);
//        }
//    }
//    closedir(dpdf);
//
//    dpdf = opendir("./svm_data/vehicles_GTI_Left/");
//    if(dpdf!= NULL)
//    {
//        while (epdf = readdir(dpdf))
//        {
//            string file = epdf->d_name;
//            vehicleFiles.push_back("./svm_data/vehicles_GTI_Left/"+ file);
//        }
//    }
//    closedir(dpdf);
//
//	dpdf = opendir("./svm_data/vehicles_GTI_MiddleClose/");
//    if(dpdf!= NULL)
//    {
//        while (epdf = readdir(dpdf))
//        {
//            string file = epdf->d_name;
//            vehicleFiles.push_back("./svm_data/vehicles_GTI_MiddleClose/"+ file);
//        }
//    }
//    closedir(dpdf);
//
//	dpdf = opendir("./svm_data/vehicles_GTI_Right/");
//    if(dpdf!= NULL)
//    {
//        while (epdf = readdir(dpdf))
//        {
//            string file = epdf->d_name;
//            vehicleFiles.push_back("./svm_data/vehicles_GTI_Right/"+ file);
//        }
//    }
//    closedir(dpdf);
//
//    dpdf = opendir("./svm_data/non-vehicles/");
//    if(dpdf!= NULL)
//    {
//        while (epdf = readdir(dpdf))
//        {
//            string file = epdf->d_name;
//            noVehicleFiles.push_back("./svm_data/non-vehicles/"+ file);
//        }
//    }
//    closedir(dpdf);
//
//    dpdf = opendir("./svm_data/non-vehicles_GTI/");
//    if(dpdf!= NULL)
//    {
//        while (epdf = readdir(dpdf))
//        {
//            string file = epdf->d_name;
//            noVehicleFiles.push_back("./svm_data/non-vehicles_GTI/"+ file);
//        }
//    }
//    closedir(dpdf);
//
//    SupportVectorMachine svmObj (vehicleFiles, noVehicleFiles);
//    svmObj.startSvm();
//------------------------------------------------------------------


//------------------------------------------------------------------
// For testing the svm...
//    Mat testImage;
//    pair<int, float> clfConfidence;

    // Testing positive sample..
//    cout<<"Testing positive sample..."<<endl;
//    testImage = imread("./svm_data/vehicles/5110.png", 0);
//    SupportVectorMachine svmObj;
//    clfConfidence = svmObj.startSvm(testImage);
//    cout<<"Predicted class:  "<<clfConfidence.first<<endl;
//    cout<<"Prediction confidence:  "<<clfConfidence.second<<endl;
//
//    // Testing negative sample..
//    cout<<"Testing negative sample..."<<endl;
//    testImage = imread("./svm_data/non-vehicles/extra5060.png", 0);
//    clfConfidence = svmObj.startSvm(testImage);
//    cout<<"Predicted class:  "<<clfConfidence.first<<endl;
//    cout<<"Prediction confidence:  "<<clfConfidence.second<<endl;

//------------------------------------------------------------------

    SupportVectorMachine svmObj;
    pair<int, float> clfConfidence;

    Mat frame, frameGT, frameGray, region, regionGT, frameYCrCb;
    VideoCapture cap("./0008_xvid.avi");
    if(!cap.isOpened())
    {
        std::cout << "Failed to open" << std::endl;
        return -1;
    }

    VideoCapture capGT("./0008_GT.avi");
    if(!capGT.isOpened())
    {
        std::cout << "Failed to open" << std::endl;
        return -1;
    }

    cap >> frame;
    capGT >> frameGT;
    VideoWriter video("result_0008.avi",CV_FOURCC('X','V','I','D'), 10 , Size(frame.cols,frame.rows),true);

    namedWindow("Result", CV_WINDOW_AUTOSIZE );
    float totalFrames = 0, positiveFrame = 0, negativeFrame = 0;
    while(true)
    {
        cap >> frame;
        capGT >> frameGT;
        if(!frame.empty() and !frameGT.empty())
        {
            int windowSize = 50, stepSize = 30, positivePrediction = 0, negativePrediction = 0;
            float overlapRequired = 0.4, nonZeroPixelsGT, totalPixelsGT;
            string scoreString;

            for(int i = 0; i < frame.rows - windowSize ; )
            {
                for(int j = 0; j < frame.cols - windowSize;  )
                {
                    Rect roi(j, i, windowSize, windowSize);
                    region = frame(roi);
                    clfConfidence = svmObj.startSvm(region);
                    if (clfConfidence.first > 0 and clfConfidence.second >= 0.45) {
                        regionGT = frameGT(roi);
                        cvtColor(regionGT, regionGT, CV_BGR2GRAY );

                        nonZeroPixelsGT = countNonZero(regionGT);
                        totalPixelsGT = regionGT.rows * regionGT.cols;
                        cout<<"Non zero pixels: "<<nonZeroPixelsGT<<"  Total pixels: "<<totalPixelsGT<<endl;

                        if ( nonZeroPixelsGT/totalPixelsGT >= overlapRequired) {
                            positivePrediction ++;
                        }

                        else {
                            negativePrediction ++;
                        }
                        cv::rectangle(frame,roi,cv::Scalar(0, 0, 255));
                        scoreString = to_string(clfConfidence.second);
                        scoreString.erase(scoreString.find_last_not_of('0') + 1, string::npos );
                        putText(frame, scoreString, cv::Point(j, i + 12), FONT_HERSHEY_PLAIN, 1,cv::Scalar(0, 255, 0), 2);
                    }
                    j = j + stepSize;
                }
                i = i + stepSize;
            }
            //cout<<"Positive predictions: "<< positivePrediction<< "  Negative predictions: "<< negativePrediction<< endl;
            if (positivePrediction > negativePrediction) {
                positiveFrame ++;
            }
            else {
                negativeFrame ++;
            }
            totalFrames ++;
            positivePrediction = 0, negativePrediction = 0;
            video.write(frame);

            imshow("Result", frame);
            if(waitKey(10) >= 0) break;
        }
        else{
            break;
        }
    }
    cout<< "Positive Frames: "<< positiveFrame<< " Negative Frames: "<<negativeFrame<< " Total Frames: "<< totalFrames<<endl;
    cout<< "Final score: "<< positiveFrame/totalFrames<<endl;
    return 0;
}
