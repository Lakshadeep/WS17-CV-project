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

    Mat frame, frameGray, region;
    VideoCapture cap("./0008_manual.avi"); // open the default camera
    if(!cap.isOpened())
    {
        std::cout << "Failed to open" << std::endl;
        return -1;
    }

    cap >> frame; // get a new frame from camera
    VideoWriter video("result_0008.avi",CV_FOURCC('M','J','P','G'),10, Size(frame.cols,frame.rows),true);

    namedWindow("Result", CV_WINDOW_AUTOSIZE );

    for(;;)
    {
        cap >> frame; // get a new frame from camera
        if(!frame.empty())
        {
            cvtColor(frame, frameGray, CV_BGR2GRAY );
            int windowSize = 50,stepSize = 30;
            string scoreString;

            for(int i = 0; i < frameGray.rows - windowSize ; )
            {
                for(int j = 0; j < frameGray.cols - windowSize;  )
                {
                    Rect roi(j, i, windowSize, windowSize);
                    region = frameGray(roi);
                    clfConfidence = svmObj.startSvm(region);
                    if (clfConfidence.first > 0){
                        cv::rectangle(frame,cv::Point(j, i),cv::Point(j + windowSize, i + windowSize),cv::Scalar(0, 0, 255));
                        scoreString = to_string(clfConfidence.second);
                        scoreString.erase(scoreString.find_last_not_of('0') + 1, string::npos );
                        putText(frame, scoreString, cv::Point(j, i + 12), FONT_HERSHEY_PLAIN, 1,cv::Scalar(0, 255, 0), 2);
                    }
                    j = j + stepSize;
                }
                i = i + stepSize;
            }
            video.write(frame);

            imshow("Result", frame);
            if(waitKey(10) >= 0) break;
        }
    }
    return 0;
}
