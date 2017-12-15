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
//    dpdf = opendir("/media/lakshadeep/Common/MAS/Semester2/computer_vision/project_final/test_data/datasets/svm_data/vehicles/");
//    if(dpdf!= NULL)
//    {
//        while (epdf = readdir(dpdf))
//        {
//            string file = epdf->d_name;
//            vehicleFiles.push_back("/media/lakshadeep/Common/MAS/Semester2/computer_vision/project_final/test_data/datasets/svm_data/vehicles/"+ file);
//        }
//    }
//    closedir(dpdf);
//
//    dpdf = opendir("/media/lakshadeep/Common/MAS/Semester2/computer_vision/project_final/test_data/datasets/svm_data/non-vehicles/");
//    if(dpdf!= NULL)
//    {
//        while (epdf = readdir(dpdf))
//        {
//            string file = epdf->d_name;
//            noVehicleFiles.push_back("/media/lakshadeep/Common/MAS/Semester2/computer_vision/project_final/test_data/datasets/svm_data/non-vehicles/"+ file);
//        }
//    }
//    closedir(dpdf);
//
//    SupportVectorMachine svmObj (vehicleFiles, noVehicleFiles);
//    svmObj.startSvm();
//------------------------------------------------------------------



    SupportVectorMachine svmObj;

    Mat frame;
    VideoCapture cap("./0009.avi"); // open the default camera
    if(!cap.isOpened())
    {
        std::cout << "Failed to open" << std::endl;
        return -1;
    }

    cap >> frame; // get a new frame from camera
    //VideoWriter video(".0009.avi",CV_FOURCC('M','J','P','G'),10, Size(frame.cols,frame.rows),true);

    namedWindow("Result", CV_WINDOW_AUTOSIZE );
    svmObj.hog.load("cars.yml");

    for(;;)
    {
        cap >> frame; // get a new frame from camera
        if(!frame.empty())
        {

            vector< Rect > detections;
            vector< double > foundWeights;

            svmObj.hog.detectMultiScale( frame, detections, foundWeights );
            for ( size_t j = 0; j < detections.size(); j++ )
            {
                if(foundWeights[j] > 0.5)
                {
                    Scalar color = Scalar( 0, foundWeights[j] * foundWeights[j] * 200, 0 );
                    rectangle( frame, detections[j], color, frame.cols / 400 + 1 );
                }
            }


            //video.write(frame);

            imshow("Result", frame);
            if(waitKey(10) >= 0) break;
        }
    }
    return 0;
}
