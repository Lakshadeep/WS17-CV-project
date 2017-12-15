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

    Mat frame, frameGT;
    VideoCapture cap("./0009.avi"); // open the default camera
    VideoCapture capGT("./0009_GT.avi"); // open the default camera
    if(!cap.isOpened())
    {
        std::cout << "Failed to open" << std::endl;
        return -1;
    }

    cap >> frame; // get a new frame from camera
    //VideoWriter video(".0000_result.avi",CV_FOURCC('M','J','P','G'),10, Size(frame.cols,frame.rows),true);
    capGT >> frameGT;
    VideoWriter video("result_0009.avi",CV_FOURCC('X','V','I','D'), 10 , Size(frame.cols,frame.rows),true);


    namedWindow("Result", CV_WINDOW_AUTOSIZE );
    svmObj.hog.load("cars.yml");


    int no_of_pos_detections = 0, no_of_neg_detections = 0, missed_detections = 0;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Mat gt_binary;
    int original_no_of_cars;
    int missed_cars;

    for(;;)
    {
        cap >> frame; // get a new frame from camera
        capGT >> frameGT;
        if(!frame.empty() and !frameGT.empty())
        {

            vector< Rect > detections;
            vector< double > foundWeights;

            cvtColor( frameGT, frameGT, CV_BGR2GRAY );
            blur( frameGT, frameGT, Size(3,3) );
            threshold( frameGT, gt_binary, 100,255,THRESH_BINARY );
            findContours( gt_binary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
            original_no_of_cars =  contours.size();

            svmObj.hog.detectMultiScale( frame, detections, foundWeights );


            for ( size_t j = 0; j < detections.size(); j++ )
            {
                int total_detections = 0;
                if(foundWeights[j] > 0.5)
                {
                    Scalar color = Scalar( 0, foundWeights[j] * foundWeights[j] * 200, 0 );
                    rectangle( frame, detections[j], color, frame.cols / 400 + 1 );

                    Mat regionGT = frameGT(detections[j]);
                    //cvtColor(regionGT, regionGT, CV_BGR2GRAY );
                    int nonZeroPixelsGT = countNonZero(regionGT);
                    int totalPixelsGT = regionGT.rows * regionGT.cols;

                    if ((double)nonZeroPixelsGT/totalPixelsGT >= 0.3) {
                         no_of_pos_detections++;
                    }
                    else {
                         no_of_neg_detections++;
                    }
                    total_detections++;


                }
                missed_cars = original_no_of_cars - total_detections;
            }
            if(missed_cars > 0)
            {
                missed_detections = missed_detections + missed_cars;
            }

            cout << "No of positive detections:" << no_of_pos_detections << endl;
            cout << "No of negative detections:" << no_of_neg_detections << endl;
            cout << "No of missed detections:" << missed_detections << endl;


            video.write(frame);

            imshow("Result", frame);
            if(waitKey(10) >= 0) break;
        }
    }

    return 0;
}
