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

using namespace std;

void trainSVM(const char* positiveDirectory, const char* negativeDirectory) {

    vector<string> vehicleFiles;
    vector<string> noVehicleFiles;
    DIR *dpdf;
    struct dirent *epdf;
    dpdf = opendir(positiveDirectory);
    if(dpdf!= NULL)
    {
        while (epdf = readdir(dpdf))
        {
            string file = epdf->d_name;
            vehicleFiles.push_back(positiveDirectory + file);
        }
    }
    closedir(dpdf);

    dpdf = opendir(negativeDirectory);
    if(dpdf!= NULL)
    {
        while (epdf = readdir(dpdf))
        {
            string file = epdf->d_name;
            noVehicleFiles.push_back(negativeDirectory + file);
        }
    }
    closedir(dpdf);

    SupportVectorMachine svmObj (vehicleFiles, noVehicleFiles);
    svmObj.startSvm();
}

int evaluate(const char* inputPath, const char* gtPath, const char* resultPath, float beliefThreshold, float overlapThreshold, ofstream& myStream)
{

    SupportVectorMachine svmObj;

    Mat frame, frameGT;
    VideoCapture cap(inputPath);
    VideoCapture capGT(gtPath);
    if(!cap.isOpened() or !capGT.isOpened())
    {
        std::cout << "Failed to open" << std::endl;
        return -1;
    }

    cap >> frame;
    capGT >> frameGT;
    VideoWriter video(resultPath,CV_FOURCC('X','V','I','D'), 10 , Size(frame.cols,frame.rows),true);

    namedWindow("Result", CV_WINDOW_AUTOSIZE );
    svmObj.hog.load("cars.yml");

    int no_of_pos_detections = 0, no_of_neg_detections = 0, missed_detections = 0;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Mat gt_binary;
    int original_no_of_cars;
    int missed_cars;

    while(true)
    {
        cap >> frame;
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

            missed_cars = original_no_of_cars;
            for ( size_t j = 0; j < detections.size(); j++ )
            {
                bool isPositiveDetection = false;
                if(foundWeights[j] > beliefThreshold)
                {
                    Scalar color = Scalar( 0, foundWeights[j] * foundWeights[j] * 200, 0 );
                    rectangle( frame, detections[j], color, frame.cols / 400 + 1 );

                    Mat regionGT = frameGT(detections[j]);
                    int nonZeroPixelsGT = countNonZero(regionGT);
                    int totalPixelsGT = regionGT.rows * regionGT.cols;

                    if ((double)nonZeroPixelsGT/totalPixelsGT >= overlapThreshold) {
                         no_of_pos_detections++;
                         isPositiveDetection = true;
                    }
                    else {
                         no_of_neg_detections++;
                    }
                }
                if (isPositiveDetection) missed_cars--;
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
        else {
            break;
        }
    }
    myStream << no_of_pos_detections << "," << no_of_neg_detections << "," << missed_detections << "," << (double)(no_of_pos_detections)/(no_of_pos_detections+missed_detections) << "," << (double)no_of_neg_detections/no_of_pos_detections << endl;
    destroyAllWindows();
}

int test(const char* inputPath, const char* resultPath, float beliefThreshold) {

    SupportVectorMachine svmObj;

    Mat frame;
    VideoCapture cap(inputPath);
    if(!cap.isOpened())
    {
        std::cout << "Failed to open" << std::endl;
        return -1;
    }

    cap >> frame;
    VideoWriter video(resultPath, CV_FOURCC('X','V','I','D'), 10 , Size(frame.cols,frame.rows), true);

    namedWindow("Result", CV_WINDOW_AUTOSIZE );
    svmObj.hog.load("cars.yml");

    while(true)
    {
        cap >> frame;
        if(!frame.empty())
        {
            vector< Rect > detections;
            vector< double > foundWeights;

            svmObj.hog.detectMultiScale( frame, detections, foundWeights );

            for ( size_t j = 0; j < detections.size(); j++ )
            {
                if(foundWeights[j] > beliefThreshold)
                {
                    Scalar color = Scalar( 0, foundWeights[j] * foundWeights[j] * 200, 0 );
                    rectangle( frame, detections[j], color, frame.cols / 400 + 1 );
                }
            }

            video.write(frame);

            imshow("Result", frame);
            if(waitKey(10) >= 0) break;
        }
        else {
            break;
        }
    }
    return 0;
}

int main()
{
    //**************************** Train the svm **********************************

    //trainSVM("./svm_data/vehicles/", "./svm_data/non-vehicles/");

    //*****************************************************************************

    //**************************** Evaluate algorithm *****************************

    //evaluate("./0000.avi", "./0000_GT.avi", "./result_0000.avi", 0.5, 0.3);

    //*****************************************************************************


    //**************************** Evaluate algorithm *****************************

    ofstream myStream;
    myStream.open("result.txt", ofstream::app);
    myStream << "Video," << "Positive Detections," << "Negative Detections," << "Missed Detections," << "Accuracy," << "Negative to Positive Ratio" << endl;
    //myStream << "8,";
    //evaluate("./videos/0008.avi", "./videos/0008_GT.avi", "./result_0008.avi", 50, 30, 0.3, 0.4, myStream);


    //*****************************************************************************
    string inputPath, gtPath, resultPath;

    for(int i = 0; i< 21; i++) {

        if (i<=9) {
            inputPath = "./videos/000"+to_string(i)+".avi";
            gtPath = "./videos/000"+to_string(i)+"_GT.avi";
            resultPath = "./result/result_000"+to_string(i)+".avi";
        }
        else {
            inputPath = "./videos/00"+to_string(i)+".avi";
            gtPath = "./videos/00"+to_string(i)+"_GT.avi";
            resultPath = "./result/result_00"+to_string(i)+".avi";
        }
        myStream << to_string(i) << ",";
        evaluate(inputPath.c_str(), gtPath.c_str(), resultPath.c_str(), 0.5, 0.3, myStream);
    }

    //*****************************************************************************





    //**************************** Test algorithm *********************************
    // This is required for videos which do not have a ground truth...

    //test("./0010.avi", "./result_0010.avi", 0.5);

    //*****************************************************************************
    return 0;
}
