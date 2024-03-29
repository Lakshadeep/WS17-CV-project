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

int evaluate(const char* inputPath, const char* gtPath, const char* resultPath, float windowSize, float stepSize,
                                                    float overlapThreshold, float confidenceThreshold, ofstream& myStream) {

    SupportVectorMachine svmObj;
    pair<int, float> clfConfidence;

    Mat frame, frameGT, region, regionGT;
    VideoCapture cap(inputPath);
    VideoCapture capGT(gtPath);
    if(!cap.isOpened() or !capGT.isOpened())
    {
        std::cout << "Failed to open" << std::endl;
        return -1;
    }

    cap >> frame;
    capGT >> frameGT;
    VideoWriter video(resultPath, CV_FOURCC('X','V','I','D'), 10 , Size(frame.cols,frame.rows), true);

    namedWindow("Result", CV_WINDOW_AUTOSIZE );
    int truePositives = 0, falsePositives = 0, falseNegatives = 0, allTruePositives = 0, CGT = 0;
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
            cvtColor( frameGT, frameGT, CV_BGR2GRAY );
            blur( frameGT, frameGT, Size(3,3) );
            threshold( frameGT, gt_binary, 100,255,THRESH_BINARY );
            findContours( gt_binary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
            original_no_of_cars =  contours.size();
            vector<Rect> boundRect( contours.size() );

            for( int i = 0; i < contours.size(); i++ )
            {
                boundRect[i] = boundingRect( Mat(contours[i]) );
            }

            string scoreString;
            missed_cars = original_no_of_cars;

            for(int i = 0; i < frame.rows - windowSize ; )
            {
                for(int j = 0; j < frame.cols - windowSize;  )
                {
                    bool isPositiveDetection = false;
                    Rect roi(j, i, windowSize, windowSize);
                    region = frame(roi);
                    clfConfidence = svmObj.startSvm(region);
                    if (clfConfidence.first > 0 and clfConfidence.second >= confidenceThreshold) {

                        regionGT = frameGT(roi);
                        int nonZeroPixelsGT = countNonZero(regionGT);
                        int totalPixelsGT = regionGT.rows * regionGT.cols;

                        for (int recIndex = 0; recIndex < boundRect.size(); recIndex++) {
                            if ((double)((boundRect[recIndex] & roi).area())/totalPixelsGT >= overlapThreshold ) {
                                isPositiveDetection = true;
                                truePositives++;
                                boundRect.erase(boundRect.begin() + recIndex);
                                break;
                            }
                        }

                        if ( (double)nonZeroPixelsGT/totalPixelsGT >= overlapThreshold) {
                            allTruePositives ++;
                        }
                        else {
                            falsePositives ++;
                        }

                        cv::rectangle(frame,roi,cv::Scalar(0, 0, 255));
                        scoreString = to_string(clfConfidence.second);
                        scoreString.erase(scoreString.find_last_not_of('0') + 1, string::npos );
                        putText(frame, scoreString, cv::Point(j, i + 12), FONT_HERSHEY_PLAIN, 1,cv::Scalar(0, 255, 0), 2);
                    }
                    if (isPositiveDetection) missed_cars--;

                    j = j + stepSize;
                }
                i = i + stepSize;
            }

            if(missed_cars > 0)
            {
                falseNegatives = falseNegatives + missed_cars;
            }

            CGT += original_no_of_cars;

            cout << "No of positive detections:" << truePositives << endl;
            cout << "No of negative detections:" << falsePositives << endl;
            cout << "No of missed detections:" << falseNegatives << endl;

            video.write(frame);

            imshow("Result", frame);
            if(waitKey(1) >= 0) break;
        }
        else{
            break;
        }
    }
    myStream << CGT << ",";
    myStream << truePositives << "," << allTruePositives - truePositives << "," << falsePositives << "," << falseNegatives << ",";
    myStream << (double)(truePositives)/(CGT) << "," ;
    myStream << (double)(truePositives)/(truePositives+falsePositives) << "," ;
    myStream << (double)(truePositives)/(truePositives+falseNegatives) << endl;

    destroyAllWindows();
}

int main()
{
    //**************************** Train the svm **********************************

    //trainSVM("./svm_data/vehicles/", "./svm_data/non-vehicles/");

    //*****************************************************************************

    //**************************** Evaluate algorithm *****************************

    ofstream myStream;
    myStream.open("result.txt", ofstream::app);
    myStream << "Video,";
    myStream << "CGT," << "True Positives," << "Repeated True Positives," << "False Positives," << "False Negatives,";
    myStream << "Accuracy," << "Precision," << "Recall" << endl;

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
        evaluate(inputPath.c_str(), gtPath.c_str(), resultPath.c_str(), 50, 30, 0.3, 0.45, myStream);
    }

    myStream.close();

    //*****************************************************************************
    return 0;
}
