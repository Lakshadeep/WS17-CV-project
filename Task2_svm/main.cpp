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
                                                    float overlapThreshold, float confidenceThreshold) {

    SupportVectorMachine svmObj;
    pair<int, float> clfConfidence;

    Mat frame, frameGT, frameGray, region, regionGT, frameYCrCb;
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
    float totalFrames = 0, positiveFrame = 0, negativeFrame = 0, totalPosPred = 0, totalNegPred = 0, totalPred = 0;
    while(true)
    {
        cap >> frame;
        capGT >> frameGT;
        if(!frame.empty() and !frameGT.empty())
        {
            int positivePrediction = 0, negativePrediction = 0;
            float nonZeroPixelsGT, totalPixelsGT;
            string scoreString;

            for(int i = 0; i < frame.rows - windowSize ; )
            {
                for(int j = 0; j < frame.cols - windowSize;  )
                {
                    Rect roi(j, i, windowSize, windowSize);
                    region = frame(roi);
                    clfConfidence = svmObj.startSvm(region);
                    if (clfConfidence.first > 0 and clfConfidence.second >= confidenceThreshold) {
                        regionGT = frameGT(roi);
                        cvtColor(regionGT, regionGT, CV_BGR2GRAY );

                        nonZeroPixelsGT = countNonZero(regionGT);
                        totalPixelsGT = regionGT.rows * regionGT.cols;
                        cout<<"Non zero pixels: "<<nonZeroPixelsGT<<"  Total pixels: "<<totalPixelsGT<<endl;

                        if ( nonZeroPixelsGT/totalPixelsGT >= overlapThreshold) {
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
            if (positivePrediction > negativePrediction) {
                positiveFrame ++;
            }
            else {
                negativeFrame ++;
            }
            totalFrames ++;
            totalPosPred += positivePrediction;
            totalNegPred += negativePrediction;
            totalPred +=  positivePrediction + negativePrediction;
            positivePrediction = 0, negativePrediction = 0;
            video.write(frame);

            imshow("Result", frame);
            if(waitKey(10) >= 0) break;
        }
        else{
            break;
        }
    }
    cout<< "Total predictions: "<< totalPred<< " Total positive predictions:"<<totalPosPred<< " Total negative predictions:"<<totalNegPred<<endl;
    cout<< "Positive prediction ratio: "<<totalPosPred/totalPred<< " Negative prediction ratio: "<<totalNegPred/totalPred<<endl;
    cout<<" Total Frames: "<< totalFrames<< " Positive Frames: "<< positiveFrame<< " Negative Frames: "<<negativeFrame<<endl;
    cout<< "Final score (positive frames/ total frames): "<< positiveFrame/totalFrames<<endl;

}

int main()
{
    //**************************** Train the svm **********************************

    //trainSVM("./svm_data/vehicles/", "./svm_data/non-vehicles/");

    //*****************************************************************************

    //**************************** Evaluate algorithm *****************************

    evaluate("./0008.avi", "./0008_GT.avi", "./result_0008.avi", 50, 30, 0.5, 0.4);

    //*****************************************************************************

    return 0;
}
