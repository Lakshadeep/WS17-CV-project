#ifndef SUPPORTVECTORMACHINE_H
#define SUPPORTVECTORMACHINE_H

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <map>
#include <math.h>
#include <dirent.h>
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

class SupportVectorMachine
{
    public:
        bool isTrainingComplete = false;
        vector<string> cars, noCars;
        Mat testImage;
        Ptr<SVM> svmTest;
        int positiveExamples = 0, negativeExamples = 0;
        HOGDescriptor hog;

        SupportVectorMachine(vector<string>, vector<string>);
        SupportVectorMachine();
        virtual ~SupportVectorMachine();

        pair<int, float> startSvm();
        pair<int, float> startSvm(Mat);
        map<int, Mat> createTrainData();
        Mat computeHog(Mat);
        pair<Mat, Mat> extractHogFeatures(map<int, Mat>);
        Mat extractHogFeatures();
        void trainSVM(pair<Mat, Mat>);
        pair<int, float> SVMpredict(Mat);
    protected:
    private:
};

#endif // SUPPORTVECTORMACHINE_H
