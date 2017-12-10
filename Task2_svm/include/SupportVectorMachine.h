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

        SupportVectorMachine(vector<string>, vector<string>);
        SupportVectorMachine(Mat);
        virtual ~SupportVectorMachine();

        pair<int, float> startSvm();
        map<int, Mat> createTrainData();
        pair<Mat, Mat> extractHogFeatures(map<int, Mat>);
        Mat extractHogFeatures();
        void trainSVM(pair<Mat, Mat>);
        pair<int, float> SVMpredict(Mat);
    protected:
    private:
};

#endif // SUPPORTVECTORMACHINE_H
