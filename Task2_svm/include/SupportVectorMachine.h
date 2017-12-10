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
        SupportVectorMachine(vector<string>, vector<string>);
        virtual ~SupportVectorMachine();
        int startSvm();
        map<int, Mat> createTrainData();
        void extractHogFeatures(map<int, Mat>);
        void trainSVM(map<int, Mat>);
        void isAlreadyTrained();
    protected:
    private:
};

#endif // SUPPORTVECTORMACHINE_H
