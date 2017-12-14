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

        SupportVectorMachine(vector<string>, vector<string>);
        SupportVectorMachine();
        virtual ~SupportVectorMachine();
        vector< float > get_svm_detector( const Ptr< SVM >& svm );

        vector< Mat > gradient_lst;
        vector< int > labels;
        HOGDescriptor hog;

        void startSvm();
        map<int, Mat> createTrainData();
        void extractHogFeatures(map<int, Mat> trainData, const Size wsize);
        void convert_data( const vector< Mat > & train_samples, Mat& trainData);
        Mat extractHogFeatures();
        void trainSVM();
        pair<int, float> SVMpredict(Mat);
    protected:
    private:
};

#endif // SUPPORTVECTORMACHINE_H
