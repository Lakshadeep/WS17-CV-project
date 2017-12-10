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
//    SupportVectorMachine svmObj (vehicleFiles, noVehicleFiles);
//------------------------------------------------------------------


//------------------------------------------------------------------
// For testing the svm...
    Mat testImage;
    pair<int, float> clfConfidence;

    // Testing positive sample..
    cout<<"Testing positive sample..."<<endl;
    testImage = imread("./svm_data/vehicles/1.png", 0);
    SupportVectorMachine svmObjPos (testImage);
    clfConfidence = svmObjPos.startSvm();
    cout<<"Predicted class:  "<<clfConfidence.first<<endl;
    cout<<"Prediction confidence:  "<<clfConfidence.second<<endl;

    // Testing negative sample..
    cout<<"Testing negative sample..."<<endl;
    testImage = imread("./svm_data/non-vehicles/extra1.png", 0);
    SupportVectorMachine svmObjNeg (testImage);
    clfConfidence = svmObjNeg.startSvm();
    cout<<"Predicted class:  "<<clfConfidence.first<<endl;
    cout<<"Prediction confidence:  "<<clfConfidence.second<<endl;

//------------------------------------------------------------------
    return 0;
}
