/*********************************** NOTE ********************************************/
/// This program was used for performing the analysis of pixel-wise correlation technique.
/// It takes the input image, applies pixel-wise correlation to it with each template
/// and saves the results in a text file.
/*************************************************************************************/


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
#include "opencv2/ximgproc/segmentation.hpp"
#include <fstream>

using namespace cv;
using namespace std;
using namespace cv::ximgproc::segmentation;

/// Directory/File paths
string test_data_directory = "/media/lakshadeep/Common/MAS/Semester2/computer_vision/project_final/test_data/ss/good/36040160/";
string test_file = "150.png";
string model_directory = "/media/lakshadeep/Common/MAS/Semester2/computer_vision/project_final/models/distant/";
string experiment_results_directory = "/media/lakshadeep/Common/MAS/Semester2/computer_vision/project_final/experiment_results/correlation/36040160/";

float correlation(Mat reg_prop, vector<string> model_list)
{
    float corr = 0.0, pt = 0, sum_T= 0, sum_R = 0, template_count = 0;
    Scalar ssim=0;
    Mat T,R;
    int index = 0;
    T = imread(model_list[0], 0);
    resize(reg_prop, R, Size(T.size()), 0, 0);
    imshow("Resized", R);
    imwrite(experiment_results_directory + test_file, R);
    R.convertTo(R, CV_32F);
    sum_R = sum(R)[0];
    ofstream log_file;
    log_file.open (experiment_results_directory + test_file + string(".txt"));

    for(auto f = model_list.begin(); f!=model_list.end(); f++)
    {
        if(model_list[index].find(".png") != string::npos)
        {
            T = imread(model_list[index], 0);
            pt = countNonZero(T);
            T.convertTo(T, CV_32F);
            sum_T = sum(T)[0];
            float corr_single =  ((pt * sum(R.mul(T))[0]) - (sum_R * sum_T)) / (sqrt((pt * sum(R.mul(R))[0]) - pow(sum_R,2)) * sqrt((pt * sum(T.mul(T))[0]) - pow(sum_T,2)));
            cout << "Template " << index << " :" << corr_single << endl;
            log_file << "Template " << index << " :" << corr_single << "\n";
            corr += corr_single;
            template_count ++;
        }
        index++;
    }

    log_file << "Average correlation: " << corr/ template_count << "\n";
    log_file.close();
    return corr/ template_count;
}

int main(int, char**)
{

    Mat roi = imread(test_data_directory + test_file);
    Mat roi_gray;
    cvtColor(roi, roi_gray, CV_BGR2GRAY );

    vector<string> model_files;
    DIR *dpdf;
    struct dirent *epdf;
    dpdf = opendir(model_directory.c_str());
    if(dpdf!= NULL)
    {
        while (epdf = readdir(dpdf))
        {
            string file = epdf->d_name;
            // this condition avoids addition of "." and ".." pointers to the templates list
            if(file.length() > 4)
            {
                model_files.push_back(model_directory+ file);
            }
        }
    }
    closedir(dpdf);

    float score = correlation(roi_gray, model_files);
    cout << "Correlation score:" << score << endl;

    imshow("Original", roi);
    while(1)
    {
        if(waitKey(10) >= 0) break;
    }
    destroyAllWindows();
    return 0;
}
