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
string experiment_results_directory = "/media/lakshadeep/Common/MAS/Semester2/computer_vision/project_final/experiment_results/mssim/36040160/";


// from opencv documentation...
Scalar getMSSIM( const Mat& i1, const Mat& i2)
{
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d = CV_32F;
    Mat I1, I2;
    i1.convertTo(I1, d);            // cannot calculate on one byte large values
    i2.convertTo(I2, d);
    Mat I2_2   = I2.mul(I2);        // I2^2
    Mat I1_2   = I1.mul(I1);        // I1^2
    Mat I1_I2  = I1.mul(I2);        // I1 * I2
    /*************************** END INITS **********************************/
    Mat mu1, mu2;                   // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);
    Mat mu1_2   =   mu1.mul(mu1);
    Mat mu2_2   =   mu2.mul(mu2);
    Mat mu1_mu2 =   mu1.mul(mu2);
    Mat sigma1_2, sigma2_2, sigma12;
    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;
    Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);                 // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);                 // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    Mat ssim_map;
    divide(t3, t1, ssim_map);        // ssim_map =  t3./t1;
    Scalar mssim = mean(ssim_map);   // mssim = average of ssim map
    return mssim;
}

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
    log_file.open(experiment_results_directory + test_file + string(".txt"));

    for(auto f = model_list.begin(); f!=model_list.end(); f++)
    {
        if(model_list[index].find(".png") != string::npos)
        {
            T = imread(model_list[index], 0);
            Scalar ssim_single = getMSSIM(R, T);
            cout << "Template " << index << " :" << ssim_single << endl;
            log_file << "Template " << index << " :" << ssim_single << "\n";
            ssim += ssim_single;
            template_count ++;
        }
        index++;
    }

    log_file << "Average: " << sum(ssim)[0]/ (template_count*3) << "\n";
    log_file.close();
    return sum(ssim)[0]/ (template_count*3);
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
