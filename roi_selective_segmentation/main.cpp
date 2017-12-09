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

using namespace cv;
using namespace std;
using namespace cv::ximgproc::segmentation;


/// Directory/File paths
string test_data_directory = "/media/lakshadeep/Common/MAS/Semester2/computer_vision/project_final/test_data/";
string test_file = "roi/good/11140320.png";
string model_directory = "/media/lakshadeep/Common/MAS/Semester2/computer_vision/project_final/models/distant/";
string experiment_results_directory = "/media/lakshadeep/Common/MAS/Semester2/computer_vision/project_final/experiment_results/roi/ss/good/11140320/";

int main(int, char**)
{

    Mat roi = imread(test_data_directory + test_file);
    vector<float> scores_collect;
    float score = -2;
    Ptr<SelectiveSearchSegmentation> ss = createSelectiveSearchSegmentation();
    ss->setBaseImage(roi);
    ss->switchToSelectiveSearchFast();
    vector<Rect> rects;
    ss->process(rects);
    cout << "No of identified ROI by ss:" << rects.size() << endl;

    int x,y,h,w;
    for(int i = 0; i < rects.size(); i++)
    {
        Rect current_rect = rects.at(i);
        x = current_rect.x;
        y = current_rect.y;
        h = current_rect.height;
        w = current_rect.width;
        Rect region(y, x, h, w);
        Mat cropped =  roi(region);
        string name = to_string(i) + string(".png");
        imwrite(experiment_results_directory + name, cropped);
        //cv::rectangle(roi,cv::Point(y,x),cv::Point(y + h, x + w),cv::Scalar(0, 0, 255));
    }
    imshow("Original", roi);
    while(1)
    {
        if(waitKey(10) >= 0) break;
    }
    destroyAllWindows();
    return 0;
}
