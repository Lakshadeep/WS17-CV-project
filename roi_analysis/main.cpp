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

/// Generate grad_x and grad_y
Mat grad_x, grad_y;
Mat abs_grad_x, abs_grad_y;
Mat grad;

Mat frame;
Mat frame_blur;
Mat frame_gray;

int scale = 1;
int delta = 0;
int ddepth = CV_16S;
RNG rng(12345);

/// Directory/File paths
string test_data_directory = "/media/lakshadeep/Common/MAS/Semester2/computer_vision/project_final/test_data/";
string test_file = "0008.avi";
string model_directory = "/media/lakshadeep/Common/MAS/Semester2/computer_vision/project_final/models/distant/";
string experiment_results_directory = "/media/lakshadeep/Common/MAS/Semester2/computer_vision/project_final/experiment_results/roi/0008/";


int main(int, char**)
{
    VideoCapture cap(test_data_directory + test_file); // open the default camera
    if(!cap.isOpened())
    {
        std::cout << "Failed to open" << std::endl;
        return -1;
    }

    vector<string> files;
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
                files.push_back(model_directory+ file);
            }
        }
    }
    closedir(dpdf);

    namedWindow("Original", CV_WINDOW_AUTOSIZE );

    cap >> frame; // get a new frame from camera
    resize(frame, frame, Size(), 0.5, 0.5);

    int it_control = 0;
    int64 frame_no = 0;

    for(;;)
    {
        cap >> frame; // get a new frame from camera
        frame_no++;
        if(!frame.empty())
        {
            resize(frame, frame, Size(), 0.5, 0.5);
            GaussianBlur(frame, frame_blur, Size(3,3), 0, 0, BORDER_DEFAULT );
            cvtColor(frame_blur, frame_gray, CV_BGR2GRAY );

            Sobel(frame_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
            convertScaleAbs( grad_x, abs_grad_x );

            Sobel(frame_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
            convertScaleAbs( grad_y, abs_grad_y );

            /// Total Gradient (approximate)
            addWeighted( abs_grad_x, 0.3, abs_grad_y, 0.7, 0, grad );

            Mat thresholded_grad;
            threshold(grad, thresholded_grad, 50 ,255,THRESH_BINARY );

            //Canny(frame, grad, 100, 150, 3);

            Mat cropped;
            Mat region_proposal;
            int i = 0, j = 0;
            int window_size = 100;
            int step_size = 40;

            for(i = 0; i < thresholded_grad.rows - window_size ; )
            {
                for(j = 0; j < thresholded_grad.cols - window_size;  )
                {
                    Rect roi(j, i, window_size, window_size);
                    cropped =  thresholded_grad(roi);
                    Mat original_roi = frame(roi);

                    vector<Vec4i> lines;
                    HoughLinesP(cropped, lines, 1, CV_PI/180, 50, 20, 10 ); //50,20,10
                    int count_same = 0;
                    for( size_t i = 0; i < lines.size(); i++ )
                    {
                        //Vec4i l = lines[i];
                        double angle = lines[i][1];
                        //cout << angle << endl;
                        if(((85 < angle  &&  angle < 95) || (265 < angle  &&  angle < 275)) && (lines[i][0] > 20))
                        {
                            count_same++;
                        }

                    }
                    if(count_same > 1)
                    {
                        string name = to_string(frame_no) + to_string(i) + to_string(j) + string(".png");
                        imwrite(experiment_results_directory + name, original_roi);
                    }



                    j = j + step_size;
                }
                i = i + step_size;
            }

            imshow("Original", frame);
            if(waitKey(10) >= 0) break;
            it_control ++;
        }
    }
    destroyAllWindows();
    return 0;
}
