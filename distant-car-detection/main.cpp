/*********************************** NOTE ********************************************/
/// This program detects the distant cars using classical computer vision approach

/// To run this program you need to install segmentation module from
/// ximgproc which is currently provided with opencv_contrib repository.
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

using namespace cv;
using namespace std;
using namespace cv::ximgproc::segmentation;

Scalar getMSSIM( const Mat& i1, const Mat& i2);

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
string test_file = "0009.avi";
string model_directory = "/media/lakshadeep/Common/MAS/Semester2/computer_vision/project_final/models/distant/";
string experiment_results_directory = "/media/lakshadeep/Common/MAS/Semester2/computer_vision/project_final/experiment_results/";


float correlation(Mat reg_prop, vector<string> model_list)
{
    float corr = 0.0, pt = 0, sum_T= 0, sum_R = 0, template_count = 0;
    Scalar ssim=0;
    Mat T,R;
    int index = 0;
    T = imread(model_list[0], 0);
    resize(reg_prop, R, Size(T.size()), 0, 0);
    R.convertTo(R, CV_32F);
    sum_R = sum(R)[0];

    for(auto f = model_list.begin(); f!=model_list.end(); f++)
    {
        if(model_list[index].find(".png") != string::npos)
        {
            T = imread(model_list[index], 1);

            /// uncomment to apply pixelwise correlation
            /**
            pt = countNonZero(T);
            T.convertTo(T, CV_32F);
            sum_T = sum(T)[0];
            corr += ((pt * sum(R.mul(T))[0]) - (sum_R * sum_T)) / (sqrt((pt * sum(R.mul(R))[0]) - pow(sum_R,2)) * sqrt((pt * sum(T.mul(T))[0]) - pow(sum_T,2)));
            **/

            /// uncomment to use SSIM
            ssim += getMSSIM(R, T);

            template_count ++;
        }
        index++;
    }

    // return corr;                            // for pixelwise correlation
    return sum(ssim)[0]/ (template_count*3);   // for ssim
}

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
                files.push_back(model_directory + file);
            }
        }
    }
    closedir(dpdf);

    namedWindow("Original", CV_WINDOW_AUTOSIZE );
    namedWindow("Thresholded", CV_WINDOW_AUTOSIZE );

    cap >> frame; // get a new frame from camera
    resize(frame, frame, Size(), 0.5, 0.5);
    VideoWriter video("distant_car_detection_0008_1.avi",CV_FOURCC('M','J','P','G'),10, Size(frame.cols,frame.rows),true);
    int it_control = 0;

    for(;;)
    {
        cap >> frame; // get a new frame from camera
        if(!frame.empty())
        {
            resize(frame, frame, Size(), 0.5, 0.5);
            GaussianBlur(frame, frame_blur, Size(3,3), 0, 0, BORDER_DEFAULT );
            cvtColor(frame_blur, frame_gray, CV_BGR2GRAY );

            Sobel(frame_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
            convertScaleAbs( grad_x, abs_grad_x );

            Sobel(frame_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
            convertScaleAbs( grad_y, abs_grad_y );

            // Total Gradient (approximate)
            addWeighted( abs_grad_x, 0.3, abs_grad_y, 0.7, 0, grad );

            Mat thresholded_grad;
            threshold(grad, thresholded_grad, 50 ,255,THRESH_BINARY );

            Mat cropped;
            Mat region_proposal;
            int i = 0, j = 0;
            int window_size = 100;  /// set sliding window size here
            int step_size = 40;     /// set sliding window step size here

            for(i = 0; i < thresholded_grad.rows - window_size ; )
            {
                for(j = 0; j < thresholded_grad.cols - window_size;  )
                {
                    Rect roi(j, i, window_size, window_size);
                    cropped =  thresholded_grad(roi);

                    vector<Vec4i> lines;
                    HoughLinesP(cropped, lines, 1, CV_PI/180, 50, 20, 10 ); //50,20,10
                    int count_same = 0;
                    for( size_t i = 0; i < lines.size(); i++ )
                    {
                        double angle = lines[i][1];

                        // we look for horizontal parallel lines here
                        if(((85 < angle  &&  angle < 95) || (265 < angle  &&  angle < 275)) && (lines[i][0] > 20))
                        {
                            count_same++;
                        }
                        // uncomment for visualising the lines
                        // line( frame, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
                    }

                    if(count_same > 1)
                    {
                        cv::rectangle(frame,cv::Point(j, i),cv::Point(j + window_size, i + window_size),cv::Scalar(255, 0, 0));
                        Mat im_gray;
                        im_gray = frame(roi);// frame_gray
                        vector<float> scores_collect;
                        float score = -2;
                        Ptr<SelectiveSearchSegmentation> ss = createSelectiveSearchSegmentation();
                        ss->setBaseImage(frame(roi));
                        ss->switchToSelectiveSearchFast();
                        vector<Rect> rects;
                        ss->process(rects);

                        vector<Rect> squares;
                        int num_squares = 0;
                        float rect_ratio;
                        for(int rec= 0; rec!= rects.size()-1; rec++) {
                            rect_ratio = (float)rects.at(rec).width/(float)rects.at(rec).height;
                            if(rect_ratio > 0.9 and rect_ratio < 1.1){
                                squares.push_back(rects.at(rec));
                            }
                        }

                        int x,y, h, w;

                        for(int sq= 0; sq!= squares.size()-1; sq++)
                        {
                            Rect current_square = squares.at(sq);
                            score = correlation(im_gray(current_square), files);

                            x = current_square.x;
                            y = current_square.y;
                            h = current_square.height;
                            w = current_square.width;

                            if(score> 0.2)
                            {
                                score = round(score*1000);
                                score = score/1000;
                                string score_string = to_string(score);
                                score_string.erase(score_string.find_last_not_of('0') + 1, string::npos );
                                cv::rectangle(frame,cv::Point(y + j, x + i),cv::Point(y + j + h, x + i + w),cv::Scalar(0, 0, 255));
                                putText(frame, score_string, cv::Point(y + j + h, x + i + w), FONT_HERSHEY_PLAIN, 1,cv::Scalar(0, 0, 255), 2);
                            }
                        }
                    }
                    j = j + step_size;
                }
                i = i + step_size;
            }

            video.write(frame);

            imshow("Original", frame);

            if(waitKey(10) >= 0) break;
            it_control ++;
        }
    }
    destroyAllWindows();
    return 0;
}
