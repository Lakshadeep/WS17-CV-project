#include "SupportVectorMachine.h"

SupportVectorMachine::SupportVectorMachine(vector<string> vehicleFiles, vector<string> noVehicleFiles)
{
    cars = vehicleFiles;
    noCars = noVehicleFiles;
}

SupportVectorMachine::~SupportVectorMachine()
{
    //dtor
}

int SupportVectorMachine::startSvm()
{
    map<int, Mat> trainData;
    trainData = SupportVectorMachine::createTrainData();
    cout<<"Train data vehicles:.."<<trainData.size()<<endl;

    SupportVectorMachine::extractHogFeatures(trainData);
    SupportVectorMachine::trainSVM(trainData);
    return trainData.size();
}

void SupportVectorMachine::isAlreadyTrained()
{

}

void SupportVectorMachine::extractHogFeatures(map<int, Mat> trainData)
{
    Mat img;
    HOGDescriptor hog;
    vector< float> descriptorsValues;
    vector< Point> locations;
    img = trainData[1];
    //namedWindow("Image", CV_WINDOW_AUTOSIZE );
    //imshow("Image", imgGray);
    //waitKey(0);
    hog.compute(img, descriptorsValues, Size(64, 64), Size(0, 0), locations);
    cout<<"Hog values:..."<<endl;
    cout<<descriptorsValues.size()<<endl;
    cout<<locations.size()<<endl;
}

void SupportVectorMachine::trainSVM(map<int, Mat> trainData)
{

    int numImages = 2;
    int imgArea = 64*64;
    Mat trainingDataMat(numImages,imgArea,CV_32FC1);

    int curCol = 0;
    for(int imIndex = 0; imIndex < numImages; imIndex++) {
        for (int rowIndex = 0; rowIndex < trainData[imIndex].rows; rowIndex++) {
            for (int colIndex = 0; colIndex < trainData[imIndex].cols; colIndex++) {
                trainingDataMat.at<float>(0,curCol++) = trainData[imIndex].at<uchar>(rowIndex,colIndex);
            }
        }
    }
    cout<<"svm mat size: "<<trainingDataMat.size()<<endl;
}

map<int, Mat> SupportVectorMachine::createTrainData()
{
    map<int, Mat> trainData;

    int key = 1;
    for(int index = 0; index < cars.size(); index++ )
    {
        if(cars[index].find(".png") != string::npos)
        {
            Mat img = imread(cars[index], 0);
            trainData[key] = img;
            key++;
        }
    }

    key = -1;
    for(int index = 0; index < noCars.size(); index++ )
    {
        if(noCars[index].find(".png") != string::npos)
        {
            Mat img = imread(noCars[index], 0);
            trainData[key] = img;
            key--;
        }
    }
    return trainData;
}
