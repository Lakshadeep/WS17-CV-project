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
    vector< float> descriptors;
    vector< Point> locations;
    img = trainData[1];
    //namedWindow("Image", CV_WINDOW_AUTOSIZE );
    //imshow("Image", imgGray);
    //waitKey(0);
    hog.compute(img, descriptors, Size(64, 64), Size(0, 0), locations);
    cout<<"Hog values:..."<<endl;
    cout<<descriptors.size()<<endl;
    cout<<locations.size()<<endl;
}

void SupportVectorMachine::trainSVM(map<int, Mat> trainData)
{

    Mat trainingDataMat(4, trainData[0].rows * trainData[0].cols, CV_32FC1);
    Mat labelsMat(4, 1, CV_32SC1);

    int curCol = 0, index = 0;
    for(int imIndex = -2; imIndex < 2; imIndex++) {
        for (int rowIndex = 0; rowIndex < trainData[imIndex].rows; rowIndex++) {
            for (int colIndex = 0; colIndex < trainData[imIndex].cols; colIndex++) {
                trainingDataMat.at<float>(index,curCol++) = trainData[imIndex].at<uchar>(rowIndex,colIndex);
                }
            }
        if (imIndex >= 0) {
            labelsMat.at<float>(index, 0) = 1;
        }
        else {
            labelsMat.at<float>(index, 0) = -1;
        }
        curCol = 0;
        index++;
    }

    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);
    svm->save("cars.yml");
}

map<int, Mat> SupportVectorMachine::createTrainData()
{
    map<int, Mat> trainData;

    int key = 0;
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
