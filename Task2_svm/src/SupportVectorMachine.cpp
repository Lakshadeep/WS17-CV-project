#include "SupportVectorMachine.h"

SupportVectorMachine::SupportVectorMachine(vector<string> vehicleFiles, vector<string> noVehicleFiles)
{
    cars = vehicleFiles;
    noCars = noVehicleFiles;
}

// Constructor for test..
SupportVectorMachine::SupportVectorMachine()
{
    svmTest = svmTest->load("cars.yml");
}

SupportVectorMachine::~SupportVectorMachine()
{
    //dtor
}

pair<int, float> SupportVectorMachine::startSvm()
{
    pair<int, float> clfConfidence;
    map<int, Mat> trainData;
    pair<Mat, Mat> hogFeaturesLabels;

    trainData = SupportVectorMachine::createTrainData();
    cout<<"Total training data.."<<trainData.size()<<endl;
    cout<<"Positive Training data: "<<positiveExamples<<" Negative Training data: "<<negativeExamples<<endl;
    hogFeaturesLabels = SupportVectorMachine::extractHogFeatures(trainData);
    SupportVectorMachine::trainSVM(hogFeaturesLabels);
    clfConfidence.first = 0;
    clfConfidence.second = 0;
    return clfConfidence;
}

// method for test..
pair<int, float> SupportVectorMachine::startSvm(Mat image)
{
    pair<int, float> clfConfidence;
    Mat testFeatures;
    testImage = image;
    testFeatures = SupportVectorMachine::extractHogFeatures();
    clfConfidence = SVMpredict(testFeatures);
    return clfConfidence;
}

pair<int, float> SupportVectorMachine::SVMpredict(Mat testFeatures)
{
    pair<int, float> clfConfidence;
    clfConfidence.first = svmTest->predict(testFeatures);
    clfConfidence.second = svmTest->predict(testFeatures, noArray(), ml::StatModel::RAW_OUTPUT);
    clfConfidence.second = 1/(1 + exp(-clfConfidence.second));

    return clfConfidence;
}

Mat SupportVectorMachine::computeHog(Mat image) {

    Mat featuresMat, imageYCrCb;
    vector< float> descriptors, descriptorsYCrCb;

    cvtColor(image, imageYCrCb, CV_BGR2YCrCb);

    hog.compute(image, descriptors, Size(8, 8), Size(2, 2));
    transpose(Mat(descriptors).clone(), featuresMat);

    hog.compute(imageYCrCb, descriptorsYCrCb, Size(8, 8), Size(2, 2));
    transpose(Mat(descriptorsYCrCb).clone(), descriptorsYCrCb);
    hconcat(featuresMat, descriptorsYCrCb, featuresMat);

    return featuresMat;
}

Mat SupportVectorMachine::extractHogFeatures()
{
    Scalar mean, standardDeviation;
    hog.winSize.height = 64;

    resize(testImage, testImage, Size(64, 64));
    Mat featuresMat = SupportVectorMachine::computeHog(testImage);
    Mat testDataMat(1, featuresMat.cols, CV_32FC1);
    featuresMat.copyTo(testDataMat.row(0));

    meanStdDev ( testDataMat.row(0), mean, standardDeviation );
    for (int teMatIndex = 0; teMatIndex <= featuresMat.cols; teMatIndex++) {
        testDataMat.at<float>(0,teMatIndex) -= mean[0];
        testDataMat.at<float>(0,teMatIndex) /= standardDeviation[0];
    }

    return testDataMat;
}

pair<Mat, Mat> SupportVectorMachine::extractHogFeatures(map<int, Mat> trainData)
{
    int numTrainImages = positiveExamples + negativeExamples;
    Mat img, featuresMat;
    Scalar mean, standardDeviation;
    hog.winSize.height = 64;

    featuresMat = SupportVectorMachine::computeHog(trainData[0]);
    Mat trainingDataMat(numTrainImages, featuresMat.cols, CV_32FC1);
    Mat labelsMat(numTrainImages, 1, CV_32SC1);
    int index = 0;

    for(int imIndex = -negativeExamples; imIndex < positiveExamples; imIndex++) {

        featuresMat = SupportVectorMachine::computeHog(trainData[imIndex]);
        featuresMat.copyTo(trainingDataMat.row(index));
        meanStdDev ( trainingDataMat.row(index), mean, standardDeviation );
        for (int trMatIndex = 0; trMatIndex <= featuresMat.cols; trMatIndex++) {
            trainingDataMat.at<float>(index,trMatIndex) -= mean[0];
            trainingDataMat.at<float>(index,trMatIndex) /= standardDeviation[0];
        }

        if (imIndex >= 0) {
            labelsMat.at<float>(index) = 1;
        }
        else {
            labelsMat.at<float>(index) = -1;
        }
        index++;
    }

    pair<Mat, Mat> hogFeaturesLabels;
    hogFeaturesLabels.first.push_back(trainingDataMat);
    hogFeaturesLabels.second.push_back(labelsMat);
    return hogFeaturesLabels;
}

void SupportVectorMachine::trainSVM(pair<Mat, Mat> svmData)
{

    Ptr<SVM> svm = SVM::create();
    //svm->setCoef0(0.0);
    //svm->setDegree(2);
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-6 ));
    //svm->setGamma(0);
    //svm->setNu(0.5);
    //svm->setP(0.1);
    //svm->setC(0.01);
    //svm->trainAuto(svmData.first, ROW_SAMPLE, svmData.second);
    svm->train(svmData.first, ROW_SAMPLE, svmData.second);
    svm->save("cars.yml");
    cout<<"SVM has been trained..."<<endl;
}

map<int, Mat> SupportVectorMachine::createTrainData()
{
    map<int, Mat> trainData;

    int key = 0;
    for(int index = 0; index < cars.size(); index++ )
    {
        if(cars[index].find(".png") != string::npos)
        {
            Mat img = imread(cars[index], 1);
            trainData[key] = img;
            key++;
            positiveExamples++;
        }
    }

    key = -1;
    for(int index = 0; index < noCars.size(); index++ )
    {
        if(noCars[index].find(".png") != string::npos)
        {
            Mat img = imread(noCars[index], 1);
            trainData[key] = img;
            key--;
            negativeExamples++;
        }
    }
    return trainData;
}
