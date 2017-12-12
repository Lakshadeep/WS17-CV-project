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

Mat SupportVectorMachine::extractHogFeatures()
{
    Scalar mean, standardDeviation;
    HOGDescriptor hog;
    //hog.winSize.height = 64;
    vector< float> descriptors;
    vector< Point> locations;
    Mat testDataMat(1, 34020, CV_32FC1);
    int curCol = 0;

    resize(testImage, testImage, Size(128, 128));
    hog.compute(testImage, descriptors, Size(8, 8), Size(0, 0), locations);
    for (int desIndex = 0; desIndex < descriptors.size(); desIndex++) {
        testDataMat.at<float>(0,curCol++) = descriptors.at(desIndex);
    }

    meanStdDev ( testDataMat.row(0), mean, standardDeviation );
    for (int teMatIndex = 0; teMatIndex <= curCol; teMatIndex++) {
        testDataMat.at<float>(0,teMatIndex) -= mean[0];
        testDataMat.at<float>(0,teMatIndex) /= standardDeviation[0];
    }
    return testDataMat;
}

pair<Mat, Mat> SupportVectorMachine::extractHogFeatures(map<int, Mat> trainData)
{
    int numTrainImages = positiveExamples + negativeExamples;
    Mat img;
    Scalar mean, standardDeviation;
    HOGDescriptor hog;
    //hog.winSize.width = 128;
    cout<<"Pos: "<<positiveExamples<<" Neg: "<<negativeExamples<<endl;
    vector< float> descriptors;
    vector< Point> locations;

    Mat trainingDataMat(numTrainImages, 34020, CV_32FC1); //64260 size of hog features for 4x4
    Mat labelsMat(numTrainImages, 1, CV_32SC1);
    int curCol = 0, index = 0;

    for(int imIndex = -negativeExamples; imIndex < positiveExamples; imIndex++) {
        img = trainData[imIndex];
        resize(img, img, Size(128, 128));
        hog.compute(img, descriptors, Size(8, 8), Size(0, 0), locations);
        for (int desIndex = 0; desIndex < descriptors.size(); desIndex++) {
            trainingDataMat.at<float>(index,curCol++) = descriptors.at(desIndex);
        }

        meanStdDev ( trainingDataMat.row(index), mean, standardDeviation );
        for (int trMatIndex = 0; trMatIndex <= curCol; trMatIndex++) {
            trainingDataMat.at<float>(index,trMatIndex) -= mean[0];
            trainingDataMat.at<float>(index,trMatIndex) /= standardDeviation[0];
        }

        if (imIndex >= 0) {
            labelsMat.at<float>(index) = 1;
        }
        else {
            labelsMat.at<float>(index) = -1;
        }
        curCol = 0;
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
    svm->setNu(0.5);
    svm->setP(0.1);
    svm->setC(0.01);
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
            Mat img = imread(cars[index], 0);
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
            Mat img = imread(noCars[index], 0);
            trainData[key] = img;
            key--;
            negativeExamples++;
        }
    }
    return trainData;
}
