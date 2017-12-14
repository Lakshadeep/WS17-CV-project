#include "SupportVectorMachine.h"

SupportVectorMachine::SupportVectorMachine(vector<string> vehicleFiles, vector<string> noVehicleFiles)
{
    cars = vehicleFiles;
    noCars = noVehicleFiles;
}

// Constructor for test..
SupportVectorMachine::SupportVectorMachine()
{
}

SupportVectorMachine::~SupportVectorMachine()
{
    //dtor
}

void SupportVectorMachine::startSvm()
{
    map<int, Mat> trainData;

    trainData = SupportVectorMachine::createTrainData();
    cout<<"Total training data.."<<trainData.size()<<endl;
    Size w_size = trainData[0].size();
    SupportVectorMachine::extractHogFeatures(trainData, w_size);
    SupportVectorMachine::trainSVM();
}


void SupportVectorMachine::extractHogFeatures(map<int, Mat> trainData, const Size wsize)
{

    int numTrainImages = positiveExamples + negativeExamples;
    Mat img, gray;
    hog.winSize = wsize;

    cout<<"Pos: "<<positiveExamples<<" Neg: "<<negativeExamples<<endl;
    vector< float> descriptors;

    int curCol = 0, index = 0;

    for(int imIndex = -negativeExamples; imIndex < positiveExamples; imIndex++) {
        img = trainData[imIndex];

        Rect r = Rect(( img.cols - wsize.width ) / 2,( img.rows - wsize.height ) / 2, wsize.width, wsize.height);
        cvtColor( trainData[imIndex](r), gray, COLOR_BGR2GRAY );
        hog.compute( gray, descriptors, Size( 8, 8 ), Size( 0, 0 ));
        gradient_lst.push_back(Mat(descriptors).clone());

        curCol = 0;
        index++;
    }
}

void SupportVectorMachine::convert_data( const vector< Mat > & train_samples, Mat& trainData )
{

    const int rows = (int)train_samples.size();
    const int cols = (int)std::max( train_samples[0].cols, train_samples[0].rows );
    Mat tmp( 1, cols, CV_32FC1 ); //< used for transposition if needed
    trainData = Mat( rows, cols, CV_32FC1 );

    for( size_t i = 0 ; i < train_samples.size(); ++i )
    {

        if( train_samples[i].cols == 1 )
        {
            transpose( train_samples[i], tmp );
            tmp.copyTo( trainData.row( (int)i ) );
        }
        else if( train_samples[i].rows == 1 )
        {
            train_samples[i].copyTo( trainData.row( (int)i ) );
        }
    }
}

void SupportVectorMachine::trainSVM()
{
    Mat train_data;
    convert_data( gradient_lst, train_data );
    Ptr<SVM> svm = SVM::create();
    svm->setCoef0( 0.0 );
    svm->setDegree( 3 );
    svm->setTermCriteria( TermCriteria( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-3 ) );
    svm->setGamma( 0 );
    svm->setKernel( SVM::LINEAR );
    svm->setNu( 0.5 );
    svm->setP( 0.1 ); // for EPSILON_SVR, epsilon in loss function?
    svm->setC( 0.01 ); // From paper, soft classifier
    svm->setType( SVM::EPS_SVR ); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
    svm->train(train_data, ROW_SAMPLE, labels);

    hog.setSVMDetector( get_svm_detector( svm ) );
    hog.winSize = Size(64,64);

    hog.save("cars.yml");
    cout<<"SVM has been trained..."<<endl;
}

vector< float > SupportVectorMachine::get_svm_detector( const Ptr< SVM >& svm )
{
    // get the support vectors
    Mat sv = svm->getSupportVectors();
    const int sv_total = sv.rows;
    // get the decision function
    Mat alpha, svidx;
    double rho = svm->getDecisionFunction( 0, alpha, svidx );

    CV_Assert( alpha.total() == 1 && svidx.total() == 1 && sv_total == 1 );
    CV_Assert( (alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
               (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f) );
    CV_Assert( sv.type() == CV_32F );

    vector< float > hog_detector( sv.cols + 1 );
    memcpy( &hog_detector[0], sv.ptr(), sv.cols*sizeof( hog_detector[0] ) );
    hog_detector[sv.cols] = (float)-rho;
    return hog_detector;
}

map<int, Mat> SupportVectorMachine::createTrainData()
{
    map<int, Mat> trainData;
    labels.clear();

    int key = 0;
    for(int index = 0; index < cars.size(); index++ )
    {
        if(cars[index].find(".png") != string::npos)
        {
            Mat img = imread(cars[index]);
            trainData[key] = img;
            key++;
            positiveExamples++;
        }
    }
    labels.assign( positiveExamples, +1);

    key = -1;
    for(int index = 0; index < noCars.size(); index++ )
    {
        if(noCars[index].find(".png") != string::npos)
        {
            Mat img = imread(noCars[index]);
            trainData[key] = img;
            key--;
            negativeExamples++;
        }
    }
    labels.insert( labels.end(), negativeExamples, -1 );
    return trainData;
}
