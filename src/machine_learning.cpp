#include <opencv2/opencv.hpp>
#include "opencv2/ximgproc.hpp"
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <math.h>
#include <Eigen/Dense>

#include <miscellaneous.h>

using namespace cv;
using namespace cv::ml;
using namespace std;
using namespace Eigen;
using namespace ximgproc;

void trainSVM(Mat img, Mat mask, string fname){
	Mat imageR, maskR;
	resize(img.clone(), imageR, Size(512,512), 0, 0, INTER_LINEAR);
	resize(mask.clone(), maskR, Size(512,512), 0, 0, INTER_LINEAR);

	Mat labMat;
	cvtColor(imageR, labMat, cv::COLOR_BGR2Lab);
	vector<Mat> lab;
	split(labMat, lab);

	vector<int> labels_vec;
	vector<float> l_vec, a_vec, B_vec;
	for(unsigned int i=0; i<512; i++){
		for(unsigned int j=0; j<512; j++){
			labels_vec.push_back(maskR.at<uchar>(i,j));
			l_vec.push_back(lab[0].at<uchar>(i,j));
			a_vec.push_back(lab[1].at<uchar>(i,j));
			B_vec.push_back(lab[2].at<uchar>(i,j));
		}
	}

	int n = labels_vec.size();
	int labels_arr[n];
	float training_arr[n][3];
	for(int i=0; i<n; i++){
		labels_arr[i] = labels_vec[i];
		training_arr[i][0] = l_vec[i];
		training_arr[i][1] = a_vec[i];
		training_arr[i][2] = B_vec[i];
	}

	Mat trainingMat(n, 3, CV_32F, training_arr);
	Mat labelsMat(n, 1, CV_32SC1, labels_arr);

	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 500, 1e-16));
	cout << "Training SVM..." << endl;
	svm->train(trainingMat, ROW_SAMPLE, labelsMat);
	svm->save(fname);
}

Mat predictSVM(Mat img,string fname){
	Mat imageR;
	resize(img, imageR, Size(512*2,512*2), 0, 0, INTER_LINEAR);
	cout << "Loading..." << endl;
	Ptr<SVM> svm = SVM::load(fname);

	Mat labMat;
	cvtColor(imageR, labMat, cv::COLOR_BGR2Lab);
	vector<Mat> lab;
	split(labMat, lab);

	cout << "Predicting..." << endl;
	Mat response = Mat::zeros(imageR.size(),0);
	for(unsigned int i=0; i<imageR.rows; i++){
		for(unsigned int j=0; j<imageR.cols; j++){
			float arr[3] = {lab[0].at<uchar>(i,j),lab[1].at<uchar>(i,j),lab[2].at<uchar>(i,j)};
			Mat arrM(1,3,CV_32F,arr);
		    float back = svm->predict(arrM);
		    response.at<uchar>(i,j) = back*255;
		}
	}
	Mat output;
	resize(response, output, img.size(), 0, 0, INTER_LINEAR);
	return(output);
}

void trainBC(Mat img, Mat mask, string fname){
	Mat imageR, maskR;
	resize(img.clone(), imageR, Size(512,512), 0, 0, INTER_LINEAR);
	resize(mask.clone(), maskR, Size(512,512), 0, 0, INTER_LINEAR);

	Mat labMat;
	cvtColor(imageR, labMat, cv::COLOR_BGR2Lab);
	vector<Mat> lab;
	split(labMat, lab);

	vector<int> labels_vec;
	vector<float> l_vec, a_vec, B_vec;
	for(unsigned int i=0; i<512; i++){
		for(unsigned int j=0; j<512; j++){
			labels_vec.push_back(maskR.at<uchar>(i,j));
			l_vec.push_back(lab[0].at<uchar>(i,j));
			a_vec.push_back(lab[1].at<uchar>(i,j));
			B_vec.push_back(lab[2].at<uchar>(i,j));
		}
	}

	int n = labels_vec.size();
	int labels_arr[n];
	float training_arr[n][3];
	for(int i=0; i<n; i++){
		labels_arr[i] = labels_vec[i];
		training_arr[i][0] = l_vec[i];
		training_arr[i][1] = a_vec[i];
		training_arr[i][2] = B_vec[i];
	}

	Mat trainingMat(n, 3, CV_32F, training_arr);
	Mat labelsMat(n, 1, CV_32SC1, labels_arr);

	Ptr<NormalBayesClassifier> classifier = NormalBayesClassifier::create();
	classifier->train(trainingMat, ROW_SAMPLE, labelsMat);
	Ptr<TrainData> tdat = TrainData::create(trainingMat,ROW_SAMPLE,labelsMat);
	classifier->save(fname);
}

Mat predictBC(Mat img,string fname){
	Mat imageR;
	resize(img.clone(), imageR, Size(512*2,512*2), 0, 0, INTER_LINEAR);

	cout << "Loading..." << endl;
	Ptr<NormalBayesClassifier> classifier = NormalBayesClassifier::load(fname);

	Mat labMat;
	cvtColor(imageR, labMat, cv::COLOR_BGR2Lab);
	vector<Mat> lab;
	split(labMat, lab);

	cout << "Predicting..." << endl;
	Mat response = Mat::zeros(imageR.size(),0);
	for(unsigned int i=0; i<imageR.rows; i++){
		for(unsigned int j=0; j<imageR.cols; j++){
			float arr[3] = {lab[0].at<uchar>(i,j),lab[1].at<uchar>(i,j),lab[2].at<uchar>(i,j)};
			Mat arrM(1,3,CV_32F,arr);
		    float back = classifier->predict(arrM);
		    response.at<uchar>(i,j) = back;
		}
	}
	Mat output;
	resize(response, output, img.size(), 0, 0, INTER_LINEAR);
	return(output);
}

void trainBoost(Mat img, Mat mask, string fname){
	Mat imageR, maskR;
	resize(img.clone(), imageR, Size(512,512), 0, 0, INTER_LINEAR);
	resize(mask.clone(), maskR, Size(512,512), 0, 0, INTER_LINEAR);

	Mat labMat;
	cvtColor(imageR, labMat, cv::COLOR_BGR2Lab);
	vector<Mat> lab;
	split(labMat, lab);

	vector<int> labels_vec;
	vector<float> l_vec, a_vec, B_vec;
	for(unsigned int i=0; i<512; i++){
		for(unsigned int j=0; j<512; j++){
			labels_vec.push_back(maskR.at<uchar>(i,j));
			l_vec.push_back(lab[0].at<uchar>(i,j));
			a_vec.push_back(lab[1].at<uchar>(i,j));
			B_vec.push_back(lab[2].at<uchar>(i,j));
		}
	}

	int n = labels_vec.size();
	int labels_arr[n];
	float training_arr[n][3];
	for(int i=0; i<n; i++){
		labels_arr[i] = labels_vec[i];
		training_arr[i][0] = l_vec[i];
		training_arr[i][1] = a_vec[i];
		training_arr[i][2] = B_vec[i];
	}

	Mat trainingMat(n, 3, CV_32F, training_arr);
	Mat labelsMat(n, 1, CV_32SC1, labels_arr);

	Ptr<Boost> boost = Boost::create();
	boost->setBoostType(Boost::GENTLE);
	boost->setWeakCount(1000);

	cout << "Training Boost..." << endl;
	boost->train(trainingMat, ROW_SAMPLE, labelsMat);
	boost->save(fname);
}

Mat predictBoost(Mat img,string fname){
	Mat imageR;
	resize(img.clone(), imageR, Size(512*2,512*2), 0, 0, INTER_LINEAR);

	cout << "Loading..." << endl;
	Ptr<Boost> classifier = Boost::load(fname);

	Mat labMat;
	cvtColor(imageR, labMat, cv::COLOR_BGR2Lab);
	vector<Mat> lab;
	split(labMat, lab);

	cout << "Predicting..." << endl;
	Mat response = Mat::zeros(imageR.size(),0);
	for(unsigned int i=0; i<imageR.rows; i++){
		for(unsigned int j=0; j<imageR.cols; j++){
			float arr[3] = {lab[0].at<uchar>(i,j),lab[1].at<uchar>(i,j),lab[2].at<uchar>(i,j)};
			Mat arrM(1,3,CV_32F,arr);
		    float back = classifier->predict(arrM);
		    cout << back << endl;
		    response.at<uchar>(i,j) = back;
		}
	}
	Mat output;
	resize(response, output, img.size(), 0, 0, INTER_LINEAR);
	return(output);
}
