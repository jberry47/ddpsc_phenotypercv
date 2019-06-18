#include <opencv2/opencv.hpp>
#include "opencv2/ximgproc.hpp"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <math.h>
#include <Eigen/Dense>

using namespace cv;
using namespace std;
using namespace Eigen;
using namespace ximgproc;

float extractRGB_chips(Mat img,Mat &mask){
	//-- Averages the histogram for a given channel
	Mat img1 = img;
    Mat hist;
	int dims = 1;
	int histSize = 255;
	float hranges[] = { 0, 255 };
	const float *ranges = {hranges};

	calcHist(&img1,1,0,mask,hist, dims, &histSize, &ranges ,true ,false);

	int sum=0;
	for(int i = 0;i<255;i++){
		sum += hist.at<float>(i,0);
	}
	Mat weights = hist/sum;
	float hist_avg=0.0;
	for(int i = 0;i<255;i++){
		hist_avg += i*weights.at<float>(i,0);
	}
	return hist_avg;
}

MatrixXd getRGBarray(Mat img){
	//-- Loops over chips and gets RGB values of each one
	MatrixXd sourceColors(22,3);
	vector<Mat> bgr;
	split(img, bgr);
	Mat b = bgr[0];
	Mat g = bgr[1];
	Mat r = bgr[2];
	for(unsigned int i=1;i<23;i++){
	    	stringstream ss;
	    	ss << i;
	    	string str = ss.str();
	    	string file_name = "card_masks/"+str+"_mask.png";
	    	Mat mask = imread(file_name,0);
	    	Mat cc;
	   		threshold(mask,cc,90,255,THRESH_BINARY);
	   		float b_avg = extractRGB_chips(b, cc);
	   	    float g_avg = extractRGB_chips(g, cc);
	   	    float r_avg = extractRGB_chips(r, cc);
	   	    sourceColors(i-1,0) = b_avg;
	   	    sourceColors(i-1,1) = g_avg;
	   	    sourceColors(i-1,2) = r_avg;
	}
    return(sourceColors);
}

void get_standardizations(Mat img, float &det, MatrixXd &rh,MatrixXd &gh,MatrixXd &bh){
	//-- Extending source RGB chips to squared and cubic terms
	MatrixXd source1, source2, source3;
	source1 = getRGBarray(img);
	source2 = (source1.array() * source1.array()).matrix();
	source3 = (source2.array() * source1.array()).matrix();
	MatrixXd source(source1.rows(),source1.cols()+source2.cols()+source3.cols());
	source << source1, source2, source3;

	//-- Computing Moore-Penrose Inverse
	MatrixXd M = (source.transpose()*source).inverse()*source.transpose();

	//-- Reading target homography
	MatrixXd target(22,3);
	fstream file;
	file.open("target_homography.csv");
	string value;
	int rowCounter = 0;
	while ( getline ( file, value) )
	{
	     vector<float> result;
	     stringstream substr(value);
	     string item;
	     while (getline(substr, item, ',')) {
		     const char *cstr = item.c_str();
		     char* pend;
		     float num = strtof(cstr,&pend);
	         result.push_back(num);
	     }
	     target(rowCounter,0) = result[0];
	     target(rowCounter,1) = result[1];
	     target(rowCounter,2) = result[2];
	     rowCounter++;
	}

	//-- Computing linear target RGB standardizations
	rh = M*target.col(2);
	gh = M*target.col(1);
	bh = M*target.col(0);

	//-- Extending target RGB chips to squared and cubic terms
	MatrixXd target1, target2, target3;
	target2 = (target.array() * target.array()).matrix();
	target3 = (target2.array() * target.array()).matrix();

	//-- Computing square and cubic target RGB standardizations
	MatrixXd r2h,g2h,b2h,r3h,g3h,b3h;
	r2h = M*target2.col(2);
	g2h = M*target2.col(1);
	b2h = M*target2.col(0);
	r3h = M*target3.col(2);
	g3h = M*target3.col(1);
	b3h = M*target3.col(0);

	//-- Computing D
	MatrixXd H(9,9);
	H << rh.col(0),gh.col(0),bh.col(0),r2h.col(0),g2h.col(0),b2h.col(0),r3h.col(0),g3h.col(0),b3h.col(0);
	det = H.transpose().determinant();
}

Mat color_homography(Mat img, MatrixXd r_coef,MatrixXd g_coef,MatrixXd b_coef){
	Mat b, g, r, b2, g2, r2, b3, g3, r3;
	vector<Mat> bgr(3);
	split(img,bgr);

	//-- Computing linear, squared, and cubed images
	b = bgr[0];
	g = bgr[1];
	r = bgr[2];
	b2 = b.mul(b);
	g2 = g.mul(g);
	r2 = r.mul(r);
	b3 = b2.mul(b);
	g3 = g2.mul(g);
	r3 = r2.mul(r);

	//-- Computing homography
	b = 0+r*b_coef(0,0)+g*b_coef(1,0)+b*b_coef(2,0)+r2*b_coef(3,0)+g2*b_coef(4,0)+b2*b_coef(5,0)+r3*b_coef(6,0)+g3*b_coef(7,0)+b3*b_coef(8,0);
	g = 0+r*g_coef(0,0)+g*g_coef(1,0)+b*g_coef(2,0)+r2*g_coef(3,0)+g2*g_coef(4,0)+b2*g_coef(5,0)+r3*g_coef(6,0)+g3*g_coef(7,0)+b3*g_coef(8,0);
	r = 0+r*r_coef(0,0)+g*r_coef(1,0)+b*r_coef(2,0)+r2*r_coef(3,0)+g2*r_coef(4,0)+b2*r_coef(5,0)+r3*r_coef(6,0)+g3*r_coef(7,0)+b3*r_coef(8,0);

	//-- Combining channels and returning
	bgr[0] = b;
	bgr[1] = g;
	bgr[2] = r;
	Mat adjImage;
	merge(bgr,adjImage);
	return adjImage;
}
