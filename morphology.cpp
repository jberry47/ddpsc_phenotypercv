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

Mat find_endpoints(Mat input){
    Mat out = Mat::zeros(input.size(),input.type());
    Mat kernel1 = (Mat_<int>(3, 3) <<
           -1, -1, -1,
           -1, 1, -1,
           0, 1, 0);

    Mat kernel2 = (Mat_<int>(3, 3) <<
           -1, -1, -1,
           -1, 1, 0,
           -1, 0, 1);

    Mat temp_ker;

    Mat cur_ker = kernel1;
    for(int i = 1; i <=4; i++){
        rotate(cur_ker,temp_ker,ROTATE_90_CLOCKWISE);
        Mat output_image;
        morphologyEx(input, output_image, MORPH_HITMISS, temp_ker);
        bitwise_or(out, output_image,out);
        cur_ker = temp_ker;
    }

    cur_ker = kernel2;
    for(int i = 1; i <=4; i++){
        rotate(cur_ker,temp_ker,ROTATE_90_CLOCKWISE);
        Mat output_image;
        morphologyEx(input, output_image, MORPH_HITMISS, temp_ker);
        bitwise_or(out, output_image,out);
        cur_ker = temp_ker;
    }
    return(out);
}

Mat find_branchpoints(Mat input){
    Mat out = Mat::zeros(input.size(),input.type());
	vector<Mat> kernels(5);
    kernels[0] = (Mat_<int>(3, 3) <<
           -1, 1, -1,
           1, 1, 1,
           -1, -1, -1);
    kernels[1] = (Mat_<int>(3, 3) <<
           1, -1, 1,
           -1, 1, -1,
           1, -1, -1);
    kernels[2] = (Mat_<int>(3, 3) <<
               1, -1, 1,
               0, 1, 0,
               0, 1, 0);
    kernels[3] = (Mat_<int>(3, 3) <<
               -1, 1, -1,
               1, 1, 0,
               -1, 0, 1);
    kernels[4] = (Mat_<int>(3, 3) <<
                   -1, 1, -1,
                   1, 1, 1,
                   -1, 1, -1);

    Mat cur_ker, temp_ker;
    for(int ker=0;ker<=4;ker++){
    	cur_ker = kernels[ker];
    	for(int i = 1; i <=4; i++){
   	        rotate(cur_ker,temp_ker,ROTATE_90_CLOCKWISE);
   	        Mat output_image;
   	        morphologyEx(input, output_image, MORPH_HITMISS, temp_ker);
   	        bitwise_or(out, output_image,out);
   	        cur_ker = temp_ker;
   	    }
    }
	return(out);
}

Mat length_filter(Mat input, int size){
	//-- Get contours of mask
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
    findContours( input, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

	Mat out = Mat::zeros(input.size(),input.type());
	Mat temp, temp_mask;
	double area;
    //-- Keep only those contours that have a length > size
    vector<Point> cc;
    for(unsigned int i=0; i < contours.size(); i++){
    	temp = Mat::zeros(input.size(),input.type());
    	temp_mask = Mat::zeros(input.size(),input.type());
    	drawContours(temp, contours, i, 255, cv::FILLED);
    	bitwise_and(temp,input,temp_mask);
    	area = sum(temp_mask/255)[0];
    	if(area > size){
    		drawContours(out, contours, i, 255, cv::FILLED);
    	}
    }
	Mat kept_mask;
	bitwise_and(input,out,kept_mask);
	return(kept_mask);
}

Mat prune(Mat input, int size){
	Mat out = input.clone();
	Mat endpoints;
	for(int i=0; i < size; i++){
		endpoints = find_endpoints(out);
		out = out-endpoints;
	}
	return(out);
}

Mat segment_skeleton(Mat input){
	Mat bp = find_branchpoints(input);
	Mat bp_dil;
	dilate(bp, bp_dil, Mat(), Point(-1, -1), 3, 1, 1);
	Mat segments = input-bp_dil;

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
    findContours( segments, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

	Mat out = Mat::zeros(input.size(),CV_8UC3);
    for(unsigned int i=0; i < contours.size(); i++){
    	Scalar color( rand()&255, rand()&255, rand()&255 );
    	drawContours(out, contours, i, color, cv::FILLED);
    }
	return(out);
}
