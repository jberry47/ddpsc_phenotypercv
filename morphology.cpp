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
	vector<Mat> kernels(2);
    kernels[0] = (Mat_<int>(3, 3) <<
           -1, -1, -1,
           -1, 1, -1,
           0, 1, 0);

    kernels[1] = (Mat_<int>(3, 3) <<
           -1, -1, -1,
           -1, 1, 0,
           -1, 0, 1);

    Mat cur_ker, temp_ker;
    for(int ker=0;ker<=1;ker++){
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
	double length;
    //-- Keep only those contours that have a area > size
    for(unsigned int i=0; i < contours.size(); i++){
    	temp = Mat::zeros(input.size(),input.type());
    	temp_mask = Mat::zeros(input.size(),input.type());
    	drawContours(temp, contours, i, 255, cv::FILLED);
    	bitwise_and(temp,input,temp_mask);
    	length = sum(temp_mask/255)[0];
    	if(length > size){
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

Mat segment_skeleton(Mat input, bool colored=false){
	Mat bp = find_branchpoints(input);
	Mat bp_dil;
	dilate(bp, bp_dil, Mat(), Point(-1, -1), 3, 1, 1);
	Mat segments = input-bp_dil;

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
    findContours( segments, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

    Mat out;
    if(colored){
    	out = Mat::zeros(input.size(),CV_8UC3);
        for(unsigned int i=0; i < contours.size(); i++){
        	Scalar color( rand()&255, rand()&255, rand()&255 );
        	drawContours(out, contours, i, color, cv::FILLED);
        }
    }else{
    	out = Mat::zeros(input.size(),CV_8UC1);
        for(unsigned int i=0; i < contours.size(); i++){
        	drawContours(out, contours, i, 255, cv::FILLED);
        }
    }
	return(out);
}

Mat classify_skeleton(Mat skel, Mat tips){
	vector<vector<Point> > skel_contours;
	vector<Vec4i> skel_hierarchy;
    findContours( skel, skel_contours, skel_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

	vector<vector<Point> > tips_contours;
	vector<Vec4i> tips_hierarchy;
    findContours( tips, tips_contours, tips_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

    Mat kept = Mat::zeros(skel.size(),CV_8UC3);

    Mat temp_tip, temp_seg, band;
    double length;
    bool drawn = false;
    for(unsigned int seg=0; seg < skel_contours.size(); seg++){
      	for(unsigned int tip=0; tip < tips_contours.size(); tip++){
      		temp_tip = Mat::zeros(skel.size(),skel.type());
      		drawContours(temp_tip, tips_contours, tip, 255, cv::FILLED);

     		temp_seg = Mat::zeros(skel.size(),skel.type());
      		drawContours(temp_seg, skel_contours, seg, 255, cv::FILLED);

      		band = Mat::zeros(skel.size(),skel.type());
      		bitwise_and(temp_tip,temp_seg,band);
        	length = sum(band/255)[0];
      		if(length >0){
      			drawn = true;
      			drawContours(kept, skel_contours, seg, Scalar( 0, 255,0 ), cv::FILLED);
       		}
       	}
      	if(!drawn){
      		drawContours(kept, skel_contours, seg, Scalar( 255, 0,255 ), cv::FILLED);
      	}
      	drawn = false;
    }
   	return(kept);
}

vector<vector<double> > calc_leaf_curvatures(Mat classified_skel){
	vector<Mat> bgr(3);
	split(classified_skel,bgr);
	Mat leaves = bgr[1];

	vector<vector<Point> > skel_contours;
	vector<Vec4i> skel_hierarchy;
    findContours(leaves, skel_contours, skel_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

	vector<double> my_curv;
	vector<double> my_path;
	vector<double> my_ed;
	vector<vector<double> > my_data;
    Mat src;
    for(unsigned int seg=0; seg < skel_contours.size(); seg++){
    	src = Mat::zeros(classified_skel.size(),CV_8UC1);
    	drawContours(src, skel_contours, seg, 255, cv::FILLED);
    	Mat tips = find_endpoints(src);

    	vector<Point> cc;
    	for(int i=0;i<tips.rows;i++){
    		for(int j=0; j<tips.cols;j++){
    			if(tips.at<uchar>(i,j)==255){
    				cc.push_back(Point(i,j));
    			}
    		}
    	}

    	if(cc.size()==2){
        	Point p1 = cc[0];
        	Point p2 = cc[1];
        	double ed = sqrt(pow(p1.x-p2.x,2) + pow(p1.y-p2.y,2));
        	my_ed.push_back(ed);
        	double path_length = sum(src/255)[0];
        	my_path.push_back(path_length);
        	double curv = path_length/ed;
        	my_curv.push_back(curv);
    	}
    }
    my_data.push_back(my_ed);
    my_data.push_back(my_path);
    my_data.push_back(my_curv);
	return(my_data);
}
