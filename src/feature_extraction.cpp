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

int is_oof(Mat img){
	//-- Get contours of mask
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
    findContours( img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

    //-- Get contours of rectangular roi
    Mat src = Mat::zeros(img.size(),img.type())+255;

    vector<vector<Point> > contours_roi;
    vector<Vec4i> hierarchy_roi;
    findContours( src, contours_roi, hierarchy_roi, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

    int check = 0;
    //-- Keep only those contours that have a point inside roi
    for(unsigned int i=0; i < contours.size(); i++){
      	for(unsigned int j=0; j<contours[i].size(); j++){
      		int test = pointPolygonTest(contours_roi[0],Point2f(contours[i][j]),false);
      		if(test == 0){
      			check = 1;
      		}
       	}
    }
	return check;
}

vector<Point> keep_roi(Mat img,Point tl, Point br, Mat &mask){
	//-- Get contours of mask
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
    findContours( img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

    //-- Get contours of rectangular roi
    Mat src = Mat::zeros(img.size(),img.type());
    rectangle(src,tl,br,255,cv::FILLED);

    vector<vector<Point> > contours_roi;
    vector<Vec4i> hierarchy_roi;
    findContours( src, contours_roi, hierarchy_roi, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

    //-- Keep only those contours that have a point inside roi
    vector<Point> cc;
    Mat kept = Mat::zeros(img.size(),img.type());
    for(unsigned int i=0; i < contours.size(); i++){
      	for(unsigned int j=0; j<contours[i].size(); j++){
      		int test = pointPolygonTest(contours_roi[0],Point2f(contours[i][j]),false);
      		if(test==1 || test == 0){
      			for(unsigned int k=0; k<contours[i].size(); k++){
      				cc.push_back(contours[i][k]);
      			}
      			drawContours(kept, contours, i, 255, cv::FILLED);
      			break;
      		}
       	}
    }
	Mat kept_mask;
	bitwise_and(img,kept,kept_mask);

    mask = kept_mask;
	return cc;
}

float get_fd(Mat mask){
	//-- Need to remap the image to 2048x2048 so box counting can be used
	Mat img_bc;
	resize(mask, img_bc, Size(2048,2048), 0, 0, INTER_LINEAR);

	//-- Initializing variables
	double width = 2048.0;
	double p = log(width)/log(double(2.0));
	VectorXf N = VectorXf::Zero(int(p)+1);
	double sumImg = sum(img_bc)[0];
	N(int(p)) = sumImg;

	//-- Boxcounting
	double siz;
	double siz2;
	float running_sum;
    for (int g = int(p)-1; g > 0; g--){
    	siz = pow(2.0, double(p-g));
    	siz2 = round(siz/2.0);
    	running_sum = 0;
    	for (int i = 0; i < int(width-siz+1); i = i+int(siz)){
    		for (int j = 0; j < int(width-siz+1); j = j+int(siz)){
    			img_bc.at<uchar>(i,j) = (bool(img_bc.at<uchar>(i,j)) || bool(img_bc.at<uchar>(i+siz2,j))
    				|| bool(img_bc.at<uchar>(i,j+siz2)) || bool(img_bc.at<uchar>(i+siz2,j+siz2)));
    			running_sum = running_sum+float(img_bc.at<uchar>(i,j));
    		}
    	}
    	N(g) = running_sum;
	}
    N = N.colwise().reverse().eval();

    //-- Getting bin sizes
    VectorXf R = VectorXf::Zero(int(p)+1);
    R(0) = 1.0;
    for (int k = 1; k < R.size(); k++){
    	R(k) = pow(2.0, double(k));
    }

    //-- Calculating log-log slopes
	float slope [R.size()-1];
	for(int i=1;i < R.size()-1 ;i++){
		slope[i] = (log10(N(i+1))-log10(N(i)))/(log10(R(i+1))-log10(R(i)));
	}

	//-- Getting average slope (fractal dimension)
	float sum = 0.0, average;
	int s_count =0;
	for(int i=1; i < R.size()-1; i++){
		if(-slope[i] < 2 && -slope[i] > 0){
			sum += -slope[i];
			s_count++;
		}
	}
	average = sum / s_count;
	return average;
}

vector<double> get_shapes(vector<Point> cc,Mat mask){
    //-- Get measurements
    Moments mom = moments(mask,true);
    double area = mom.m00;
	if(area>0){
	    vector<Point>hull;
	    convexHull( Mat(cc), hull, false );
	    double hull_verticies = hull.size();
	    double hull_area = contourArea(Mat(hull));
	    double solidity = area/hull_area;
	    double perimeter = arcLength(Mat(cc),false);
	    double cmx = mom.m10 / mom.m00;
	    double cmy = mom.m01 / mom.m00;
	    Rect boundRect = boundingRect( cc );
	    double width = boundRect.width;
	    double height = boundRect.height;
	    double circ = 4*M_PI*area/(perimeter*perimeter);
	    double angle = -1;
	    double ex = -1;
	    double ey = -1;
	    double emajor = -1;
	    double eminor = -1;
	    double eccen = -1;
	    double round = -1;
	    double ar = -1;
	    if(cc.size() >= 6){
	        Mat pointsf;
	    	Mat(cc).convertTo(pointsf, CV_32F);
	   	    RotatedRect ellipse = fitEllipse(pointsf);
	   	    angle = ellipse.angle;
	  	    ex = ellipse.center.x;
	   	    ey = ellipse.center.y;
	   	    if(ellipse.size.height > ellipse.size.width){
	   	    	emajor = ellipse.size.height;
	   	    	eminor = ellipse.size.width;
	   	    }else{
	   	    	eminor = ellipse.size.height;
	  	   	    emajor = ellipse.size.width;
	   	    }
	   	    eccen = sqrt((1- eminor / emajor)*2);
	   	    round = eminor/emajor;
	   	    ar = emajor/eminor;
	    }
	    float fd = get_fd(mask);
	    double oof = is_oof(mask);
	    double shapes[20] = {area,hull_area,solidity,perimeter,width,height,cmx,cmy,hull_verticies,ex,ey,emajor,eminor,angle,eccen,circ,round,ar,fd,oof};
	    vector<double> shapes_v(shapes,shapes+20);
	    return shapes_v;
	}else{
	    double shapes[20] = {0, 0, -nan("1"), 0, 0, 0, -nan("1"), -nan("1"), 0, -1, -1, -1, -1, -1, -1, -nan("1"), -1, -1, -nan("1"), 0};
	    vector<double> shapes_v(shapes,shapes+20);
	    return shapes_v;
	}
}

Mat get_color(Mat img,Mat mask){
	Mat composite;
	cvtColor(img,composite,COLOR_BGR2HSV);
    vector<Mat> channels1;
    split(composite, channels1);
    Mat hist;
	int dims = 1; // Only 1 channel, the hue channel
	int histSize = 180; // 180 bins, actual range is 0-360.
	float hranges[] = { 0, 180 }; // hue varies from 0 to 179, see cvtColor
	const float *ranges = {hranges};

	//-- Compute the histogram
	calcHist(&channels1[0],1,0,mask,hist, dims, &histSize, &ranges	,true ,false);
	return hist;
}

Mat get_nir(Mat img,Mat mask){
    Mat hist;
	int dims = 1;
	int histSize = 255;
	float hranges[] = { 0, 255 };
	const float *ranges = {hranges};

	//-- Compute the histogram
	calcHist(&img,1,0,mask,hist, dims, &histSize, &ranges	,true ,false);
	return hist;
}

